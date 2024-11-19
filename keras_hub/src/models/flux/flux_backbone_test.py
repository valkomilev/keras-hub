import os
os.environ["CUDA_VISIBLE_DEVICES"] = ''
import keras
import pytest
from keras import ops
from einops import rearrange, repeat
from keras_hub.src.models.flux.flux_backbone import FluxBackbone
from huggingface_hub import hf_hub_download
from keras_hub.src.tests.test_case import TestCase
from transformers import (CLIPTextModel, CLIPTokenizer, T5EncoderModel,
                          T5Tokenizer)
import math
import torch
from safetensors.torch import load_file as load_sft

from  .flux_HFEmbedder import HFEmbedder

class AttnBlock(keras.Model):
    def __init__(self, in_channels: int):
        super().__init__()
        self.in_channels = in_channels

        self.norm = keras.layers.GroupNormalization(groups=32, epsilon=1e-6)

        self.q = keras.layers.Conv2D( in_channels, kernel_size=1)
        self.k = keras.layers.Conv2D( in_channels, kernel_size=1)
        self.v = keras.layers.Conv2D( in_channels, kernel_size=1)
        self.proj_out = keras.layers.Conv2D( in_channels, kernel_size=1)

    def attention(self, h_) :
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        b, c, h, w = q.shape
        q = rearrange(q, "b c h w -> b 1 (h w) c").contiguous()
        k = rearrange(k, "b c h w -> b 1 (h w) c").contiguous()
        v = rearrange(v, "b c h w -> b 1 (h w) c").contiguous()
        h_ = torch.nn.functional.scaled_dot_product_attention(q, k, v)

        return rearrange(h_, "b 1 (h w) c -> b c h w", h=h, w=w, c=c, b=b)

    def forward(self, x):
        return x + self.proj_out(self.attention(x))



def swish(x) :
    return x * keras.activations.sigmoid(x)


class AutoEncoderParams:
    def __init__(self,resolution,in_channels,ch,out_ch,ch_mult,num_res_blocks,z_channels,scale_factor,shift_factor):
        self.resolution = resolution
        self.in_channels = in_channels
        self.ch = ch
        self.out_ch = out_ch
        self.ch_mult = ch_mult
        self.num_res_blocks = num_res_blocks
        self.z_channels = z_channels
        self.scale_factor = scale_factor
        self.shift_factor = shift_factor

ae_params=AutoEncoderParams(
            resolution=256,
            in_channels=3,
            ch=128,
            out_ch=3,
            ch_mult=[1, 2, 4, 4],
            num_res_blocks=2,
            z_channels=16,
            scale_factor=0.3611,
            shift_factor=0.1159,
        )
class ResnetBlock(keras.Model):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels

        self.norm1 = keras.layers.GroupNormalization(groups=32, epsilon=1e-6)
        self.conv1 = keras.layers.Conv2D( out_channels, kernel_size=3, strides=(1,1),padding="same")
        self.norm2 = keras.layers.GroupNormalization(groups=32, epsilon=1e-6)
        self.conv2 = keras.layers.Conv2D(out_channels, kernel_size=3,  strides=(1,1),padding="same")
        if self.in_channels != self.out_channels:
            self.nin_shortcut = keras.layers.Conv2D(out_channels, kernel_size=1,  strides=(1,1),padding="same")

    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = swish(h)
        h = self.conv1(h)

        h = self.norm2(h)
        h = swish(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            x = self.nin_shortcut(x)

        return x + h


class Downsample(keras.Model):
    def __init__(self, in_channels: int):
        super().__init__()
        # no asymmetric padding in torch conv, must do it ourselves
        self.conv =  keras.layers.Conv2D(in_channels, kernel_size=3, strides=(2,2))

    def forward(self, x):
        pad = (0, 1, 0, 1)
        x = ops.pad(pad, mode="constant", constant_values=0)(x)
        x = self.conv(x)
        return x


class Upsample(keras.Model):
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv =  keras.layers.Conv2D( in_channels, kernel_size=3,strides=(1,1))

    def forward(self, x):
        x = keras.layers.UpSampling2D(2, interpolation="nearest")(x)
        x = self.conv(x)
        return x


class Encoder(keras.Model):
    def __init__(
        self,
        resolution: int,
        in_channels: int,
        ch: int,
        ch_mult: list[int],
        num_res_blocks: int,
        z_channels: int,
    ):
        super().__init__()
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        # downsampling
        self.conv_in = keras.layers.Conv2D(in_channels, kernel_size=3, strides=(1,2))

        curr_res = resolution
        in_ch_mult = (1,) + tuple(ch_mult)
        self.in_ch_mult = in_ch_mult
        self.down = []
        block_in = self.ch
        for i_level in range(self.num_resolutions):
            block = []
            attn = []
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out))
                block_in = block_out
            down = keras.Model()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = keras.Model()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in)

        # end
        self.norm_out = keras.layers.GroupNormalization(groups=32, epsilon=1e-6)
        self.conv_out =keras.layers.Conv2D(2 * z_channels, kernel_size=3, strides=(1,1))

    def forward(self, x):
        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1])
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)
        # end
        h = self.norm_out(h)
        h = swish(h)
        h = self.conv_out(h)
        return h


class Decoder(keras.Model):
    def __init__(
        self,
        ch: int,
        out_ch: int,
        ch_mult: list[int],
        num_res_blocks: int,
        in_channels: int,
        resolution: int,
        z_channels: int,
    ):
        super().__init__()
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.ffactor = 2 ** (self.num_resolutions - 1)

        # compute in_ch_mult, block_in and curr_res at lowest res
        block_in = ch * ch_mult[self.num_resolutions - 1]
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        self.z_shape = (1, z_channels, curr_res, curr_res)

        # z to block_in
        self.conv_in = keras.layers.Conv2D( block_in, kernel_size=3, strides=(1,1))

        # middle
        self.mid = keras.Model()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in)

        # upsampling
        self.up = []
        for i_level in reversed(range(self.num_resolutions)):
            block = []
            attn = []
            block_out = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks + 1):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out))
                block_in = block_out
            up = keras.Model()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in)
                curr_res = curr_res * 2
            self.up.insert(0, up)  # prepend to get consistent order

        # end

        self.norm_out = keras.layers.GroupNormalization(groups=32, epsilon=1e-6)
        #self.norm_out = nn.GroupNorm(num_groups=32, num_channels=block_in, eps=1e-6, affine=True)
        self.conv_out = keras.layers.Conv2D( out_ch, kernel_size=3, strides=(1,1))
        #self.conv_out = nn.Conv2d(block_in, out_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, z) :
        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        h = self.norm_out(h)
        h = swish(h)
        h = self.conv_out(h)
        return h


class DiagonalGaussian(keras.Model):
    def __init__(self, sample: bool = True, chunk_dim: int = 1):
        super().__init__()
        self.sample = sample
        self.chunk_dim = chunk_dim

    def forward(self, z):
        mean, logvar = torch.chunk(z, 2, dim=self.chunk_dim)
        if self.sample:
            std = torch.exp(0.5 * logvar)
            return mean + std * torch.randn_like(mean)
        else:
            return mean

class AutoEncoder(keras.Model):
    def __init__(self, params: AutoEncoderParams):
        super().__init__()
        self.encoder = Encoder(
            resolution=params.resolution,
            in_channels=params.in_channels,
            ch=params.ch,
            ch_mult=params.ch_mult,
            num_res_blocks=params.num_res_blocks,
            z_channels=params.z_channels,
        )
        self.decoder = Decoder(
            resolution=params.resolution,
            in_channels=params.in_channels,
            ch=params.ch,
            out_ch=params.out_ch,
            ch_mult=params.ch_mult,
            num_res_blocks=params.num_res_blocks,
            z_channels=params.z_channels,
        )
        self.reg = DiagonalGaussian()

        self.scale_factor = params.scale_factor
        self.shift_factor = params.shift_factor

    def encode(self, x):
        z = self.reg(self.encoder(x))
        z = self.scale_factor * (z - self.shift_factor)
        return z

    def decode(self, z):
        z = z / self.scale_factor + self.shift_factor
        return self.decoder(z)

    def forward(self, x):
        return self.decode(self.encode(x))

def unpack(x, height, width):
    return rearrange(
        x,
        "b (h w) (c ph pw) -> b c (h ph) (w pw)",
        h=math.ceil(height / 16),
        w=math.ceil(width / 16),
        ph=2,
        pw=2,
    )

def print_load_warning(missing: list[str], unexpected: list[str]) -> None:
    if len(missing) > 0 and len(unexpected) > 0:
        print(f"Got {len(missing)} missing keys:\n\t" + "\n\t".join(missing))
        print("\n" + "-" * 79 + "\n")
        print(f"Got {len(unexpected)} unexpected keys:\n\t" + "\n\t".join(unexpected))
    elif len(missing) > 0:
        print(f"Got {len(missing)} missing keys:\n\t" + "\n\t".join(missing))
    elif len(unexpected) > 0:
        print(f"Got {len(unexpected)} unexpected keys:\n\t" + "\n\t".join(unexpected))

def load_ae(name: str, device, hf_download: bool = True) -> AutoEncoder:
    ckpt_path = "ae.safetensors"#configs[name].ae_path
    repo_id = "black-forest-labs/FLUX.1-schnell",
    repo_flow = "flux1-schnell.safetensors",
    repo_ae = "ae.safetensors",
    # if (
    #     ckpt_path is None
    #     and repo_id is not None
    #     and repo_ae is not None
    #     and hf_download
    # ):
    print(repo_id, repo_ae)
    ckpt_path = hf_hub_download(repo_id[0], repo_ae[0])
    #exit(0)
    # Loading the autoencoder
    print("Init AE")
    ae = AutoEncoder(ae_params)

    if ckpt_path is not None:
        sd = load_sft(ckpt_path, device=str(device))
        print("load_state_dict")
        print(ae.decoder.summary())
        missing, unexpected = ae.load_state_dict(sd, strict=False, assign=True)
        print('missing',len(missing))
        print_load_warning(missing, unexpected)
    return ae

def denoise(
    model,
    # model input
    img,
    img_ids,
    txt,
    txt_ids,
    vec,
    # sampling parameters
    timesteps: list[float],
    guidance: float = 4.0,
):
    # this is ignored for schnell
    guidance_vec = ops.full((img.shape[0],), guidance, dtype=img.dtype)
    for t_curr, t_prev in zip(timesteps[:-1], timesteps[1:]):
        t_vec = ops.full((img.shape[0],), t_curr)
        pred = model(
            img=img,
            img_ids=img_ids,
            txt=txt,
            txt_ids=txt_ids,
            y=vec,
            timesteps=t_vec,
            guidance=guidance_vec,
        )

        img = img + (t_prev - t_curr) * pred

    return img
def get_noise(
    num_samples: int,
    height: int,
    width: int
):
    return keras.random.uniform(
        (num_samples,
        16,
        # allow for packing
        2 * math.ceil(height / 16),
        2 * math.ceil(width / 16))
    )
def prepare(t5: HFEmbedder, clip: HFEmbedder, img, prompt: str | list[str]) :
    bs, c, h, w = img.shape
    if bs == 1 and not isinstance(prompt, str):
        bs = len(prompt)

    img = rearrange(img, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
    if img.shape[0] == 1 and bs > 1:
        img = repeat(img, "1 ... -> bs ...", bs=bs)

    img_ids = keras.ops.zeros((h // 2, w // 2, 3))
    img_ids[..., 1] = img_ids[..., 1] + keras.ops.arange(h // 2)[:, None]
    img_ids[..., 2] = img_ids[..., 2] + keras.ops.arange(w // 2)[None, :]
    img_ids = repeat(img_ids, "h w c -> b (h w) c", b=bs)

    if isinstance(prompt, str):
        prompt = [prompt]
    txt = t5(prompt)
    if txt.shape[0] == 1 and bs > 1:
        txt = repeat(txt, "1 ... -> bs ...", bs=bs)
    txt_ids = keras.ops.zeros((bs, txt.shape[1], 3))

    vec = clip(prompt)
    if vec.shape[0] == 1 and bs > 1:
        vec = repeat(vec, "1 ... -> bs ...", bs=bs)

    return {
        "img": img.to(img.device),
        "img_ids": img_ids.to(img.device),
        "txt": txt.to(img.device),
        "txt_ids": txt_ids.to(img.device),
        "vec": vec.to(img.device),
    }

class FluxBackboneTest(TestCase):

    def load_t5( self,max_length: int = 512) -> HFEmbedder:
        # max length 64, 128, 256 and 512 should work (if your sequence is short enough)
        return HFEmbedder("google/t5-v1_1-base", max_length=max_length)

    def load_clip(self) -> HFEmbedder:
        return HFEmbedder("openai/clip-vit-large-patch14", max_length=77)


    def setUp(self):
        h = 600
        w = 600
        bs = 4
        prompt = 'some text'
        is_schnell  = False
        x = get_noise(
            1,
            h,
            w
        )
        self.height = h
        self.width = w
        img = keras.random.normal((3,h,w,bs), mean=0.0, stddev=1.0, dtype=None, seed=None)
        img = rearrange(img, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
        t5 = self.load_t5( max_length=256 if is_schnell else 512)
        self.ae = load_ae('flux-schnell', device="cpu" )
        clip = self.load_clip()
        #print('self.ae',self.ae)
        num_steps = 1#int(st.number_input("Number of steps", min_value=1, value=(4 if is_schnell else 50)))
        self.init_kwargs = {
            "in_channels":64,
            "vec_in_dim":int(768/256),#768
            "context_in_dim":int(768/256),#4096
            "hidden_size":24*32,#3072,
            "mlp_ratio":4.0,
            "num_heads":24,#int(24/2),
            "depth":19,#19,
            "depth_single_blocks":38,#38,
            "axes_dim":[16/4, 56/4, 56/4],
            "theta":10_000,
            "qkv_bias":True,
            "guidance_embed":False,
        }
        timesteps =  keras.ops.linspace(1, 0, num_steps + 1)

        guidance = 4.0
        img_ids = keras.ops.zeros((h // 2, w // 2, 3))
        img_ids[..., 1] = img_ids[..., 1] + keras.ops.arange(h // 2)[:, None]
        img_ids[..., 2] = img_ids[..., 2] + keras.ops.arange(w // 2)[None, :]
        img_ids = repeat(img_ids, "h w c -> b (h w) c", b=bs)
        txt = t5(prompt)
        if txt.shape[0] == 1 and bs > 1:
            txt = repeat(txt, "1 ... -> bs ...", bs=bs)
        txt_ids = keras.ops.zeros((bs, txt.shape[1], 3))

        vec = clip(prompt)
        if vec.shape[0] == 1 and bs > 1:
            vec = repeat(vec, "1 ... -> bs ...", bs=bs)
        x = x.to('cpu')
        timesteps = timesteps.to('cpu')

        inp = prepare(t5=t5, clip=clip, img=x, prompt='a photo of a forest with mist swirling around the tree trunks. The word "FLUX" is painted over it in big, red brush strokes with visible texture')

        self.input_data = {
            "img": inp['img'],
            #"token_ids": ops.ones((h, w), dtype="int32"),
        "img_ids": inp['img_ids'],
        "txt": inp['txt'],
        "txt_ids": inp['txt_ids'],
        "vec": inp['vec'],
        "timesteps":timesteps,
        "guidance":guidance
        }

    def test_backbone_basics(self):
        #print(self.input_data)
        import os
        from PIL import Image
        import io
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        backbone = FluxBackbone(**self.init_kwargs)
        x = denoise(backbone,**self.input_data)

        x = unpack(x.float(), self.height, self.width)
        # torch.save(x, "tensor.pt")
        # # Save to io.BytesIO buffer
        # buffer = io.BytesIO()
        # torch.save(x, buffer)
        # exit(0)
        x = self.ae.decode(x)

        x = x.clamp(-1, 1)
        #x = embed_watermark(x.float())
        print("x.shape",x.shape)
        #x = rearrange(x[0], "c h w -> h w c")
        print((127.5 * (x + 1.0)).cpu().byte().numpy().shape)
        print("x.shape",x.shape)
        img = Image.fromarray((127.5 * (x[0] + 1.0)).cpu().byte().numpy())
        import matplotlib.image
        # print('img.shape',img.shape)

        matplotlib.image.imsave('img.png', img)
        #backbone = backbone.to('cpu')

        # Check serialization (without a full save).
        #self.run_serialization_test(backbone)
        #print('test1')
        # Call model eagerly.
        #output = backbone(**self.input_data)
        # self.run_vision_backbone_test(
        #     cls=FluxBackbone,
        #     init_kwargs=self.init_kwargs,
        #     input_data=self.input_data,
        #     expected_output_shape=(2, 1444, 64),
        # )
    #
    # @pytest.mark.large
    # def test_saved_model(self):
    #     self.run_model_saving_test(
    #         cls=GemmaBackbone,
    #         init_kwargs=self.init_kwargs,
    #         input_data=self.input_data,
    #     )
    #
    # @pytest.mark.kaggle_key_required
    # @pytest.mark.extra_large
    # def test_smallest_preset(self):
    #     # TODO: Fails with OOM on current GPU CI
    #     self.run_preset_test(
    #         cls=GemmaBackbone,
    #         preset="gemma_2b_en",
    #         input_data={
    #             "token_ids": ops.array([[651, 4320, 8426, 25341, 235265]]),
    #             "padding_mask": ops.ones((1, 5), dtype="int32"),
    #         },
    #         expected_output_shape=(1, 5, 2048),
    #         # The forward pass from a preset should be stable!
    #         expected_partial_output=ops.array(
    #             [1.073359, 0.262374, 0.170238, 0.605402, 2.336161]
    #         ),
    #     )
    #
    # @pytest.mark.kaggle_key_required
    # @pytest.mark.extra_large
    # def test_all_presets(self):
    #     for preset in GemmaBackbone.presets:
    #         self.run_preset_test(
    #             cls=GemmaBackbone,
    #             preset=preset,
    #             input_data=self.input_data,
    #         )
    #
    # def test_architecture_characteristics(self):
    #     model = GemmaBackbone(**self.init_kwargs)
    #     self.assertEqual(model.count_params(), 3216)
    #     self.assertEqual(len(model.layers), 6)
    #
    # def test_distribution(self):
    #     if keras.backend.backend() != "jax":
    #         self.skipTest("`ModelParallel` testing requires the Jax backend.")
    #     devices = keras.distribution.list_devices("CPU")
    #     if len(devices) == 1:
    #         self.skipTest("`ModelParallel` testing requires multiple devices.")
    #     device_mesh = keras.distribution.DeviceMesh(
    #         shape=(1, len(devices)),
    #         axis_names=("batch", "model"),
    #         devices=devices,
    #     )
    #
    #     layout_map = GemmaBackbone.get_layout_map(device_mesh)
    #     distribution = keras.distribution.ModelParallel(layout_map=layout_map)
    #     with distribution.scope():
    #         model = GemmaBackbone(**self.init_kwargs)
    #
    #     for w in model.weights:
    #         if "token_embedding/embeddings" in w.path:
    #             self.assertEqual(
    #                 tuple(w.value.sharding.spec), ("model", "batch")
    #             )
    #         if "attention/query/kernel" in w.path:
    #             self.assertEqual(
    #                 tuple(w.value.sharding.spec), ("model", "batch", None)
    #             )
    #         if "attention/key/kernel" in w.path:
    #             self.assertEqual(
    #                 tuple(w.value.sharding.spec), ("model", "batch", None)
    #             )
    #         if "attention/value/kernel" in w.path:
    #             self.assertEqual(
    #                 tuple(w.value.sharding.spec), ("model", "batch", None)
    #             )
    #         if "attention/attention_output/kernel" in w.path:
    #             self.assertEqual(
    #                 tuple(w.value.sharding.spec), ("model", None, "batch")
    #             )
    #         if "ffw_gating/kernel" in w.path:
    #             self.assertEqual(
    #                 tuple(w.value.sharding.spec), ("batch", "model")
    #             )
    #         if "ffw_gating_2/kernel" in w.path:
    #             self.assertEqual(
    #                 tuple(w.value.sharding.spec), ("batch", "model")
    #             )
    #         if "ffw_linear" in w.path:
    #             self.assertEqual(
    #                 tuple(w.value.sharding.spec), ("model", "batch")
    #             )
    #
    # def test_distribution_with_lora(self):
    #     if keras.backend.backend() != "jax":
    #         self.skipTest("`ModelParallel` testing requires the Jax backend.")
    #     devices = keras.distribution.list_devices("CPU")
    #     if len(devices) == 1:
    #         self.skipTest("`ModelParallel` testing requires multiple devices.")
    #     device_mesh = keras.distribution.DeviceMesh(
    #         shape=(1, len(devices)),
    #         axis_names=("batch", "model"),
    #         devices=devices,
    #     )
    #
    #     layout_map = GemmaBackbone.get_layout_map(device_mesh)
    #     distribution = keras.distribution.ModelParallel(device_mesh, layout_map)
    #     with distribution.scope():
    #         model = GemmaBackbone(**self.init_kwargs)
    #         model.enable_lora(rank=4)
    #
    #     for w in model.weights:
    #         if "attention/query/lora_kernel_a" in w.path:
    #             self.assertEqual(
    #                 tuple(w.value.sharding.spec), (None, None, None)
    #             )
    #         if "attention/query/lora_kernel_b" in w.path:
    #             self.assertEqual(tuple(w.value.sharding.spec), (None, None))
    #         if "attention/value/lora_kernel_a" in w.path:
    #             self.assertEqual(
    #                 tuple(w.value.sharding.spec), (None, None, None)
    #             )
    #         if "attention/value/lora_kernel_b" in w.path:
    #             self.assertEqual(tuple(w.value.sharding.spec), (None, None))
    #
