import keras
import pytest
from keras import ops
from einops import rearrange, repeat
from keras_hub.src.models.flux.flux_backbone import FluxBackbone
from keras_hub.src.tests.test_case import TestCase
from transformers import (CLIPTextModel, CLIPTokenizer, T5EncoderModel,
                          T5Tokenizer)
import math
from  .flux_HFEmbedder import HFEmbedder
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
        "img": img,
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
        img = keras.random.normal((3,h,w,bs), mean=0.0, stddev=1.0, dtype=None, seed=None)
        img = rearrange(img, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
        t5 = self.load_t5( max_length=256 if is_schnell else 512)
        clip = self.load_clip()
        num_steps = 1#int(st.number_input("Number of steps", min_value=1, value=(4 if is_schnell else 50)))
        self.init_kwargs = {
            "in_channels":64,
            "vec_in_dim":int(768/256),#768
            "context_in_dim":int(768/256),#4096
            "hidden_size":12*16,#3072,
            "mlp_ratio":4.0,
            "num_heads":int(24/2),
            "depth":9,#19,
            "depth_single_blocks":18,#38,
            "axes_dim":[4, 6,6 ],#[16/8, 56/8, 56/8],
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
        print('y',vec.shape)
        if vec.shape[0] == 1 and bs > 1:
            vec = repeat(vec, "1 ... -> bs ...", bs=bs)
        inp = prepare(t5=t5, clip=clip, img=x, prompt='a photo of a forest with mist swirling around the tree trunks. The word "FLUX" is painted over it in big, red brush strokes with visible texture')
        self.input_data = {
            "img": inp['img'],
            #"token_ids": ops.ones((h, w), dtype="int32"),
        "img_ids": inp['img_ids'],
        "txt": inp['txt'],
        "txt_ids": inp['txt_ids'],
        "y": inp['vec'],
        "timesteps":timesteps,
        "guidance":guidance
        }

    def test_backbone_basics(self):
        #print(self.input_data)
        import os
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

        self.run_backbone_test(
            cls=FluxBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output_shape=(2, 5, 16),
        )
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
