import os
os.environ["CUDA_VISIBLE_DEVICES"] = ''
import keras

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.layers.modeling.embed_nd import EmbedND
from keras_hub.src.layers.modeling.flux_lastlayer import LastLayer
from keras_hub.src.layers.modeling.double_stream_block import DoubleStreamBlock
from keras_hub.src.layers.modeling.single_stream_block import SingleStreamBlock
from keras_hub.src.layers.modeling.f_net_encoder import FNetEncoder
from keras_hub.src.layers.modeling.mlp_embedder import MLPEmbedder
from keras_hub.src.layers.modeling.position_embedding import PositionEmbedding
from keras_hub.src.layers.modeling.reversible_embedding import (
    ReversibleEmbedding,
)
from keras_hub.src.models.backbone import Backbone
from keras_hub.src.utils.keras_utils import gelu_approximate
from .flux_timestep_embedding import timestep_embedding


def f_net_kernel_initializer(stddev=0.02):
    return keras.initializers.RandomNormal(stddev=stddev)


def f_net_bias_initializer(stddev=0.02):
    return keras.initializers.RandomNormal(stddev=stddev)


@keras_hub_export("keras_hub.models.FuxBackbone")
class FluxBackbone(Backbone):
    """A FNet encoder network.

    This class implements a bi-directional Fourier Transform-based encoder as
    described in ["FNet: Mixing Tokens with Fourier Transforms"](https://arxiv.org/abs/2105.03824).
    It includes the embedding lookups and `keras_hub.layers.FNetEncoder` layers,
    but not the masked language model or next sentence prediction heads.

    The default constructor gives a fully customizable, randomly initialized
    FNet encoder with any number of layers and embedding dimensions. To
    load preset architectures and weights, use the `from_preset()` constructor.

    Note: unlike other models, FNet does not take in a `"padding_mask"` input,
    the `"<pad>"` token is handled equivalently to all other tokens in the input
    sequence.

    Disclaimer: Pre-trained models are provided on an "as is" basis, without
    warranties or conditions of any kind.

    Args:
        vocabulary_size: int. The size of the token vocabulary.
        num_layers: int. The number of FNet layers.
        hidden_dim: int. The size of the FNet encoding and pooler layers.
        intermediate_dim: int. The output dimension of the first Dense layer in
            a two-layer feedforward network for each FNet layer.
        dropout: float. Dropout probability for the embeddings and FNet encoder.
        max_sequence_length: int. The maximum sequence length that this encoder
            can consume. If None, `max_sequence_length` uses the value from
            sequence length. This determines the variable shape for positional
            embeddings.
        num_segments: int. The number of types that the 'segment_ids' input can
            take.
        dtype: string or `keras.mixed_precision.DTypePolicy`. The dtype to use
            for model computations and weights. Note that some computations,
            such as softmax and layer normalization, will always be done at
            float32 precision regardless of dtype.

    Examples:
    ```python
    input_data = {
        "token_ids": np.ones(shape=(1, 12), dtype="int32"),
        "segment_ids": np.array([[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0]]),
    }

    # Pretrained BERT encoder.
    model = keras_hub.models.FNetBackbone.from_preset("f_net_base_en")
    model(input_data)

    # Randomly initialized FNet encoder with a custom config.
    model = keras_hub.models.FNetBackbone(
        vocabulary_size=32000,
        num_layers=4,
        hidden_dim=256,
        intermediate_dim=512,
        max_sequence_length=128,
    )
    model(input_data)
    ```
    """

    def __init__(
        self,
            in_channels,
            vec_in_dim,
            context_in_dim,
            hidden_size,
            mlp_ratio,
            num_heads,
            depth,
            depth_single_blocks,
            axes_dim,
            theta,
            qkv_bias,
            guidance_embed,
        **kwargs,
    ):
        super().__init__()
        self.name = 'Flux'
        self._build_shapes_dict = None
        self.compiled = False
        self._trainable = True
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.axes_dim = axes_dim
        self.mlp_ratio = mlp_ratio
        self.depth = depth
        self.depth_single_blocks = depth_single_blocks
        self.axes_dim = axes_dim
        self.theta = theta
        self.vec_in_dim = vec_in_dim
        self.qkv_bias = qkv_bias
        self.guidance_embed = guidance_embed
        self.context_in_dim = context_in_dim
        if self.hidden_size % self.num_heads != 0:
            raise ValueError(
                f"Hidden size {self.hidden_size} must be divisible by num_heads {self.num_heads}"
            )
        pe_dim = self.hidden_size // self.num_heads
        if sum(self.axes_dim) != pe_dim:
            raise ValueError(f"Got {self.axes_dim} but expected positional dim {pe_dim}")
        self.num_heads = self.num_heads
        self.pe_embedder = EmbedND(dim=pe_dim, theta=self.theta, axes_dim=self.axes_dim)
        #self.img_in = keras.layers.Dense(self.in_channels, self.hidden_size)#, bias=True)
        self.img_in = keras.layers.Dense(self.hidden_size)#, bias=True)
        self.time_in = MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size)
        self.vector_in = MLPEmbedder(vec_in_dim, self.hidden_size)
        self.guidance_in = (
            MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size) if self.guidance_embed else keras.ops.identity(1)
        )
        #self.txt_in = keras.layers.Dense(self.context_in_dim, self.hidden_size)
        self.txt_in = keras.layers.Dense(self.hidden_size)

        self.double_blocks = [#keras.Sequential(

                DoubleStreamBlock(
                    self.hidden_size,
                    self.num_heads,
                    mlp_ratio=self.mlp_ratio,
                    qkv_bias=self.qkv_bias,
                )
                for _ in range(self.depth)
            ]
       # )

        self.single_blocks = [#keras.Sequential(

                SingleStreamBlock(self.hidden_size, self.num_heads, mlp_ratio=self.mlp_ratio)
                for _ in range(self.depth_single_blocks)
            ]
        #)

        self.final_layer = LastLayer(self.hidden_size, 1, self.out_channels)


    def call(        self,
        img,
        img_ids,
        txt,
        txt_ids,
        timesteps,
        y,
        guidance):
        if img.ndim != 3 or txt.ndim != 3:
            raise ValueError("Input img and txt tensors must have 3 dimensions.")

            # running on sequences img
        img = self.img_in(img)
        timestep_embedding(timesteps, 256)
        vec = self.time_in(timestep_embedding(timesteps, 256))
        #TODO: problem with the guidance not being a list
        #if self.guidance_embed:
        #    if guidance is None:
        #        raise ValueError("Didn't get guidance strength for guidance distilled model.")
        #    vec = vec + self.guidance_in(timestep_embedding(guidance, 256))

        vec = vec + self.vector_in(y)
        txt = self.txt_in(txt)

        ids = keras.layers.Concatenate( axis=1)((txt_ids, img_ids))
        pe = self.pe_embedder(ids)

        for block in self.double_blocks:
            img, txt = block(img=img, txt=txt, vec=vec, pe=pe)

        img = keras.layers.Concatenate(axis= 1)((txt, img))
        for block in self.single_blocks:
            img = block(img, vec=vec, pe=pe)
        img = img[:, txt.shape[1]:, ...]

        img = self.final_layer(img, vec)  # (N, T, patch_size ** 2 * out_channels)
        return img

    def predict(        self,
        img,
        img_ids,
        txt,
        txt_ids,
        timesteps,
        y,
        guidance):
        if img.ndim != 3 or txt.ndim != 3:
            raise ValueError("Input img and txt tensors must have 3 dimensions.")

            # running on sequences img
        img = self.img_in(img)
        timestep_embedding(timesteps, 256)
        vec = self.time_in(timestep_embedding(timesteps, 256))
        #TODO: problem with the guidance not being a list
        #if self.guidance_embed:
        #    if guidance is None:
        #        raise ValueError("Didn't get guidance strength for guidance distilled model.")
        #    vec = vec + self.guidance_in(timestep_embedding(guidance, 256))

        vec = vec + self.vector_in(y)
        txt = self.txt_in(txt)

        ids = keras.layers.Concatenate( axis=1)((txt_ids, img_ids))
        pe = self.pe_embedder(ids)

        for block in self.double_blocks:
            img, txt = block(img=img, txt=txt, vec=vec, pe=pe)

        img = keras.layers.Concatenate(axis= 1)((txt, img))
        for block in self.single_blocks:
            img = block(img, vec=vec, pe=pe)
        img = img[:, txt.shape[1]:, ...]

        img = self.final_layer(img, vec)  # (N, T, patch_size ** 2 * out_channels)
        return img

    def get_config(self):
        config = super().get_config()
        config.update(
             {
                 "in_channels": 64,
                 "vec_in_dim": 768,
                 "context_in_dim": 4096,
                 "hidden_size": 3072,
                 "mlp_ratio": 4.0,
                 "num_heads": 24,
                 "depth": 19,
                 "depth_single_blocks": 38,
                 "axes_dim": [16, 56, 56],
                 "theta": 10_000,
                 "qkv_bias": True,
                 "guidance_embed": True,
             }
         )
        return config
