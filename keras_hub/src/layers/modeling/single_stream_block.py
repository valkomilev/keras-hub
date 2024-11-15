import keras
from keras import ops

from einops import rearrange
from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.utils.keras_utils import clone_initializer
from keras_hub.src.layers.modeling.flux_math import attention

class ModulationOut:
    shift: None
    scale: None
    gate: None


class Modulation(keras.layers.Layer):
    def __init__(self, dim: int, double: bool):
        super().__init__()
        self.is_double = double
        self.multiplier = 6 if double else 3
        self.lin = keras.layers.Linear(dim, self.multiplier * dim, bias=True)

    def forward(self, vec) :
        out = self.lin(ops.silu(vec))[:, None, :].chunk(self.multiplier, dim=-1)

        return (
            ModulationOut(*out[:3]),
            ModulationOut(*out[3:]) if self.is_double else None,
        )


class RMSNorm(keras.layers.Layer):
    def __init__(self, dim: int):
        super().__init__()
        self.scale = ops.Parameter(keras.ops.ones(dim))

    def forward(self, x):
        x_dtype = x.dtype
        x = x.float()
        rrms = ops.rsqrt(ops.mean(x**2, dim=-1, keepdim=True) + 1e-6)
        return (x * rrms).to(dtype=x_dtype) * self.scale

class QKNorm(keras.layers.Layer):
    def __init__(self, dim: int):
        super().__init__()
        self.query_norm = RMSNorm(dim)
        self.key_norm = RMSNorm(dim)

    def forward(self, q, k, v):
        q = self.query_norm(q)
        k = self.key_norm(k)
        return q.to(v), k.to(v)



class SelfAttention(keras.layers.Layer):
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.qkv = keras.layers.Linear(dim, dim * 3, bias=qkv_bias)
        self.norm = QKNorm(head_dim)
        self.proj = keras.layers.Linear(dim, dim)

    def forward(self, x, pe):
        qkv = self.qkv(x)
        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        q, k = self.norm(q, k, v)
        x = attention(q, k, v, pe=pe)
        x = self.proj(x)
        return x

@keras_hub_export("keras_hub.layers.SingleStreamBlock")
class SingleStreamBlock(keras.layers.Layer):
    """FNet encoder.

    This class follows the architecture of FNet encoder layer in the
    [FNet paper](https://arxiv.org/abs/2105.03824). Users can instantiate
    multiple instances of this class to stack up the encoder.

    Note on masking: In the official FNet code, padding tokens are added to the
    the input. However, the padding masks are deleted, i.e., mixing of
    all tokens is done. This is because certain frequencies will be zeroed
    out if we apply padding masks in every encoder layer. Hence, we don't
    take padding mask as input in the call() function.

    Args:
        intermediate_dim: int. The hidden size of feedforward network.
        dropout: float. The dropout value, applied in the
            feedforward network. Defaults to `0.`.
        activation: string or `keras.activations`. The
            activation function of feedforward network.
            Defaults to `"relu"`.
        layer_norm_epsilon: float. The epsilon value in layer
            normalization components. Defaults to `1e-5`.
        kernel_initializer: `str` or `keras.initializers` initializer.
            The kernel initializer for the dense layers.
            Defaults to `"glorot_uniform"`.
        bias_initializer: "string" or `keras.initializers` initializer.
            The bias initializer for the dense layers.
            Defaults to `"zeros"`.
        **kwargs: other keyword arguments passed to `keras.layers.Layer`,
            including `name`, `trainable`, `dtype` etc.

    Example:

    ```python
    # Create a single FNet encoder layer.
    encoder = keras_hub.layers.FNetEncoder(
        intermediate_dim=64)

    # Create a simple model containing the encoder.
    input = keras.Input(shape=(10, 64))
    output = encoder(input)
    model = keras.Model(inputs=input, outputs=output)

    # Call encoder on the inputs.
    input_data = np.random.uniform(size=(1, 10, 64))
    output = model(input_data)
    ```

    References:
     - [Lee-Thorp et al., 2021](https://arxiv.org/abs/2105.03824)
    """

    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float, qkv_bias: bool = False):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias


    def build(self, inputs_shape):
        self.hidden_dim = self.hidden_size
        self.num_heads = self.num_heads
        head_dim = self.hidden_size // self.num_heads
        self.scale = self.qk_scale or head_dim**-0.5

        self.mlp_hidden_dim = int(self.hidden_size * self.mlp_ratio)
        # qkv and mlp_in
        self.linear1 = keras.activations.linear(self.hidden_size, self.hidden_size * 3 + self.mlp_hidden_dim)
        # proj and mlp_out
        self.linear2 = keras.activations.linear(self.hidden_size + self.mlp_hidden_dim, self.hidden_size)

        self.norm = QKNorm(head_dim)

        self.hidden_size = self.hidden_size
        self.pre_norm = keras.layers.LayerNormalization(self.hidden_size, elementwise_affine=False, eps=1e-6)

        self.mlp_act = keras.activations.gelu(approximate="tanh")
        self.modulation = Modulation(self.hidden_size, double=False)


    def call(self, img, txt, vec, pe):
        img_mod1, img_mod2 = self.img_mod(vec)
        txt_mod1, txt_mod2 = self.txt_mod(vec)

        # prepare image for attention
        img_modulated = self.img_norm1(img)
        img_modulated = (1 + img_mod1.scale) * img_modulated + img_mod1.shift
        img_qkv = self.img_attn.qkv(img_modulated)
        img_q, img_k, img_v = rearrange(img_qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        img_q, img_k = self.img_attn.norm(img_q, img_k, img_v)

        # prepare txt for attention
        txt_modulated = self.txt_norm1(txt)
        txt_modulated = (1 + txt_mod1.scale) * txt_modulated + txt_mod1.shift
        txt_qkv = self.txt_attn.qkv(txt_modulated)
        txt_q, txt_k, txt_v = rearrange(txt_qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        txt_q, txt_k = self.txt_attn.norm(txt_q, txt_k, txt_v)

        # run actual attention
        q = keras.layers.Concatenate((txt_q, img_q), dim=2)
        k = keras.layers.Concatenate((txt_k, img_k), dim=2)
        v = keras.layers.Concatenate((txt_v, img_v), dim=2)

        attn = attention(q, k, v, pe=pe)
        txt_attn, img_attn = attn[:, : txt.shape[1]], attn[:, txt.shape[1] :]

        # calculate the img bloks
        img = img + img_mod1.gate * self.img_attn.proj(img_attn)
        img = img + img_mod2.gate * self.img_mlp((1 + img_mod2.scale) * self.img_norm2(img) + img_mod2.shift)

        # calculate the txt bloks
        txt = txt + txt_mod1.gate * self.txt_attn.proj(txt_attn)
        txt = txt + txt_mod2.gate * self.txt_mlp((1 + txt_mod2.scale) * self.txt_norm2(txt) + txt_mod2.shift)
        return img, txt



    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "intermediate_dim": self.intermediate_dim,
                "dropout": self.dropout,
                "activation": keras.activations.serialize(self.activation),
                "layer_norm_epsilon": self.layer_norm_epsilon,
                "kernel_initializer": keras.initializers.serialize(
                    self.kernel_initializer
                ),
                "bias_initializer": keras.initializers.serialize(
                    self.bias_initializer
                ),
            }
        )
        return config

    def compute_output_shape(self, inputs_shape):
        return inputs_shape
