import keras
from keras import ops

from einops import rearrange
from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.utils.keras_utils import clone_initializer
from keras_hub.src.layers.modeling.flux_math import attention

class ModulationOut:
    def __init__(self,shift,scale,gate):
        self.shift = shift
        self.scale = scale
        self.gate = gate


class Modulation(keras.layers.Layer):
    def __init__(self, dim: int, double: bool):
        super().__init__()
        self.is_double = double
        self.multiplier = 6 if double else 3
        self.lin = keras.layers.Dense( self.multiplier * dim)

    def forward(self, vec) :
        out = self.lin(ops.silu(vec))[:, None, :].chunk(self.multiplier, dim=-1)

        return (
            ModulationOut(*out[:3]),
            ModulationOut(*out[3:]) if self.is_double else None,
        )


class RMSNorm(keras.layers.Layer):
    def __init__(self, dim: int):
        super().__init__()
        self.scale =keras.ops.ones(dim)# ops.Parameter(keras.ops.ones(dim))

    def forward(self, x):
        x_dtype = x.dtype
        x = x.float()
        rrms = ops.rsqrt(ops.mean(x**2, axis=0) + 1e-6)
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

    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float, qk_scale=float):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.qk_scale = qk_scale


    def build(self, inputs_shape):

        head_dim = self.hidden_size // self.num_heads
        self.scale = self.qk_scale or head_dim**-0.5

        self.mlp_hidden_dim = int(self.hidden_size * self.mlp_ratio)
        # qkv and mlp_in
        self.linear1 = keras.layers.Dense(self.hidden_size * 3 + self.mlp_hidden_dim)
        # proj and mlp_out
        self.linear2 = keras.layers.Dense( self.hidden_size)

        self.norm = QKNorm(head_dim)

        self.hidden_size = self.hidden_size
        self.pre_norm = keras.layers.LayerNormalization(0, epsilon=1e-6)

        self.mlp_act = keras.layers.Dense( self.hidden_size,activation="tanh")#nn.GELU(approximate="tanh")
        self.modulation = Modulation(self.hidden_size, double=False)



    def call(self, x,  vec, pe):
        mod, _ = self.modulation(vec)
        x_mod = (1 + mod.scale) * self.pre_norm(x) + mod.shift

        qkv, mlp,_ = ops.split(self.linear1(x_mod), [3 * self.hidden_size, self.mlp_hidden_dim], axis=-1)
        #qkv, mlp = self.linear1(x_mod), [3 * self.hidden_size, self.mlp_hidden_dim],
        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        q, k = self.norm(q, k, v)

        # compute attention
        attn = attention(q, k, v, pe=pe)
        # compute activation in mlp stream, cat again and run second linear layer
        output = self.linear2(ops.concatenate((attn, self.mlp_act(mlp)), 2))
        return x + mod.gate * output



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
