import keras
from keras import ops

from einops import rearrange
from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.utils.keras_utils import clone_initializer


@keras_hub_export("keras_hub.layers.EmbedND")
class EmbedND(keras.layers.Layer):
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

    def __init__(
        self,
        dim,
        theta,
        axes_dim,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.dim = dim
        self.theta = theta
        self.axes_dim = axes_dim

    def build(self, inputs_shape):
        # Create layers based on input shape.
        n_axes = inputs_shape[-1]
        self.concat = keras.layers.Concatenate(axis=-3)

        self.built = True

    def call(self, inputs, training=None):
        """Forward pass of the FNetEncoder.

        Args:
            inputs: a Tensor. The input data to TransformerEncoder, should be
                of shape [batch_size, sequence_length, feature_dim].
            training: a boolean indicating whether the layer should behave in
                training mode or in inference mode.

        Returns:
            A Tensor of the same shape as the `inputs`.
        """

        def rope(pos, dim, theta) :
            assert dim % 2 == 0
            scale = keras.ops.arange(0, dim, 2, dtype='float64') / dim
            omega = 1.0 / (theta ** scale)
            out = ops.einsum("...n,d->...nd", pos, omega)
            out = ops.stack([ops.cos(out), - ops.sin(out), ops.sin(out), ops.cos(out)], axis=-1)
            out = rearrange(out, "b n d (i j) -> b n d i j", i=2, j=2)
            return out.float()

        n_axes = inputs.shape[-1]
        emb = keras.layers.Concatenate(axis=-3)\
            ([rope(inputs[..., i], self.axes_dim[i], self.theta) for i in range(n_axes)])
        return emb.unsqueeze(1)



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
