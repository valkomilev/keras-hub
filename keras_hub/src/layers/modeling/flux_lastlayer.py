import keras
from keras import ops

from einops import rearrange
from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.utils.keras_utils import clone_initializer
from keras_hub.src.layers.modeling.flux_math import attention

class LastLayer(keras.layers.Layer):
    def __init__(self, hidden_size: int, patch_size: int, out_channels: int):
        super().__init__()
        #self.norm_final = keras.layers.LayerNormalization(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm_final = keras.layers.LayerNormalization(0,  epsilon=1e-6)
        #self.linear = keras.layers.Dense(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.linear = keras.layers.Dense( patch_size * patch_size * out_channels)
        #        self.adaLN_modulation = keras.Sequential(keras.activations.silu(),
        #                                         keras.activations.linear(hidden_size, 2 * hidden_size, bias=True))
        self.adaLN_modulation = keras.Sequential()
        self.adaLN_modulation.add(keras.layers.Dense( 2 * hidden_size,activation='silu'))
        self.adaLN_modulation.add(keras.layers.Dense( 2 * hidden_size))

    def forward(self, x, vec):
        shift, scale = self.adaLN_modulation(vec).chunk(2, dim=1)
        (1 + scale[:, None, :])
        self.norm_final(x)
        shift[:, None, :]
        x = (1 + scale[:, None, :]) * self.norm_final(x) + shift[:, None, :]
        x = self.linear(x)
        return x