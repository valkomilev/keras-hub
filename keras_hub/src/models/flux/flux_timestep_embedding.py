import keras
from keras import ops
import math

from einops import rearrange
from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.utils.keras_utils import clone_initializer
from keras_hub.src.layers.modeling.flux_math import attention

def timestep_embedding( t,  dim,
                        max_period=10000, time_factor: float = 1000.0):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        print('t', t)
        t = time_factor * t
        half = dim // 2
        freqs = keras.ops.exp(-math.log(max_period) * keras.ops.arange(start=0, stop=half, dtype='float32') / half)

        args = t[:, None].float() * freqs[None]
        embedding =  keras.layers.Concatenate(axis=-1)([keras.ops.cos(args), keras.ops.sin(args)])
        if dim % 2:
            embedding =  keras.layers.Concatenate( axis=-1)([embedding, keras.ops.zeros_like(embedding[:, :1])])
        #if torch.is_floating_point(t):
        #    embedding = embedding.to(t)
        return embedding
