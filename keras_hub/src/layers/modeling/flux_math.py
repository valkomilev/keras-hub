import keras
from keras import ops
from einops import rearrange

def batch_dot(x, y, axes=None):
    """DEPRECATED."""
    x_shape = x.shape
    y_shape = y.shape

    x_ndim = len(x_shape)
    y_ndim = len(y_shape)

    if x_ndim < 2 or y_ndim < 2:
        raise ValueError(
            "Cannot do batch_dot on inputs "
            "with rank < 2. "
            "Received inputs with tf.shapes "
            + str(x_shape)
            + " and "
            + str(y_shape)
            + "."
        )

    x_batch_size = x_shape[0]
    y_batch_size = y_shape[0]

    if x_batch_size is not None and y_batch_size is not None:
        if x_batch_size != y_batch_size:
            raise ValueError(
                "Cannot do batch_dot on inputs "
                "with different batch sizes. "
                "Received inputs with tf.shapes "
                + str(x_shape)
                + " and "
                + str(y_shape)
                + "."
            )
    if isinstance(axes, int):
        axes = [axes, axes]

    if axes is None:
        if y_ndim == 2:
            axes = [x_ndim - 1, y_ndim - 1]
        else:
            axes = [x_ndim - 1, y_ndim - 2]


    # if tuple, convert to list.
    axes = list(axes)

    # convert negative indices.
    if axes[0] < 0:
        axes[0] += x_ndim
    if axes[1] < 0:
        axes[1] += y_ndim

    # sanity checks
    if 0 in axes:
        raise ValueError(
            "Cannot perform batch_dot over axis 0. "
            "If your inputs are not batched, "
            "add a dummy batch dimension to your "
            "inputs using K.expand_dims(x, 0)"
        )
    a0, a1 = axes
    d1 = x_shape[a0]
    d2 = y_shape[a1]

    if d1 is not None and d2 is not None and d1 != d2:
        raise ValueError(
            "Cannot do batch_dot on inputs with tf.shapes "
            + str(x_shape)
            + " and "
            + str(y_shape)
            + " with axes="
            + str(axes)
            + ". x.shape[%d] != y.shape[%d] (%d != %d)."
            % (axes[0], axes[1], d1, d2)
        )

    # backup ndims. Need them later.
    orig_x_ndim = x_ndim
    orig_y_ndim = y_ndim

    # if rank is 2, expand to 3.
    if x_ndim == 2:
        x = ops.expand_dims(x, 1)
        a0 += 1
        x_ndim += 1
    if y_ndim == 2:
        y = ops.expand_dims(y, 2)
        y_ndim += 1

    # bring x's dimension to be reduced to last axis.
    if a0 != x_ndim - 1:
        pattern = list(range(x_ndim))
        for i in range(a0, x_ndim - 1):
            pattern[i] = pattern[i + 1]
        pattern[-1] = a0
        x = ops.transpose(x, pattern)

    # bring y's dimension to be reduced to axis 1.
    if a1 != 1:
        pattern = list(range(y_ndim))
        for i in range(a1, 1, -1):
            pattern[i] = pattern[i - 1]
        pattern[1] = a1
        #y = ops.transpose(y, pattern)

    # normalize both inputs to rank 3.
    if x_ndim > 3:
        # squash middle dimensions of x.
        x_shape = x.shape
        x_mid_dims = x_shape[1:-1]
        x_squashed_shape = ops.stack([x_shape[0], -1, x_shape[-1]])
        print('x_squashed_shape',x_squashed_shape)
        print('x',x.shape)
        #x = ops.reshape(x, x_squashed_shape[x_shape[0], -1, x_shape[-1]])
        x_squashed = True
    else:
        x_squashed = False

    if y_ndim > 3:
        # squash trailing dimensions of y.
        y_shape = y.shape
        y_trail_dims = y_shape[2:]
        y_squashed_shape = ops.stack([y_shape[0], y_shape[1], -1])
        #y = ops.reshape(y, y_squashed_shape)
        y_squashed = True
    else:
        y_squashed = False
    print('x-',x.shape)
    print('y-',y.shape)
    result = ops.matmul(x, y)
    print('post matmul',result.shape)
    # if inputs were squashed, we have to reshape the matmul output.
    output_shape = result.shape
    do_reshape = False
    return result
    if x_squashed:
        output_shape = ops.concatenate(
            [output_shape[:1], x_mid_dims, output_shape[-1:]], 0
        )
        do_reshape = True

    if y_squashed:
        output_shape = ops.concatenate([output_shape[:-1], y_trail_dims], 0)
        do_reshape = True

    if do_reshape:
        result = ops.reshape(result, output_shape)

    # if the inputs were originally rank 2, we remove the added 1 dim.
    if orig_x_ndim == 2:
        result = ops.squeeze(result, 1)
    elif orig_y_ndim == 2:
        result = ops.squeeze(result, -1)

    return result

class ScaledDotProductAttention(keras.layers.Layer):
    r"""The attention layer that takes three inputs representing queries, keys and values.

    \text{Attention}(Q, K, V) = \text{softmax}(\frac{Q K^T}{\sqrt{d_k}}) V

    See: https://arxiv.org/pdf/1706.03762.pdf
    """

    def __init__(self,
                 return_attention=False,
                 history_only=False,
                 **kwargs):
        """Initialize the layer.

        :param return_attention: Whether to return attention weights.
        :param history_only: Whether to only use history data.
        :param kwargs: Arguments for parent class.
        """
        super(ScaledDotProductAttention, self).__init__(**kwargs)
        self.supports_masking = True
        self.return_attention = return_attention
        self.history_only = history_only
        self.intensity = self.attention = None

    def get_config(self):
        config = {
            'return_attention': self.return_attention,
            'history_only': self.history_only,
        }
        base_config = super(ScaledDotProductAttention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, list):
            query_shape, key_shape, value_shape = input_shape
        else:
            query_shape = key_shape = value_shape = input_shape
        output_shape = query_shape[:-1] + value_shape[-1:]
        if self.return_attention:
            attention_shape = query_shape[:2] + (key_shape[1],)
            return [output_shape, attention_shape]
        return output_shape

    def compute_mask(self, inputs, mask=None):
        if isinstance(mask, list):
            mask = mask[0]
        if self.return_attention:
            return [mask, None]
        return mask

    def call(self, inputs, mask=None, **kwargs):
        if isinstance(inputs, list):
            query, key, value = inputs
        else:
            query = key = value = inputs
        if isinstance(mask, list):
            mask = mask[1]
        feature_dim = query.shape[-1]

        e = batch_dot(query, key, axes=2) /ops.sqrt(feature_dim) #ops.sqrt(K.cast(feature_dim, dtype=K.floatx()))

        if self.history_only:
            query_len, key_len = query[1], key[1]
            indices = keras.ops.expand_dims(ops.arange(0, key_len), axis=0)
            upper = keras.ops.expand_dims(ops.arange(0, query_len), axis=-1)
            e -= 10000.0 * keras.ops.expand_dims(indices > upper, axis=0)
        if mask is not None:
            e -= 10000.0 * (1.0 - keras.ops.expand_dims(mask, axis=-2))
        self.intensity = e
        e = ops.exp(e - ops.max(e, axis=-1, keepdims=True))
        self.attention = e / ops.sum(e, axis=-1, keepdims=True)
        print('value', value.shape)
        print('attention', self.attention.shape)
        v = batch_dot(self.attention, value,axes=[2,1])

        if self.return_attention:
            return [v, self.attention]
        return v
def apply_rope(xq, xk, freqs_cis):
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)
    xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
    xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]
    return xq_out.reshape(*xq.shape).type_as(xq), xk_out.reshape(*xk.shape).type_as(xk)


def attention(q, k, v, pe):
    q, k = apply_rope(q, k, pe)

    scaled_attention = ScaledDotProductAttention()
    x = scaled_attention([q, k, v])
    x = rearrange(x, "B H L D -> B L (H D)")

    return x