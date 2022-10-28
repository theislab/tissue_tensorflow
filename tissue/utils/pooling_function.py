import tensorflow as tf


class NodePoolingLayer(tf.keras.layers.Layer):

    def __init__(
            self,
            method,
            dropout_rate: float = 0.,
            **kwargs
    ):
        self.method = method
        self.dropout_rate = dropout_rate
        if self.method == "attention":
            self.attention_weights = tf.keras.layers.Dense(
                units=1,
                activation=None,
                use_bias=True,
            )
        super().__init__(**kwargs)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'method': self.method,
            'dropout_rate': self.dropout_rate
        })
        return config

    def call(self, inputs):
        x, mask = inputs
        if self.method == "max":
            # make zero rows negative such that it does not influence max
            mask_neg = tf.ones_like(mask) - mask
            mask_neg *= -10e9
            x *= mask
            x += mask_neg
            if self.dropout_rate > 0.:
                x = tf.keras.layers.Dropout(rate=self.dropout_rate)(x)
            x = tf.math.reduce_max(x, axis=1, name="max_pool")
        elif self.method == "mean":
            x = x * mask
            if self.dropout_rate > 0.:
                x = tf.keras.layers.Dropout(rate=self.dropout_rate)(x)

            sum = tf.math.reduce_sum(mask, axis=1)
            sum = tf.where(tf.equal(sum, 0), tf.ones_like(sum), sum)
            
            x = tf.math.reduce_sum(x, axis=1, name="mean_pool") / sum
            # x = tf.math.reduce_sum(x, axis=1, name="mean_pool") / tf.math.reduce_sum(mask, axis=1)

        elif self.method == "attention":
            # Compute attention weight:
            x_weight = tf.reshape(x, shape=(-1, x.shape[-1]), name="reshape_for_weight")
            weight = self.attention_weights(x_weight)
            weight = tf.reshape(weight, shape=(-1, x.shape[1], 1), name="reshape_weight")
            # make zero rows negative such that they do not influence softmax
            mask_neg = tf.ones_like(mask) - mask
            mask_neg *= -10e9
            weight *= mask
            weight += mask_neg
            weight = tf.keras.layers.Softmax(axis=-2)(weight)
            x = x * weight
            x = x * mask
            if self.dropout_rate > 0.:
                x = tf.keras.layers.Dropout(rate=self.dropout_rate)(x)
            x = tf.math.reduce_sum(x, axis=1, name="mean_pool_attention_weighted") / \
                tf.math.reduce_sum(mask, axis=1)
        else:
            raise ValueError("pooling %s not recognized" % self.method)
        return x
