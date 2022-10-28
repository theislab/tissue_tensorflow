import tensorflow as tf

from tissue.utils.sparse import sparse_dense_matmult_batch


class GCNLayer(tf.keras.layers.Layer):
    
    def __init__(
            self,
            output_dim,
            dropout_rate,
            activation,
            l2_reg,
            use_bias: bool = False,
            padded: bool = True,
            **kwargs
    ):

        super().__init__(**kwargs)

        self.output_dim = output_dim
        self.dropout_rate = dropout_rate
        self.activation = tf.keras.activations.get(activation)
        self.l2_reg = l2_reg
        self.use_bias = use_bias
        self.padded = padded

        self.kernel = None
        self.bias = None

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'output_dim': self.output_dim,
            'dropout_rate': self.dropout_rate,
            'activation': self.activation,
            'l2_reg': self.l2_reg,
            'use_bias': self.use_bias
        })
        return config

    def build(self, input_shapes):
        input_shape = input_shapes[0]

        # Layer kernel
        self.kernel = self.add_weight(
            name='kernel',
            shape=(int(input_shape[2]), self.output_dim),
            initializer=tf.keras.initializers.glorot_uniform(),
            regularizer=tf.keras.regularizers.l2(self.l2_reg)
        )
        # Layer bias
        if self.use_bias:
            self.bias = self.add_weight(
                name='bias',
                shape=(self.output_dim,)
            )

    def call(self, inputs):
        x = inputs[0]
        a = inputs[1]

        if isinstance(a, tf.SparseTensor):
            t = sparse_dense_matmult_batch(a, x)
        else:
            t = tf.matmul(a, x)
        output = tf.tensordot(t, self.kernel, axes=1)
        output = tf.keras.layers.Dropout(self.dropout_rate)(output)

        if self.use_bias:
            output = tf.add(output, self.bias)
        if self.activation is not None:
            output = self.activation(output)

        if self.padded:
            mask = tf.cast(tf.reduce_sum(tf.cast(x != 0, tf.float32), axis=-1, keepdims=True) > 0, tf.float32)
            output *= mask

        return output
