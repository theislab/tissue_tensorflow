import tensorflow as tf



class GATLayerSingleBatch(tf.keras.layers.Layer):

    def __init__(
            self,
            n_nodes: int,
            output_dim,
            attention_dim,
            number_heads,
            dropout_rate,
            activation,
            l2_reg,
            use_bias,
            batched: bool = True,
            step_len: int = 100,
            **kwargs
    ):
        """
        Initialise a Graph Attention layer.
        Args:
            output_dim (int): Number of output features.
            dropout_rate (float): Internal dropout rate.
            activation (str): The activation function to use.
            use_bias (bool): Whether to use bias or not. Defaults to `False`.
        """

        super().__init__(**kwargs)
        self.output_dim = output_dim
        self.attention_dim = attention_dim
        self.number_heads = number_heads
        self.dropout_rate = dropout_rate
        self.activation = tf.keras.activations.get(activation)
        self.use_bias = use_bias
        self.l2_reg = l2_reg

        self.kernel = None
        self.attention_kernel = None
        self.attention_kernel_2 = None
        self.bias = None

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'output_dim': self.output_dim,
            'attention_dim': self.attention_dim,
            'number_heads': self.number_heads,
            'dropout_rate': self.dropout_rate,
            'activation': self.activation,
            'use_bias': self.use_bias,
            'l2_reg': self.l2_reg
        })
        return config

    def build(self, input_shapes):
        input_dim = int(input_shapes[0][2])
        self.kernel = []
        self.attention_kernel = []
        self.attention_kernel_2 = []
        if self.use_bias:
            self.bias = []
        for _ in range(self.number_heads):
            self.kernel.append(
                self.add_weight(
                    shape=(input_dim, self.output_dim),
                    initializer=tf.initializers.glorot_uniform(),
                    name='kernel',
                    regularizer=tf.keras.regularizers.l2(self.l2_reg)
                )
            )
            self.attention_kernel.append(
                self.add_weight(
                    shape=(input_dim, self.attention_dim),
                    initializer=tf.initializers.glorot_uniform(),
                    name='attention_kernel',
                    regularizer=tf.keras.regularizers.l2(self.l2_reg)
                )
            )
            self.attention_kernel_2.append(
                self.add_weight(
                    shape=(input_dim, self.attention_dim),
                    initializer=tf.initializers.glorot_uniform(),
                    name='attention_kernel_2',
                    regularizer=tf.keras.regularizers.l2(self.l2_reg)
                )
            )
            if self.use_bias:
                self.bias.append(
                    self.add_weight(
                        shape=(1, self.output_dim),
                        name='b'
                    )
                )

    def call(self, inputs):
        # N number of nodes in graph
        # F number of features in each node
        # NH number of heads
        # BS batch size
        h = inputs[0]  # Node features (1 x N x F)
        a = inputs[1]  # Adjacency matrix (1 x N x N)

        h = tf.reshape(h, (h.shape[1], h.shape[2]))
        if isinstance(a, tf.SparseTensor):
            a = tf.sparse.reshape(a, (a.dense_shape[1], a.dense_shape[2]))
        else:
            a = tf.reshape(a, (a.shape[1], a.shape[2]))

        output = []
        for i in range(self.number_heads):
            # compute queries and keys
            features = tf.matmul(h, self.kernel[i])
            # (N, F) x (F, F') = (N, F')
            features_self = tf.matmul(h, self.attention_kernel[i])
            # (N, F) x (F, F') = (N, F')
            features_others = tf.matmul(h, self.attention_kernel_2[i])
            # (N, F) x (F, F') = (N, F')
            features_others = tf.transpose(features_others, [1, 0])
            # (F', N)

            dense = tf.matmul(features_self, features_others)
            masked = a * dense

            if isinstance(a, tf.SparseTensor):
                val_relu = tf.keras.layers.LeakyReLU(alpha=0.2)(masked.values)
                val_relu /= tf.sqrt(tf.cast(self.attention_dim, tf.float32))
                masked = tf.SparseTensor(values=val_relu, indices=masked.indices, dense_shape=masked.dense_shape)
                masked = tf.sparse.softmax(masked)  # (N, N)
            else:
                masked = tf.keras.layers.LeakyReLU(alpha=0.2)(masked)
                masked /= tf.sqrt(tf.cast(self.attention_dim, tf.float32))
                # set 0 values to -inf such that they don't affect the softmax scores
                mask = -1e9 * tf.where(a == 0, tf.ones_like(a), tf.zeros_like(a))
                masked += mask
                masked = tf.nn.softmax(masked)  # (N, N)
                # the masking with -inf does not work for zero rows thus the additional masking.
                masked *= tf.where(a == 0, tf.zeros_like(a), tf.ones_like(a))

            # Apply dropout to features
            dropout_attn = masked  # (N, N)
            dropout_feat = tf.keras.layers.Dropout(self.dropout_rate)(features)  # (N, F')

            # Linear combination with neighbors' features
            if isinstance(a, tf.SparseTensor):
                node_features = tf.sparse.sparse_dense_matmul(dropout_attn, dropout_feat)  # (N, F')
            else:
                node_features = tf.matmul(dropout_attn, dropout_feat)  # (N, F')

            if self.use_bias:
                node_features = node_features + self.bias[i]  # (N, F')

            if self.activation is not None:
                node_features = self.activation(node_features)  # (N, F')

            mask = tf.cast(tf.reduce_sum(tf.cast(h != 0, tf.float32), axis=-1, keepdims=True) > 0, tf.float32)
            node_features *= mask

            output.append(tf.expand_dims(node_features, 0))

        output = tf.concat(output, axis=-1)
        return output
