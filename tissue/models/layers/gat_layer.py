import tensorflow as tf

from tissue.utils.sparse import sparse_mul_small


class GATLayer(tf.keras.layers.Layer):

    def __init__(
            self,
            output_dim,
            number_heads,
            dropout_rate,
            activation,
            l2_reg,
            use_bias,
            n_nodes: int = 0,
            batched: bool = True,
            step_len: int = 100,
            attention_dim: int = 0,
            **kwargs
    ):
        """
        Initialise a Graph Attention layer.
        Args:
            output_dim (int): Number of output features.
            dropout_rate (float): Internal dropout rate.
            activation (str): The activation function to use.
            use_bias (bool): Whether to use bias or not.
        """

        super().__init__(**kwargs)
        self.output_dim = output_dim
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
                    regularizer=tf.keras.regularizers.l2(l=self.l2_reg)
                )
            )
            self.attention_kernel.append(
                self.add_weight(
                    shape=(self.output_dim, 1),
                    initializer=tf.initializers.glorot_uniform(),
                    name='attention_kernel',
                    regularizer=tf.keras.regularizers.l2(l=self.l2_reg)
                )
            )
            self.attention_kernel_2.append(
                self.add_weight(
                    shape=(self.output_dim, 1),
                    initializer=tf.initializers.glorot_uniform(),
                    name='attention_kernel_2',
                    regularizer=tf.keras.regularizers.l2(l=self.l2_reg)
                )
            )
            if self.use_bias:
                self.bias.append(
                    self.add_weight(
                        shape=(1, 1, self.output_dim,),
                        name='b'
                    )
                )

    def call(self, inputs):
        # N number of nodes in graph
        # F number of features per node
        # F' number output features
        # BS batch size
        # NH number heads
        h = inputs[0]  # Node features (BS x N x F)
        a = inputs[1]  # Adjacency matrix (BS x N x N)

        output = []
        for i in range(self.number_heads):
            features = tf.matmul(h, self.kernel[i])
            # (BS, N, F) x (F, F') = (BS, N, F')

            # Compute feature combinations
            attention = tf.matmul(features, self.attention_kernel[i])
            # (BS, N, F) x (F', 1) = (BS, N, 1)

            attention_2 = tf.matmul(features, self.attention_kernel_2[i])
            # (BS, N, F) x (F', 1) = (BS, N, 1)

            if isinstance(a, tf.SparseTensor):
                att1 = a * attention # row-wise scaling
                att2 = a * tf.transpose(attention_2, (0, 2, 1)) # column-wise scaling
                att = tf.sparse.add(att1, att2)
                att = tf.SparseTensor(
                    values=tf.keras.layers.LeakyReLU(alpha=0.2)(att.values),
                    indices=att.indices,
                    dense_shape=att.dense_shape
                )
            else:
                att = attention + tf.transpose(attention_2, (0, 2, 1))  # (BS, N, N)
                att = a * att

            if isinstance(a, tf.SparseTensor):
                # tf.sparse.softmax is doing exactly what we want.
                # Masking zeros and leaving zero rows as they are
                att = tf.sparse.softmax(att)
            else:
                # set 0 values to -inf such that they don't affect the softmax scores
                mask = -1e9 * tf.where(a == 0, tf.ones_like(a), tf.zeros_like(a))
                att += mask
                att = tf.nn.softmax(att)
                # the masking with -inf does not work for zero rows thus the additional masking.
                att *= tf.where(a == 0, tf.zeros_like(a), tf.ones_like(a))

            # Apply dropout to features
            dropout_attn = att  # (BS, N, N)
            dropout_feat = tf.keras.layers.Dropout(self.dropout_rate)(features)  # (BS, N, F')

            # Linear combination with neighbors' features
            if isinstance(a, tf.SparseTensor):
                node_features = sparse_mul_small(dropout_attn, dropout_feat)
            else:
                node_features = tf.matmul(dropout_attn, dropout_feat)  # (BS, N, F')

            if self.use_bias:
                node_features = node_features + self.bias[i]

            if self.activation is not None:
                node_features = self.activation(node_features)

            mask = tf.cast(tf.reduce_sum(tf.cast(h != 0, tf.float32), axis=-1, keepdims=True) > 0, tf.float32)
            node_features *= mask
            output.append(node_features)

        output = tf.concat(output, axis=-1)
        return output
