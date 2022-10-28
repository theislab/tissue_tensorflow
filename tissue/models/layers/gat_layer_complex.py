import numpy as np
import tensorflow as tf

from tissue.utils.sparse import sparse_mul


class GATLayerComplex(tf.keras.layers.Layer):

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
            padded: bool = False,
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
        self.batched = batched
        self.step_len = step_len
        self.n_nodes = n_nodes
        self.n_batches = self.n_nodes // self.step_len + (self.n_nodes % self.step_len > 0)
        self.padded = padded

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
        self.kernel = self.add_weight(
            shape=(self.number_heads, input_dim, self.output_dim),
            initializer=tf.initializers.glorot_uniform(),
            name='kernel',
            regularizer=tf.keras.regularizers.l2(self.l2_reg)
        )
        self.attention_kernel = self.add_weight(
            shape=(self.number_heads, input_dim, self.attention_dim),
            initializer=tf.initializers.glorot_uniform(),
            name='attention_kernel',
            regularizer=tf.keras.regularizers.l2(self.l2_reg)
        )
        self.attention_kernel_2 = self.add_weight(
            shape=(self.number_heads, input_dim, self.attention_dim),
            initializer=tf.initializers.glorot_uniform(),
            name='attention_kernel_2',
            regularizer=tf.keras.regularizers.l2(self.l2_reg)
        )
        if self.use_bias:
            self.bias = self.add_weight(
                shape=(1, 1, self.number_heads, self.output_dim,),
                name='b'
            )

    def call(self, inputs):
        # N number of nodes in graph
        # F number of features in each node
        # NH number of heads
        # BS batch size
        h = inputs[0]  # Node features (1 x N x F)
        a = inputs[1]  # Adjacency matrix (1 x N x N)

        # compute queries and keys
        features = tf.einsum('ijk, lkm -> iljm', h, self.kernel)
        # (BS, N, F) x (NH, F, F') = (BS, NH, N, F')
        features_self = tf.einsum('ijk, lkm -> iljm', h, self.attention_kernel)
        # (BS, N, F) x (NH, F, F') = (BS, NH, N, F')
        features_others = tf.einsum('ijk, lkm -> iljm', h, self.attention_kernel_2)
        # (BS, N, F) x (NH, F, F') = (BS, NH, N, F')
        features_others = tf.transpose(features_others, [1, 0, 3, 2])
        features_self = tf.transpose(features_self, [1, 0, 2, 3])

        if isinstance(a, tf.SparseTensor):
            a = tf.sparse.expand_dims(a, 0)  # (1, BS, N, N)
            a = tf.sparse.concat(0, [a] * self.number_heads)  # (NH, BS, N, N)
        else:
            a = tf.expand_dims(a, 0)  # (1, BS, N, N)
            a = tf.tile(a, [self.number_heads, 1, 1, 1])  # (NH, BS, N, N)

        if self.batched:
            masked = tf.SparseTensor(
                indices=np.empty((0, 4)),
                values=[],
                dense_shape=(0, 0, 0, 0)
            )

            def body(i, tensor):
                # Compute feature combinations
                start = self.step_len * i
                end = tf.minimum(tf.cast(self.n_nodes, tf.int64), self.step_len * (i + 1))
                idx_slice = slice(start, end)
                f_s = features_self[:, :, idx_slice, :]
                if len(f_s.shape) == 3:  # catch shape collapse if idx is length 1
                    f_s = tf.expand_dims(f_s, 2)
                dense = tf.matmul(f_s, features_others)
                # (NH, BS, 1, F') x (NH, BS, F', N) = (NH, BS, 1, N)

                # Masking
                tensor_slice = tf.sparse.slice(
                    a,
                    start=[0, 0, start, 0],
                    size=[a.dense_shape[0], a.dense_shape[1], end - start, a.dense_shape[3]]
                )
                masked_row = tensor_slice * dense  # (NH, BS, 1, N)
                tensor = tf.sparse.concat(sp_inputs=[tensor, masked_row], axis=2, expand_nonconcat_dims=True)
                return i+1, tensor
            while_cond = lambda i, *_: i < self.n_batches
            _, masked = tf.while_loop(while_cond, body, [tf.cast(0, tf.int64), masked])
        else:
            dense = tf.matmul(features_self, features_others)
            masked = a * dense

        if isinstance(a, tf.SparseTensor):
            val_relu = tf.keras.layers.LeakyReLU(alpha=0.2)(masked.values)
            val_relu /= tf.sqrt(tf.cast(self.attention_dim, tf.float32))
            masked = tf.SparseTensor(values=val_relu, indices=masked.indices, dense_shape=masked.dense_shape)
            masked = tf.sparse.transpose(masked, [1, 0, 2, 3])
            masked = tf.sparse.softmax(masked)  # (BS, NH, N, N)
        else:
            masked = tf.keras.layers.LeakyReLU(alpha=0.2)(masked)
            masked /= tf.sqrt(tf.cast(self.attention_dim, tf.float32))
            # set 0 values to -inf such that they don't affect the softmax scores
            mask = -1e9 * tf.where(a == 0, tf.ones_like(a), tf.zeros_like(a))
            masked += mask
            masked = tf.nn.softmax(masked)  # (BS, NH, N, N)
            # the masking with -inf does not work for zero rows thus the additional masking.
            masked *= tf.where(a == 0, tf.zeros_like(a), tf.ones_like(a))
            masked = tf.transpose(masked, [1, 0, 2, 3])


        # Apply dropout to features
        dropout_attn = masked  # (BS, NH, N, N)
        dropout_feat = tf.keras.layers.Dropout(self.dropout_rate)(features)  # (BS, NH, N, F')

        # Linear combination with neighbors' features
        if isinstance(a, tf.SparseTensor):
            node_features = sparse_mul(dropout_attn, dropout_feat)  # (BS, NH, N, F')
        else:
            node_features = tf.matmul(dropout_attn, dropout_feat)  # (BS, NH, N, F')

        if self.use_bias:
            node_features = tf.transpose(node_features, [0, 2, 1, 3])  # (BS, N, NH, F')
            node_features = node_features + self.bias  # (BS, N, NH, F')
            node_features = tf.transpose(node_features, [0, 2, 1, 3])  # (BS, NH, N, F')

        if self.activation is not None:
            node_features = self.activation(node_features)  # (BS, NH, N, F')

        node_features = tf.transpose(node_features, [0, 2, 1, 3])
        output = tf.reshape(node_features, (-1, h.shape[1], self.number_heads * self.output_dim))

        if self.padded:
            mask = tf.cast(tf.reduce_sum(tf.cast(h != 0, tf.float32), axis=-1, keepdims=True) > 0, tf.float32)
            output *= mask

        return output
