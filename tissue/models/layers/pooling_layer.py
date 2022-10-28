import tensorflow as tf

from tissue.utils.sparse import sparse_dense_matmult_batch


class SpectralPool(tf.keras.layers.Layer):

    def call(self, input):
        h, a, s = input
        mask = tf.cast(tf.reduce_sum(tf.cast(h != 0, tf.float32), axis=-1, keepdims=True) > 0, tf.float32)
        s *= mask
        s_t = tf.transpose(s, (0, 2, 1))
        if isinstance(a, tf.SparseTensor):
            a_new = sparse_dense_matmult_batch(a, s)
        else:
            a_new = tf.matmul(a, s)
        a_new = tf.matmul(s_t, a_new)
        a_new = a_new / tf.reduce_sum(a_new, axis=2, keepdims=True)
        # a_new = tf.nn.softmax(a_new)
        cluster_counts = tf.reduce_sum(s_t, axis=2, keepdims=True)
        h_new = tf.matmul(s_t, h) / tf.math.maximum(cluster_counts, tf.ones_like(cluster_counts))
        return [h_new, a_new]


class DiffPool(tf.keras.layers.Layer):

    def __init__(
            self,
            activation,
            entropy_weight=1,
            **kwargs
    ):
        if isinstance(activation, str):
            self.activation = tf.keras.activations.get(activation)
        else:
            self.activation = activation
        self.entropy_weight = entropy_weight
        super().__init__(**kwargs)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'activation': self.activation
        })
        return config

    def call(self, inputs):
        h, a, s = inputs

        mask = tf.cast(tf.reduce_sum(tf.cast(h != 0, tf.float32), axis=-1, keepdims=True) > 0, tf.float32)
        s = tf.nn.softmax(s)
        s *= mask

        s_t = tf.transpose(s, (0, 2, 1))
        h_new = tf.matmul(s_t, h)
        if isinstance(a, tf.SparseTensor):
            a_new = sparse_dense_matmult_batch(a, s)
        else:
            a_new = tf.matmul(a, s)
        a_new = tf.matmul(s_t, a_new)

        factors = tf.cast(tf.shape(s)[2], tf.float32) / tf.reduce_sum(mask, axis=1, keepdims=True)
        h_new *= factors
        a_new = a_new / tf.reduce_sum(a_new, axis=2, keepdims=True)
        #a_new = tf.nn.softmax(a_new)

        if s.shape[2] == 1:
            h_new = tf.squeeze(h_new, axis=2)

        if s.shape[2] != 1:
            # # nearby nodes should be pooled together
            # if isinstance(a, tf.SparseTensor):
            #     a_temp = tf.SparseTensor(values=tf.ones_like(a.values), dense_shape=a.dense_shape, indices=a.indices)
            #     link_prediction_objective = tf.norm(
            #         tf.sparse.add(a_temp, -tf.matmul(s, s_t)),
            #         ord='fro',
            #         axis=(1, 2)
            #     )
            # else:
            #     link_prediction_objective = tf.norm(
            #         a - tf.matmul(s, s_t),
            #         ord='fro',
            #         axis=(1, 2)
            #     )
            # # Scale link prediction objective by number of elements of adjacency matrices
            # link_prediction_objective /= tf.math.square(tf.reduce_sum(mask, axis=1))
            # link_prediction_objective = tf.reduce_mean(link_prediction_objective)
            # self.add_loss(link_prediction_objective)

            # cluster assignments should be sparse
            eps = 1e-9
            entropy = tf.multiply(-s, tf.math.log(s + eps))
            entropy = tf.reduce_sum(entropy, axis=2)
            entropy = tf.reshape(entropy, shape=[-1])
            entropy = tf.boolean_mask(entropy, entropy > 0)
            entropy = tf.reduce_mean(entropy)
            self.add_loss(entropy*self.entropy_weight)

        #self.add_metric(link_prediction_objective, name="link_prediction", aggregation='mean')
        self.add_metric(entropy*self.entropy_weight, name="entropy", aggregation='mean')

        return [h_new, a_new]
