import tensorflow as tf

from tissue.utils.sparse import sparse_dense_matmult_batch


class GsumLayer(tf.keras.layers.Layer):

    def call(self, inputs):
        x = inputs[0]
        a = inputs[1]

        if isinstance(a, tf.SparseTensor):
            x = sparse_dense_matmult_batch(a, x)
        else:
            x = tf.matmul(a, x)
        return x
