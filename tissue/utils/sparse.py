import tensorflow as tf


def sparse_dense_matmult_batch(sp_a, b):
    """
    Multiplies a tf.SparseTensor sp_a with an additional batch dimension with a 2 dimensional
    dense matrix. Returns a 3 dimensional dense matrix.
    """
    def map_function(x):
        i, dense_slice = x[0], x[1]
        sparse_slice = tf.sparse.reshape(tf.sparse.slice(
            sp_a, [i, 0, 0], [1, sp_a.dense_shape[1], sp_a.dense_shape[2]]),
            [sp_a.dense_shape[1], sp_a.dense_shape[2]])
        mult_slice = tf.sparse.sparse_dense_matmul(sparse_slice, dense_slice)
        return mult_slice

    elems = (tf.range(0, sp_a.dense_shape[0], delta=1, dtype=tf.int64), b)
    return tf.map_fn(map_function, elems, fn_output_signature=tf.float32)


def sparse_mul(sp, de):
    """
    Multiplies a 4 dimensional sparse matrix with a 4 dimensional dense matrix.
    Shapes: (BS, NH, N, N) x (BS, NH, N, F) = (BS, NH, N, F)
    """
    bs = sp.dense_shape[0]
    nh = sp.dense_shape[1]
    n = sp.dense_shape[2]
    n2 = sp.dense_shape[3]
    f = tf.shape(de)[3]

    sp_reshaped = tf.sparse.reshape(sp, shape=(bs*nh, n, n2))
    de_reshaped = tf.reshape(de, shape=(bs*nh, n2, f))

    def map(x):
        ind, dense = x[0], x[1]
        sparse_slice = tf.sparse.reshape(tf.sparse.slice(
            sp_reshaped, [ind, 0, 0], [1, sp_reshaped.dense_shape[1], sp_reshaped.dense_shape[2]]
        ), [sp_reshaped.dense_shape[1], sp_reshaped.dense_shape[2]])
        out = tf.sparse.sparse_dense_matmul(sparse_slice, dense)
        return out

    elems = (tf.range(0, sp_reshaped.dense_shape[0], dtype=tf.int64), de_reshaped)
    out = tf.map_fn(map, elems, fn_output_signature=tf.float32)
    out = tf.reshape(out, shape=(bs, nh, n, f))
    return out


def sparse_mul_small(sp, de):
    """
    Multiplies a 3 dimensional sparse matrix with a 3 dimensional dense matrix.
    Shapes: (BS, N, N) x (BS, N, F) = (BS, N, F)
    """
    def map(x):
        ind, dense = x[0], x[1]
        sparse_slice = tf.sparse.reshape(tf.sparse.slice(
            sp, [ind, 0, 0], [1, sp.dense_shape[1], sp.dense_shape[2]]
        ), [sp.dense_shape[1], sp.dense_shape[2]])
        out = tf.sparse.sparse_dense_matmul(sparse_slice, dense)
        return out

    elems = (tf.range(0, sp.dense_shape[0], dtype=tf.int64), de)
    out = tf.map_fn(map, elems, fn_output_signature=tf.float32)
    return out
