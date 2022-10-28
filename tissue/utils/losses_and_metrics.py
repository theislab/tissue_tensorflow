import tensorflow as tf
import tensorflow.keras.backend as K


def rel_cell_type_mse(y_true, y_pred):
    y_true = tf.reshape(y_true, (-1, tf.shape(y_true)[-1]))
    mask = tf.reduce_sum(y_true, axis=-1) > 0
    y_true_red = y_true[mask]
    y_pred_red = y_pred[mask]
    mask = tf.cast(mask, tf.float32)
    factor = tf.cond(
        tf.equal(tf.reduce_sum(mask), 0.),
        true_fn=lambda: 0.,
        false_fn=lambda: tf.cast(tf.size(mask), tf.float32) / tf.reduce_sum(mask)
    )
    return tf.keras.losses.MeanSquaredError()(y_true_red, y_pred_red) * factor


def custom_binary_crossentropy(y_true, y_pred):
    """
    Loss is scaled up to batch size based on non-observed labels.
    :param y_true:
    :param y_pred:
    :return:
    """
    mask = K.cast(tf.logical_not(tf.math.is_nan(y_true)), K.floatx())
    factor = tf.cond(
        tf.equal(tf.reduce_sum(mask), 0),
        true_fn=lambda: 0.,
        false_fn=lambda: tf.cast(tf.size(mask), tf.float32) / tf.reduce_sum(mask)
    )
    y_true = tf.where(tf.math.is_nan(y_true), tf.ones_like(y_true), y_true)
    return tf.keras.losses.BinaryCrossentropy()(y_true * mask, y_pred * mask) * factor


def custom_categorical_crossentropy(y_true, y_pred):
    """
    Loss is scaled up to batch size based on non-observed labels.
    :param y_true:
    :param y_pred:
    :return:
    """
    mask = tf.logical_not(tf.math.reduce_any(tf.math.is_nan(y_true), axis=1))
    y_true_red = y_true[mask]
    y_pred_red = y_pred[mask]
    factor = tf.cond(
        tf.equal(tf.size(y_true_red), 0),
        true_fn=lambda: 1.,
        false_fn=lambda: tf.cast(tf.size(y_true) / tf.size(y_true_red), tf.float32)
    )
    return tf.keras.losses.CategoricalCrossentropy()(y_true_red, y_pred_red) * factor


def custom_categorical_crossentropy_nodes(y_true, y_pred):
    """
    Cross entropy for node level supervision.
    :param y_true:
    :param y_pred:
    :return:
    """
    y_true = tf.reshape(y_true, (-1, tf.shape(y_true)[-1]))
    mask = tf.reduce_sum(y_true, axis=-1) > 0  # Select nodes that have a target label in one hot encoding.
    y_true_red = y_true[mask]
    y_pred_red = y_pred[mask]
    mask = tf.cast(mask, tf.float32)
    eps = 1e-8
    cce = tf.cond(  # Scale loss to mean across all nodes in batch.
        tf.equal(tf.reduce_sum(mask), 0),
        true_fn=lambda: 1.,
        false_fn=lambda: tf.cast(1. / tf.reduce_sum(mask), tf.float32) *
                         tf.reduce_mean(- y_true_red * tf.math.log(y_pred_red + eps) - (1 - y_true_red) * tf.math.log(1 - y_pred_red + eps))
    )
    return cce


def custom_accuracy_nodes(y_true, y_pred):
    """
    :param y_true:
    :param y_pred:
    :return: scalar accuracy across all nodes of all graphs
    """
    y_true = tf.reshape(y_true, (-1, tf.shape(y_true)[-1]))
    mask = tf.reduce_sum(y_true, axis=-1) > 0  # Select nodes that have a target label in one hot encoding.
    y_true_red = y_true[mask]
    y_pred_red = y_pred[mask]
    acc = tf.cond(
        tf.equal(tf.reduce_sum(tf.cast(mask, tf.int32)), 0),
        true_fn=lambda: 0.,
        false_fn=lambda: tf.reduce_mean(tf.cast(tf.argmax(y_pred_red, axis=-1) - tf.argmax(y_true_red, axis=-1) == 0, tf.float32))
    )
    return acc


def custom_mse(y_true, y_pred):
    """
    Loss is scaled up to batch size based on non-observed labels.
    :param y_true:
    :param y_pred:
    :return:
    """
    mask = K.cast(tf.logical_not(tf.math.is_nan(y_true)), K.floatx())
    factor = tf.cond(
        tf.equal(tf.reduce_sum(mask), 0),
        true_fn=lambda: 0.,
        false_fn=lambda: tf.cast(tf.size(mask), tf.float32) / tf.reduce_sum(mask)
    )
    y_true = tf.where(tf.math.is_nan(y_true), tf.ones_like(y_true), y_true)
    return tf.keras.losses.MeanSquaredError()(y_true * mask, y_pred * mask) * factor


def custom_survival(y_true, y_pred):
    """
    Loss is scaled up to batch size based on non-observed labels.
    First column is numeric surival and second column is whether data is censored (ie whether patient is not yet dead).
    If censored, over-estimation is not penalized. MSE loss is used for deviation.
    :param y_true:
    :param y_pred:
    :return:
    """
    y_censor = y_true[:, 1]
    y_true = y_true[:, 0]
    y_pred = y_pred[:, 0]
    mask = tf.cast(tf.logical_not(tf.math.is_nan(y_true)), dtype="float32")
    y_censor = tf.where(tf.math.is_nan(y_true), tf.ones_like(y_censor), y_censor)
    y_pred = tf.where(tf.math.is_nan(y_true), tf.zeros_like(y_pred), y_pred)
    y_true = tf.where(tf.math.is_nan(y_true), tf.zeros_like(y_true), y_true)
    factor = tf.cond(
        tf.equal(tf.reduce_sum(mask), 0),
        true_fn=lambda: 0.,
        false_fn=lambda: tf.cast(tf.size(mask), tf.float32) / tf.reduce_sum(mask)
    )
    loss = tf.math.square(y_true - y_pred) * factor
    # Perform censoring: Lower predictions are always penalized, higher predictions only penalized if not censored.
    loss = loss * y_censor + loss * (1. - y_censor) * tf.cast(y_true > y_pred, dtype=tf.float32)
    return loss


def custom_binary_acc(y_true, y_pred):
    #mask = K.cast(tf.logical_not(tf.math.is_nan(y_true)), K.floatx())
    #factor = tf.cond(
    #    tf.equal(tf.reduce_sum(mask), 0),
    #    true_fn=lambda: 0.,
    #    false_fn=lambda: tf.cast(tf.size(mask), tf.float32) / tf.reduce_sum(mask)
    #)
    #y_true = tf.where(tf.math.is_nan(y_true), tf.ones_like(y_true), y_true)
    #return tf.keras.metrics.binary_accuracy(y_true * mask, y_pred * mask) * factor
    mask = tf.logical_not(tf.math.reduce_any(tf.math.is_nan(y_true), axis=1))
    y_true_red = y_true[mask]
    y_pred_red = y_pred[mask]
    acc = tf.cond(
        tf.equal(tf.size(y_true_red), 0),
        true_fn=lambda: 0.,
        false_fn=lambda: tf.keras.metrics.binary_accuracy(y_true_red, y_pred_red)
    )
    return acc


def custom_categorical_acc(y_true, y_pred):
    #mask = K.cast(tf.logical_not(tf.math.is_nan(y_true)), K.floatx())
    #factor = tf.cond(
    #    tf.equal(tf.reduce_sum(mask), 0),
    #    true_fn=lambda: 0.,
    #    false_fn=lambda: tf.cast(tf.size(mask), tf.float32) / tf.reduce_sum(mask)
    #)
    #y_true = tf.where(tf.math.is_nan(y_true), tf.ones_like(y_true), y_true)
    #return tf.keras.metrics.categorical_accuracy(y_true * mask, y_pred * mask) * factor
    mask = tf.logical_not(tf.math.reduce_any(tf.math.is_nan(y_true), axis=1))
    y_true_red = y_true[mask]
    y_pred_red = y_pred[mask]
    acc = tf.cond(
        tf.equal(tf.size(y_true_red), 0),
        true_fn=lambda: 0.,
        false_fn=lambda: tf.keras.metrics.categorical_accuracy(y_true_red, y_pred_red)
    )
    return acc


def custom_mae(y_true, y_pred):
    #mask = K.cast(tf.logical_not(tf.math.is_nan(y_true)), K.floatx())
    #factor = tf.cond(
    #    tf.equal(tf.reduce_sum(mask), 0),
    #    true_fn=lambda: 0.,
    #    false_fn=lambda: tf.cast(tf.size(mask), tf.float32) / tf.reduce_sum(mask)
    #)
    #y_true = tf.where(tf.math.is_nan(y_true), tf.ones_like(y_true), y_true)
    #return tf.keras.metrics.mean_absolute_error(y_true * mask, y_pred * mask) * factor
    mask = tf.logical_not(tf.math.reduce_any(tf.math.is_nan(y_true), axis=1))
    y_true_red = y_true[mask]
    y_pred_red = y_pred[mask]
    mae = tf.cond(
        tf.equal(tf.size(y_true_red), 0),
        true_fn=lambda: 0.,
        false_fn=lambda: tf.keras.metrics.mean_absolute_error(y_true_red, y_pred_red)
    )
    return mae


def custom_mae_survival(y_true, y_pred):
    """
    First column is numeric surival and second column is whether data is censored (ie whether patient is not yet dead).
    If censored, over-estimation is not penalized. MSE loss is used for deviation.
    :param y_true:
    :param y_pred:
    :return:
    """
    y_true_red = y_true[:, 0]
    y_censor_red = y_true[:, 1]
    y_pred_red = y_pred[:, 0]
    # zeros mean censoring
    y_censor_red = tf.where(tf.math.is_nan(y_true_red), tf.zeros_like(y_censor_red), y_censor_red)
    y_pred_red = tf.where(tf.math.is_nan(y_true_red), tf.zeros_like(y_pred_red), y_pred_red)
    y_true_red = tf.where(tf.math.is_nan(y_true_red), tf.zeros_like(y_true_red), y_true_red)
    mae = tf.math.abs(y_true_red - y_pred_red)
    mae = mae * y_censor_red + mae * (1. - y_censor_red) * tf.cast(y_true_red > y_pred_red, dtype=tf.float32)
    mae = tf.reduce_mean(mae)
    return mae
