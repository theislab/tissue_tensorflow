import tensorflow as tf
from typing import Tuple

from tissue.utils.pooling_function import NodePoolingLayer


class ModelMultiInstance:
    
    def __init__(
            self,
            input_shapes,
            features,
            data_types,
            node_supervision: bool,
            final_pooling: str,
            depth: int,
            width: int,
            activation: str,
            l2_reg: float,
            dropout_rate_pooling: float = 0.,
            add_covar_at_nodes: bool = True,
            add_covar_at_latent: bool = True,
            depth_final_dense: int = 2,
    ):
        self.args = {argument: value for argument, value in locals().items() if argument != 'self'}

        if isinstance(activation, str) and 'leakyrelu' in activation:
            alpha = float(activation.split('_')[-1])
            activation = tf.keras.layers.LeakyReLU(alpha=alpha)

        input_x = tf.keras.layers.Input(
            shape=(input_shapes[0], input_shapes[1]),
            name='input_features'
        )
        input_c = tf.keras.layers.Input(
            shape=(input_shapes[3]),
            name='input_covar'
        )
        output = []

        # MLP on individual cells: Map each cell to a new feature space.
        x = input_x
        mask = tf.cast(tf.reduce_sum(tf.cast(input_x != 0, tf.float32), axis=-1, keepdims=True) > 0, tf.float32)

        if add_covar_at_nodes:
            # Need to broadcast (a, n) based on (a, b, c) to (a, b, n) to concatenate to (a, b, c+n)
            x = tf.concat(
                values=[x, tf.tile(tf.expand_dims(input_c, axis=1), [1, input_shapes[0], 1])],
                axis=2, name="concat_features_covar"
            )
            x *= mask
            x = tf.reshape(
                x,
                shape=(-1, input_shapes[1]+input_shapes[3]),
                name="reshape_for_dense"
            )
        else:
            x = tf.reshape(
                x,
                shape=(-1, input_shapes[1]),
                name="reshape_for_dense"
            )
        for i in range(depth):
            x = tf.keras.layers.Dense(
                units=width,
                activation=activation,
                use_bias=True,
                kernel_regularizer=tf.keras.regularizers.l2(l2_reg),
                name="Layer_dense_feature_embedding" + str(i)
            )(x)
        # Auxillary cell-type classifier loss:
        if node_supervision:
            x_out = tf.keras.layers.Dense(
                units=input_shapes[2],
                activation="softmax",
                use_bias=True,
                name="node_labels"
            )(x)
            output.append(x_out)

        x_stats = x
        x = tf.reshape(x, shape=(-1, input_shapes[0], width))
        x *= mask

        # Pooling step across bag of instances (cells).
        # Select nodes that are not padding:
        x = NodePoolingLayer(
            method=final_pooling.lower(),
            dropout_rate=dropout_rate_pooling
        )((x, mask))

        # Add covariates to embedding and perform additional forward passes through dense layers.
        if add_covar_at_latent:
            x = tf.concat(
                values=[x, input_c],
                axis=1, name="concat_features_covar_final"
            )
        for i in range(depth_final_dense):
            x = tf.keras.layers.Dense(
                x.shape[1],
                activation=activation,
                use_bias=True,
                name="final_dense_" + str(i)
            )(x)

        # Map embedding to output space by each task in multitask setting (loop over tasks):
        for feature_name, feature_len in features.items():
            dt = data_types[feature_name]
            if dt == 'percentage':
                act = 'sigmoid'
            elif dt == 'categorical':
                act = 'softmax'
            elif dt == 'continuous':
                act = 'linear'
            elif dt == 'survival':
                act = 'relu'
            else:
                raise ValueError('Data type not recognized: Use \'categorical\', \'continuous\' or \'percentage\'.')
            x_out = tf.keras.layers.Dense(
                feature_len,
                activation=act,
                use_bias=True,
                name=feature_name
            )(x)
            output.append(x_out)

        self.training_model = tf.keras.models.Model(
            inputs=[input_x, input_c],
            outputs=output,
            name='multi_instance'
        )
