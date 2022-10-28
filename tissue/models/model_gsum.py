import tensorflow as tf
from typing import Tuple

from tissue.models.layers import SpectralPool, GsumLayer
from tissue.utils.pooling_function import NodePoolingLayer


class ModelGsum:
    
    def __init__(
            self,
            input_shapes: Tuple[Tuple[int, int, int], Tuple[int]],
            features,
            data_types,
            depth: int,
            activation: str,
            dropout_rate_pooling: float,
            add_covar_at_latent: bool,
            depth_final_dense: int,
            aggregation: str,
            aggregation_n_clusters: int,
            final_pooling,
    ):
        self.args = {argument: value for argument, value in locals().items() if argument != 'self'}

        if isinstance(activation, str) and 'leakyrelu' in activation:
            alpha = float(activation.split('_')[-1])
            activation = tf.keras.layers.LeakyReLU(alpha=alpha)

        input_x = tf.keras.layers.Input(
            shape=(input_shapes[0][0], input_shapes[0][1]),
            name='input_features'
        )
        input_a = tf.keras.layers.Input(
            shape=(input_shapes[0][0], input_shapes[0][0]),
            name='input_adj_mat', sparse=True
        )
        input_c = tf.keras.layers.Input(
            shape=(input_shapes[1][0]),
            name='input_covar'
        )
        input_cluster = tf.keras.layers.Input(
            shape=(input_shapes[0][0], aggregation_n_clusters),
            dtype="float32",
            name='input_cluster'
        )
        output = []

        a = input_a
        x = input_x
        mask = tf.cast(tf.reduce_sum(tf.cast(input_x != 0, tf.float32), axis=-1, keepdims=True) > 0, tf.float32)
        for d in range(depth):
            x = GsumLayer(
                name='layer_gsum_' + str(d)
            )([x, a])

        if aggregation.lower() == "spectral":
            x, a = SpectralPool()([x, a, input_cluster])
            mask = tf.expand_dims(tf.ones_like(x[:, :, 0]), axis=-1)

        # Final pooling step across node-derived instances (nodes or aggregated nodes).
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
            inputs=[input_x, input_a, input_c, input_cluster],
            outputs=output,
            name='gsum'
        )
