from typing import Tuple

import tensorflow as tf

from tissue.models.layers import DiffPool, SpectralPool
from tissue.utils.pooling_function import NodePoolingLayer


class ModelGat:

    def __init__(
            self,
            input_shapes: Tuple[int, int, int, int, int],
            features,
            data_types,
            node_supervision: bool,
            depth_feature_embedding: int,
            depth: int,
            width: int,
            activation: str,
            dropout_rate: float,
            l2_reg: float,
            add_covar_at_nodes: bool,
            add_covar_at_latent: bool,
            depth_final_dense: int,
            aggregation: str,
            aggregation_n_clusters: int,
            aggregation_depth: int,
            final_pooling,
            number_heads: int,
            complex: bool,
            batched_gat: bool,
            step_len: int,
            attention_dim: int,
            dropout_rate_pooling: float = 0.,
            single_batched: bool = False,
            self_supervision: bool = False,
            self_supervision_labels: list = [],
    ):
        self.args = {argument: value for argument, value in locals().items() if argument != 'self'}

        if isinstance(activation, str) and 'leakyrelu' in activation:
            alpha = float(activation.split('_')[-1])
            activation = tf.keras.layers.LeakyReLU(alpha=alpha)

        if complex:
            if single_batched:
                from tissue.models.layers import GATLayerSingleBatch as GATLayer
            else:
                from tissue.models.layers import GATLayerComplex as GATLayer
        else:
            from tissue.models.layers import GATLayer as GATLayer

        input_x = tf.keras.layers.Input(
            shape=(input_shapes[0], input_shapes[1]),
            name='input_features'
        )
        input_a = tf.keras.layers.Input(
            shape=(input_shapes[0], input_shapes[0]),
            name='input_adj_mat', sparse=True
        )
        input_c = tf.keras.layers.Input(
            shape=(input_shapes[3]),
            name='input_covar'
        )
        input_cluster = tf.keras.layers.Input(
            shape=(input_shapes[0], aggregation_n_clusters),
            dtype="float32",
            name='input_cluster'
        )
        output = []

        a = input_a
        x = input_x
        mask = tf.cast(tf.reduce_sum(tf.cast(input_x != 0, tf.float32), axis=-1, keepdims=True) > 0, tf.float32)

        if add_covar_at_nodes:
            # Need to broadcast (a, n) based on (a, b, c) to (a, b, n) to concatenate to (a, b, c+n)
            x = tf.concat(
                values=[x, tf.tile(tf.expand_dims(input_c, axis=1), [1, input_shapes[0], 1])],
                axis=2, name="concat_features_covar"
            )
            x *= mask

        # Embed feature space, this happens on the cell and not on the graph level.
        if depth_feature_embedding > 0:
            n_features = x.shape[-1]
            # Reshape for node-wise layers:
            x = tf.reshape(  # (graphs, cells, number features)
                x,  # (graphs * cells, number features)
                shape=(-1, n_features),
                name="reshape_for_feature_embedding"
            )
            for i in range(depth_feature_embedding):
                x = tf.keras.layers.Dense(
                    units=width,
                    activation=activation,
                    use_bias=True,
                    kernel_regularizer=tf.keras.regularizers.l2(l2_reg),
                    name="Layer_dense_feature_embedding_" + str(i)
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
            # Reshape for graph layers:
            x = tf.reshape(  # (graphs * cells, number features)
                x,  # (graphs, cells, number features)
                shape=(-1, input_shapes[0], width), name="reshape_after_feature_embedding"
            )
            # Dense layers made zero rows probably non-zero
            x *= mask

        # Forward pass through node-wise GCN:
        for d in range(depth):
            x = GATLayer(
                n_nodes=input_shapes[0],
                output_dim=width,
                attention_dim=attention_dim,
                number_heads=number_heads,
                dropout_rate=dropout_rate,
                activation=activation,
                l2_reg=l2_reg,
                use_bias=True,
                batched=batched_gat,
                step_len=step_len,
                name='Layer_gat_' + str(d)
            )([x, a])

        if self_supervision:
            # clusterwise aggregation
            x_ss, a_ss = SpectralPool()([x, a, input_cluster])
            x_flat = tf.reshape(  # (graphs, cells, number features)
                x_ss,  # (graphs * cells, number features)
                shape=(-1, width),
                name="reshape_for_self_supervision"
            )
            # Auxillary self supervision loss:
            for self_sup_label in self_supervision_labels:
                if self_sup_label == 'relative_cell_types':
                    x_out = tf.keras.layers.Dense(
                        units=input_shapes[2],
                        activation="softmax",
                        use_bias=True,
                        name="self_supervision_" + self_sup_label
                    )(x_flat)
                    output.append(x_out)

        if aggregation.lower() == "diffpool":
            # Forward pass through node-pool-wise GAT:
            x_pool = GATLayer(
                n_nodes=input_shapes[0],
                output_dim=aggregation_n_clusters,
                attention_dim=attention_dim,
                number_heads=1,
                dropout_rate=dropout_rate,
                activation=None,
                l2_reg=l2_reg,
                use_bias=True,
                batched=batched_gat,
                step_len=step_len,
                name='Layer_gat_diffpool'
            )([x, a])
            x, a = DiffPool(
                activation=activation,
                name='Layer_diffpool'
            )([x, a, x_pool])
            mask = tf.expand_dims(tf.ones_like(x[:, :, 0]), axis=-1)
        elif aggregation.lower() == "spectral":
            if self_supervision:
                # use aggregated cluster features from self-supervision computation
                x = x_ss
                a = a_ss
            else:
                x, a = SpectralPool()([x, a, input_cluster])
            mask = tf.expand_dims(tf.ones_like(x[:, :, 0]), axis=-1)
        for d in range(aggregation_depth):
            x = GATLayer(
                n_nodes=aggregation_n_clusters,
                output_dim=width,
                attention_dim=attention_dim,
                number_heads=number_heads,
                dropout_rate=dropout_rate,
                activation=activation,
                l2_reg=l2_reg,
                use_bias=True,
                batched=False,
                step_len=step_len,
                name='Layer_gat_after_aggregation_' + str(d),
            )([x, a])

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
            name='basic_gat'
        )
