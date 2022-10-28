import tensorflow as tf
import tensorflow.keras.backend as K

from tissue.models import ModelGat
from tissue.estimators.base_estimator import EstimatorGraph
from tissue.utils.accumulative_optimizers import convert_to_accumulate_gradient_optimizer


class EstimatorGAT(EstimatorGraph):

    def init_model(
            self,
            optimizer: str = 'adam',
            learning_rate: float = 0.001,
            depth_feature_embedding: int = 0,
            depth: int = 2,
            width: int = 16,
            activation: str = 'relu',
            dropout_rate: float = 0.,
            dropout_rate_pooling: float = 0.,
            l2_reg: float = 0.,
            add_covar_at_nodes: bool = False,
            add_covar_at_latent: bool = False,
            loss_weight_others: float = 0.,
            loss_weight_type: float = 0.,
            loss_weight_self_supervision: float = 1.,
            aggregation: str = "none",
            aggregation_n_clusters: int = 10,
            aggregation_depth: int = 1,
            final_pooling: str = "max",
            depth_final_dense: int = 1,
            update_freq: int = 1,
            number_heads: int = 1,
            complex: bool = False,
            single_batched: bool = False,
            batched_gat: bool = True,
            step_len_gat_loop: int = 50,
            attention_dim: int = 4,
            **kwargs
    ):
        """
        Initializes and compiles the model.

        Parameters
        ----------
        optimizer
        learning_rate
        depth_feature_embedding
        depth
        width
        activation
        dropout_rate
        dropout_rate_pooling
        l2_reg
        add_covar_at_nodes
        add_covar_at_latent
        loss_weight_others
        loss_weight_type
        loss_weight_self_supervision
        aggregation
        aggregation_n_clusters
        aggregation_depth
        final_pooling
        depth_final_dense
        update_freq
        number_heads
        complex
        single_batched
        batched_gat
        step_len_gat_loop
        attention_dim
        kwargs

        Returns
        -------

        """
        if self.node_supervision and depth_feature_embedding==0:
            raise ValueError('For node supervision set depth_feature_embedding > 0!')
        self.model = ModelGat(
            input_shapes=(self.max_nodes, self.n_features, self.n_types, self.n_graph_covariates, self.n_cluster),
            features=self.label_length,
            data_types=self.label_data_types,
            final_pooling=final_pooling,
            node_supervision=self.node_supervision,
            depth_feature_embedding=depth_feature_embedding,
            depth=depth,
            width=width,
            activation=activation,
            dropout_rate=dropout_rate,
            dropout_rate_pooling=dropout_rate_pooling,
            l2_reg=l2_reg,
            add_covar_at_nodes=add_covar_at_nodes,
            add_covar_at_latent=add_covar_at_latent,
            depth_final_dense=depth_final_dense,
            aggregation=aggregation,
            aggregation_n_clusters=aggregation_n_clusters,
            aggregation_depth=aggregation_depth,
            number_heads=number_heads,
            complex=complex,
            batched_gat=batched_gat,
            step_len=step_len_gat_loop,
            attention_dim=attention_dim,
            single_batched=single_batched,
            self_supervision=self.self_supervision,
            self_supervision_labels=self.self_supervision_label,
        )
        optimizer = tf.keras.optimizers.get(optimizer)
        K.set_value(optimizer.lr, learning_rate)

        if single_batched:
            convert_to_accumulate_gradient_optimizer(orig_optimizer=optimizer, update_params_frequency=update_freq)

        self._compile_model(
            optimizer=optimizer,
            loss_weight_others=loss_weight_others,
            loss_weight_type=loss_weight_type,
            loss_weight_self_supervision=loss_weight_self_supervision,
        )
