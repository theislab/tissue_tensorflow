import tensorflow as tf
import tensorflow.keras.backend as K

from tissue.models import ModelGcn
from tissue.estimators.base_estimator import EstimatorGraph

    
class EstimatorGCN(EstimatorGraph):
        
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
            entropy_weight: float = 1,
            final_pooling: str = "max",
            depth_final_dense: int = 1,
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
        entropy_weight
        final_pooling
        depth_final_dense
        kwargs

        Returns
        -------

        """
        if self.node_supervision and depth_feature_embedding == 0:
            raise ValueError('For node supervision set depth_feature_embedding > 0!')

        self.model = ModelGcn(
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
            entropy_weight=entropy_weight,
            self_supervision=self.self_supervision,
            self_supervision_labels=self.self_supervision_label,
        )
        optimizer = tf.keras.optimizers.get(optimizer)
        K.set_value(optimizer.lr, learning_rate)
        self._compile_model(
            optimizer=optimizer,
            loss_weight_others=loss_weight_others,
            loss_weight_type=loss_weight_type,
            loss_weight_self_supervision=loss_weight_self_supervision,
        )
