import tensorflow as tf
import tensorflow.keras.backend as K

from tissue.models import ModelGsum
from tissue.estimators.base_estimator import EstimatorGraph

    
class EstimatorGsum(EstimatorGraph):
        
    def init_model(
            self,
            optimizer: str = 'adam',
            learning_rate: float = 0.001,
            depth_feature_embedding: int = 0,
            depth: int = 2,
            activation: str = 'relu',
            dropout_rate_pooling: float = 0.,
            add_covar_at_latent: bool = False,
            loss_weight_others: float = 0.,
            aggregation: str = "none",
            aggregation_n_clusters: int = 10,
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
        activation
        dropout_rate_pooling
        add_covar_at_latent
        loss_weight_others
        aggregation
        aggregation_n_clusters
        final_pooling
        depth_final_dense
        kwargs

        Returns
        -------

        """
        self.model = ModelGsum(
            input_shapes=((self.max_nodes, self.n_features, self.n_types), (self.n_graph_covariates,)),
            features=self.label_length,
            data_types=self.label_data_types,
            final_pooling=final_pooling,
            depth=depth,
            activation=activation,
            dropout_rate_pooling=dropout_rate_pooling,
            add_covar_at_latent=add_covar_at_latent,
            depth_final_dense=depth_final_dense,
            aggregation=aggregation,
            aggregation_n_clusters=aggregation_n_clusters,
        )
        optimizer = tf.keras.optimizers.get(optimizer)
        K.set_value(optimizer.lr, learning_rate)
        self._compile_model(
            optimizer=optimizer,
            loss_weight_others=loss_weight_others,
            loss_weight_type=0.,
            loss_weight_self_supervision=0.,
        )
