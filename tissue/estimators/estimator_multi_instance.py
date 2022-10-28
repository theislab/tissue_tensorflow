import tensorflow as tf
import tensorflow.keras.backend as K

from tissue.models import ModelMultiInstance
from tissue.estimators.base_estimator import EstimatorNoGraph, EstimatorDispersion


class EstimatorMultiInstance(EstimatorNoGraph):
        
    def init_model(
            self,
            optimizer: str = 'adam',
            learning_rate: float = 0.001,
            final_pooling: str = "max",
            depth: int = 3,
            width: int = 128,
            activation: str = 'relu',
            l2_reg: float = 0.,
            dropout_rate_pooling: float = 0.,
            add_covar_at_nodes: bool = True,
            add_covar_at_latent: bool = True,
            loss_weight_others: float = 1.,
            depth_final_dense: int = 3,
            **kwargs
    ):
        """
        Initializes and compiles the model.

        Parameters
        ----------
        optimizer
        learning_rate
        final_pooling
        depth
        width
        activation
        l2_reg
        dropout_rate_pooling
        lambda_mmd
        mmd_evaluations
        mmd_subsampling
        add_covar_at_nodes
        add_covar_at_latent
        loss_weight_others
        depth_final_dense
        kwargs

        Returns
        -------

        """
        self.model = ModelMultiInstance(
            input_shapes=(self.max_nodes, self.n_features, self.n_types, self.n_graph_covariates,),
            features=self.label_length,
            data_types=self.label_data_types,
            final_pooling=final_pooling,
            node_supervision=self.node_supervision,
            depth=depth,
            width=width,
            activation=activation,
            l2_reg=l2_reg,
            dropout_rate_pooling=dropout_rate_pooling,
            add_covar_at_nodes=add_covar_at_nodes,
            add_covar_at_latent=add_covar_at_latent,
            depth_final_dense=depth_final_dense,
        )
        optimizer = tf.keras.optimizers.get(optimizer)
        K.set_value(optimizer.lr, learning_rate)
        self._compile_model(optimizer, loss_weight_others)


# class EstimatorMultiInstanceDispersion(EstimatorDispersion):
        
#     def init_model(
#             self,
#             optimizer: str = 'adam',
#             learning_rate: float = 0.001,
#             final_pooling: str = "max",
#             depth: int = 3,
#             width: int = 128,
#             activation: str = 'relu',
#             l2_reg: float = 0.,
#             dropout_rate_pooling: float = 0.,
#             add_covar_at_nodes: bool = True,
#             add_covar_at_latent: bool = True,
#             loss_weight_others: float = 1.,
#             depth_final_dense: int = 3,
#             **kwargs
#     ):
#         """
#         Initializes and compiles the model.

#         Parameters
#         ----------
#         optimizer
#         learning_rate
#         final_pooling
#         depth
#         width
#         activation
#         l2_reg
#         dropout_rate_pooling
#         lambda_mmd
#         mmd_evaluations
#         mmd_subsampling
#         add_covar_at_nodes
#         add_covar_at_latent
#         loss_weight_others
#         depth_final_dense
#         kwargs

#         Returns
#         -------

#         """
#         self.model = ModelMultiInstance(
#             input_shapes=(self.max_nodes, self.n_features, self.n_types, self.n_graph_covariates,),
#             features=self.label_length,
#             data_types=self.label_data_types,
#             final_pooling=final_pooling,
#             node_supervision=self.node_supervision,
#             depth=depth,
#             width=width,
#             activation=activation,
#             l2_reg=l2_reg,
#             dropout_rate_pooling=dropout_rate_pooling,
#             add_covar_at_nodes=add_covar_at_nodes,
#             add_covar_at_latent=add_covar_at_latent,
#             depth_final_dense=depth_final_dense,
#         )
#         optimizer = tf.keras.optimizers.get(optimizer)
#         K.set_value(optimizer.lr, learning_rate)
#         self._compile_model(optimizer, loss_weight_others)

