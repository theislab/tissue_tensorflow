import tensorflow as tf
import tensorflow.keras.backend as K

from tissue.models import ModelReg, ModelRegDispersion
from tissue.estimators.base_estimator import EstimatorNoGraph, EstimatorDispersion


class EstimatorREG(EstimatorNoGraph):

    def init_model(
            self,
            optimizer: str = 'adam',
            learning_rate: float = 0.001,
            width: int = 128,
            activation: str = 'relu',
            l2_reg: float = 0.,
            loss_weight_others: float = 1.,
            depth: int = 3,
            **kwargs
    ):
        """
        Initializes and compiles the model.

        Parameters
        ----------
        optimizer
        learning_rate
        width
        activation
        l2_reg
        loss_weight_others
        depth
        kwargs

        Returns
        -------

        """

        self.model = ModelReg(
            input_shapes=((self.max_nodes, self.n_features), (self.n_graph_covariates,)),
            features=self.label_length,
            data_types=self.label_data_types,
            width=width,
            activation=activation,
            l2_reg=l2_reg,
            depth=depth
        )
        optimizer = tf.keras.optimizers.get(optimizer)
        K.set_value(optimizer.lr, learning_rate)
        self._compile_model(
            optimizer=optimizer,
            loss_weight_others=loss_weight_others,
            loss_weight_type=0
        )


class EstimatorREGDispersion(EstimatorDispersion):

    def init_model(
            self,
            optimizer: str = 'adam',
            learning_rate: float = 0.001,
            final_pooling: str = "max",
            depth: int = 3,
            width: int = 128,
            activation: str = 'relu',
            l2_reg: float = 0.,
            loss_weight_others: float = 1.,
            **kwargs
    ):
        """
        Initializes and compiles the model.

        Parameters
        ----------
        optimizer
        learning_rate
        width
        activation
        l2_reg
        loss_weight_others
        depth
        kwargs

        Returns
        -------

        """

        self.model = ModelRegDispersion(
            input_shapes=(self.max_nodes, self.n_features, self.n_types, self.n_graph_covariates,),
            features=self.label_length,
            data_types=self.label_data_types,
            width=width,
            activation=activation,
            l2_reg=l2_reg,
            depth=depth
        )
        optimizer = tf.keras.optimizers.get(optimizer)
        K.set_value(optimizer.lr, learning_rate)
        self._compile_model(
            optimizer=optimizer,
            loss_weight_others=loss_weight_others,
            loss_weight_type=0
        )
