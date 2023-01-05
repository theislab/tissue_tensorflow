import os
import pickle
import numpy as np
from typing import Union

from tissue.estimators import EstimatorGCN, EstimatorGAT, EstimatorMultiInstance, EstimatorREG, EstimatorGsum, EstimatorREGDispersion


def _try_save(fn, obj):
    with open(fn, 'wb') as f:
        pickle.dump(obj=obj, file=f)


class TrainModel:
    estimator: Union[EstimatorGCN, EstimatorGAT, EstimatorMultiInstance, EstimatorREG, EstimatorGsum]

    def _save_evaluation(self, fn):
        img_keys = {
            'test': self.estimator.img_keys_test,
            'val': self.estimator.img_keys_eval,
            'train': self.estimator.img_keys_train,
            'all': self.estimator.complete_img_keys,
        }
        evaluations = {}
        for partition, keys in img_keys.items():
            if len(keys) > 0:
                evaluations[partition] = self.estimator.evaluate(keys)
            else:
                evaluations[partition] = None
        _try_save(fn + '_evaluation.pickle', evaluations)

    def _save_predictions(self, fn):
        img_keys = {
            'test': self.estimator.img_keys_test,
            'val': self.estimator.img_keys_eval,
            'train': self.estimator.img_keys_train,
            'all': self.estimator.complete_img_keys,
        }
        predictions = {}
        for partition, keys in img_keys.items():
            if len(keys) > 0:
                predictions[partition] = self.estimator.predict(keys)
            else:
                predictions[partition] = None
        _try_save(fn + '_predictions.pickle', predictions)

    def _save_data_info(self, fn):
        true_test = {name: [] for name in self.estimator.graph_label_selection}
        true_train = {name: [] for name in self.estimator.graph_label_selection}
        true_val = {name: [] for name in self.estimator.graph_label_selection}
        for ind in self.estimator.img_keys_test:
            for i, name in enumerate(self.estimator.graph_label_selection):
                true_test[name].append(np.expand_dims(self.estimator.y[ind][name], axis=0))
        for ind in self.estimator.img_keys_eval:
            for i, name in enumerate(self.estimator.graph_label_selection):
                true_val[name].append(np.expand_dims(self.estimator.y[ind][name], axis=0))
        for ind in self.estimator.img_keys_train:
            for i, name in enumerate(self.estimator.graph_label_selection):
                true_train[name].append(np.expand_dims(self.estimator.y[ind][name], axis=0))
        true_values = {
            'test': true_test,
            'val': true_val,
            'train': true_train
        }
        label_transformations = {
            "continuous_mean": self.estimator.data.celldata.uns["graph_covariates"]["continuous_mean"],
            "continuous_std": self.estimator.data.celldata.uns["graph_covariates"]["continuous_std"],
            # "survival_mean": self.estimator.data.celldata.uns["graph_covariates"].survival_mean 
        }
        info = {
            'patient_dict': self.estimator.img_to_patient_dict,
            'true_targets': true_values,
            'label_transformations': label_transformations
        }
        _try_save(fn + "_datainfo.pickle", info)

    def _save_history(self, fn):
        _try_save(fn + "_history.pickle", self.estimator.history)

    def _save_hyperparam(self, fn):
        _try_save(fn + "_hyperparam.pickle", self.estimator.train_hyperparam)

    def _save_get_data_args(self, fn):
        _try_save(fn + "_get_data_args.pickle", self.estimator.get_data_args)

    
    

    def _save_model(
            self,
            fn,
            save_weights: bool = True
    ):
        if save_weights:
            self.estimator.model.training_model.save_weights(fn + '_model_weights.tf')
        _try_save(fn + '_model_args.pickle', self.estimator.model.args)

    def save(self, fn, save_weights: bool = False):
        self._save_get_data_args(fn=fn)
        self._save_model(fn=fn, save_weights=save_weights)
        self._save_evaluation(fn=fn)
        self._save_predictions(fn=fn)
        self._save_history(fn=fn)
        self._save_hyperparam(fn=fn)
        self._save_data_info(fn=fn)


class TrainModelGCN(TrainModel):

    def init_estim(self):
        self.estimator = EstimatorGCN()


class TrainModelGsum(TrainModel):

    def init_estim(self):
        self.estimator = EstimatorGsum()


class TrainModelGAT(TrainModel):

    def init_estim(self):
        self.estimator = EstimatorGAT()


class TrainModelMI(TrainModel):

    def init_estim(self, dispersion=False):
        if dispersion:
            self.estimator = EstimatorMultiInstanceDispersion()
        else:
            self.estimator = EstimatorMultiInstance()


class TrainModelREG(TrainModel):

    def init_estim(self, dispersion=False):
        if dispersion:
            self.estimator = EstimatorREGDispersion()
        else:
            self.estimator = EstimatorREG()
