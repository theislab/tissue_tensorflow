import abc
import warnings
from typing import Dict, List, Tuple, Union

import numpy as np
import tensorflow as tf

import tissue.models
from tissue.utils.losses_and_metrics import custom_categorical_acc, custom_mae, custom_mae_survival, custom_mse, \
    custom_categorical_crossentropy, custom_survival, custom_categorical_crossentropy_nodes, custom_accuracy_nodes, \
    rel_cell_type_mse


class Estimator:
    """
    Estimator class for graph models. Contains all necessary methods for data loading,
    model initialization, training, evaluation and prediction.
    """

    # data
    # model

    # a = Dict[int, scipy.sparse.spmatrix]  # adjacency matrices TODO shape
    h: Dict[int, np.ndarray]  # features TODO shape
    y = Dict[int, Tuple[np.ndarray]]  # labels TODO shape

    img_keys_test: np.ndarray  # TODO what is this
    img_keys_eval: np.ndarray  # TODO what is this
    img_keys_train: np.ndarray  # TODO what is this

    history: dict  # TODO what is this

    target_label = str  # Core label to predict, others are down-weighted
    img_to_patient_dict = Dict[int, str]  # dictionary of image identifiers to patient identifiers
    graph_covariates = Dict[int, np.ndarray]  # covariates TODO shape
    graph_label_selection = List[str]  # list of image-meta data labels to be used as labels as output of network
    complete_img_keys = List[int]  # list of target image identifiers (diseased images)

    n_features: int  # Number of node features, second dimension of self.h values.
    n_types: int  # Number of node types, second dimension of self.node_types values.
    n_graph_covariates: int  # Number of graph covariates that can be concatenated to the node features space in self.h.
    # n_cluster
    max_nodes: int  # TODO what is this
    label_data_types: Dict[str, str]  # dictionary of graph labels with their respective data type: catgorical, ...
    label_length: Dict[str, int]  # dictionary of graph labels with their respective vector length
    # train_hyperparam: dict  # TODO what is this
    # cluster_assignments
    # node_supervision
    # self_supervision
    # self_supervision_label

    def _load_data(
            self,
            data_origin: str,
            data_path: str,
            buffered_data_path: str,
            write_buffer: bool,
            label_selection,
            radius,
            cell_type_coarseness: str,
    ):
        """
        Initializes a DataLoader object.
        """
        if data_origin == 'basel_zurich':
            # import sys
            # sys.path.append('/home/iterm/mayar.ali/phd/projects/ncem/')
            from ncem.data import DataLoaderBaselZurichZenodo as DataLoader
        elif data_origin == 'ionpath':
            # import sys
            # sys.path.append('/home/iterm/mayar.ali/phd/projects/ncem/')
            from ncem.data import DataLoaderIonpath as DataLoader
        elif data_origin == 'metabric':
            # import sys
            # sys.path.append('/home/iterm/mayar.ali/phd/projects/ncem/')
            from ncem.data import DataLoaderMetabric as DataLoader   
        elif data_origin == 'schuerch':
            # import sys
            # sys.path.append('/home/iterm/mayar.ali/phd/projects/ncem/')
            from ncem.data import DataLoaderSchuerch as DataLoader    
        elif data_origin == 'hartmann':
            import sys
            sys.path.append('/home/iterm/mayar.ali/phd/projects/ncem/')
            from ncem.data import DataLoaderHartmann as DataLoader
        else:
            raise ValueError('data_origin %s not recognized' % data_origin)
        
        print(f"{buffered_data_path}/buffered_data_{str(radius)}_{cell_type_coarseness}.pickle")
        self.data = DataLoader(
            data_path=data_path,
            buffered_data_path=buffered_data_path,
            write_buffer=write_buffer,
            radius=radius,
            label_selection=label_selection,
            cell_type_coarseness=cell_type_coarseness,
        )

    def _get_dataset(
            self,
            keys,
            batch_size,
            shuffle_buffer_size,
            seed=None,
            train=True
    ):
        """
        Prepares a dataset containing batched feature matrices and adjacency matrices as input and
        tissue metadata as labels for supervised models.

        :param keys: List of sample keys to use.
        :param batch_size:
        :param shuffle_buffer_size:
        :param seed: Seed to set for np.random for node downsampling.
        :return: A tf.data.Dataset for supervised models.
        """
        pass

    def get_data(
            self,
            data_origin,
            data_path,
            radius,
            target_label: str,
            c: str = False,
            buffered_data_path: str = None,
            write_buffer: bool = False,
            node_supervision: bool = False,
            cell_type_coarseness: str = 'fine',
            graph_label_selection: Union[List[str], Tuple[str], None] = None,
            graph_covar_selection: Union[List[str], Tuple[str], None] = None,
            node_label_space_id: str = "standard",
            node_feature_transformation: str = 'standardize_per_image',
            adj_type: Union[str, None] = None,
            drop_when_missing: list = [],
            n_cluster: int = 10,
            self_supervision: bool = False,
            self_supervision_label: list = ["relative_cell_types"],
            test_split=0.1,
            validation_split=0.1,
            seed: int = 1,
    ):
        """
        Prepares the necessary data input for the model.
        """
        self.get_data_args = {argument: value for argument, value in locals().items() if argument != 'self'}
        del self.get_data_args['data_path']

        self.target_label = target_label
        self.node_supervision = node_supervision
        self.n_cluster = n_cluster
        self.self_supervision = self_supervision
        self.self_supervision_label = self_supervision_label

        if graph_covar_selection is None:
            graph_covar_selection = []
        if graph_label_selection is None:
            graph_label_selection = []
        if target_label != 'none' and target_label not in graph_label_selection:
            graph_label_selection = graph_label_selection + [target_label]
        self.graph_label_selection = graph_label_selection
        labels_to_load = graph_label_selection + graph_covar_selection

        self._load_data(
            data_origin=data_origin,
            data_path=data_path,
            buffered_data_path=buffered_data_path,
            write_buffer=write_buffer,
            label_selection=labels_to_load,
            radius=radius,
            cell_type_coarseness=cell_type_coarseness,
        )

        # Validate graph-level label selection
        labels_not_found = [
            label for label in labels_to_load
            if label not in self.data.celldata.uns["graph_covariates"]["label_selection"]
        ]
        if len(labels_not_found) > 0:
            raise ValueError(f"could not find requested graph-level labels {labels_to_load}")

        # Prepare adjacency matrix
        if adj_type is None:
            if isinstance(self, tissue.estimators.EstimatorGCN) or \
                    isinstance(self, tissue.estimators.EstimatorGsum):
                adj_type = "scaled"
            elif isinstance(self, tissue.estimators.EstimatorGAT) or \
                    isinstance(self, tissue.estimators.EstimatorMultiInstance) or \
                    isinstance(self, tissue.estimators.EstimatorREG) or \
                    isinstance(self, tissue.estimators.EstimatorMultiInstanceDispersion) or \
                    isinstance(self, tissue.estimators.EstimatorREGDispersion):
                adj_type = "full"
            else:
                raise ValueError("estimator class %s not recognized" % type(self))
        self.get_data_args['adj_type'] = adj_type

        self.a = {k: adata.obsp["adjacency_matrix_connectivities"] for k, adata in self.data.img_celldata.items()}
        if adj_type == "scaled":
            self.a = self.data._transform_all_a(self.a)

        # Prepare node features
        if node_label_space_id == "standard":
            self.data.process_node_features(
                node_feature_transformation=node_feature_transformation,
            )
            self.h = {k: adata.X for k, adata in self.data.img_celldata.items()}
        elif node_label_space_id == "type":
            self.h = {k: adata.obsm["node_types"] for k, adata in self.data.img_celldata.items()}
        else:
            raise ValueError(f"node_label_space_id {node_label_space_id} not recognized")

        # Prepare graph labels
        self.y = {
            img: {
                feature: imgdata.uns['graph_covariates']['label_tensors'][feature]
                for feature in graph_label_selection
            }
            for img, imgdata in self.data.img_celldata.items()
        }
        if node_supervision:
            for k in list(self.y.keys()):
                self.y[k]["node_labels"] = self.data.img_celldata[k].obsm['node_types']

        label_data_types = self.data.celldata.uns['graph_covariates']['label_data_types']
        self.label_data_types = {
            feature: tp for feature, tp in label_data_types.items()
            if feature in self.graph_label_selection
        }
        self.label_length = {
            k: self.y[list(self.y.keys())[0]][k].shape[0]
            for k in self.graph_label_selection
        }
        for k, v in self.label_data_types.items():
            if v == "survival":
                self.label_length[k] = 1

        # Drop images that have missing important graph label information from self.complete_img_keys
        image_keys = list(self.data.img_celldata.keys())
        nr_keys_before = len(image_keys)
        for label in drop_when_missing:
            complete_data = [key for key, labels in self.y.items() if not np.isnan(np.sum(labels[label]))]
            image_keys = [img for img in image_keys if img in complete_data]
        self.complete_img_keys = image_keys
        self.img_to_patient_dict = {
            img: patient for img, patient in self.data.celldata.uns["img_to_patient_dict"].items()
            if img in self.complete_img_keys
        }
        img_dropped = nr_keys_before - len(self.complete_img_keys)
        if img_dropped > 0:
            print(f"Dropped {img_dropped} images with missing relevant target labels")

        # Prepare graph covariates
        if len(graph_covar_selection) > 0:
            self.graph_covariates = {  # Single 1D array per observation: concatenate all covariates!
                k: np.concatenate([adata.uns["graph_covariates"]["label_tensors"][kk] for kk in graph_covar_selection],
                                  axis=0)
                for k, adata in self.data.img_celldata.items()
            }
            # Replace masked entries (np.nan) by zeros: (masking can be handled properly in output but not here):
            for k, v in self.graph_covariates.items():
                if np.any(np.isnan(v)):
                    self.graph_covariates[k][np.isnan(v)] = 0.
        else:
            self.graph_covariates = {k: np.array([], ndmin=1) for k, adata in self.data.img_celldata.items()}

        # Prepare self-supervision
        if self_supervision:
            node_to_cluster_mapping, within_cluster_a, between_cluster_a = self.data.prepare_spectral_clusters(
                a_dict=self.a,
                n_cluster=n_cluster,
                k_neighbors=10,
            )
            for label in self_supervision_label:
                labels = self.data.get_self_supervision_label(
                    label,
                    node_to_cluster_mapping,
                    between_cluster_a,
                )
                for k in list(self.complete_img_keys):
                    self.y[k][label] = labels[k]
            self.cluster_assignments = node_to_cluster_mapping
            self.a = within_cluster_a
        else:
            self.cluster_assignments = {
                key: np.zeros(shape=(value.shape[0], n_cluster), dtype='float32')
                for key, value in self.h.items()
            }

        # Prepare model input information
        self.max_nodes = max([self.a[i].shape[0] for i in self.complete_img_keys])
        self.n_features = list(self.h.values())[0].shape[1]
        self.n_graph_covariates = list(self.graph_covariates.values())[0].shape[0]
        cluster_key = self.data.celldata.uns['metadata']['cluster_col_preprocessed']
        cluster = self.data.celldata.obs[cluster_key]
        self.n_types = len(np.unique(cluster[cluster == cluster]))

        # Split data by patients
        if isinstance(test_split, float) and isinstance(validation_split, float):
            self._split_data(
                test_split=test_split,
                validation_split=validation_split,
                seed=seed
            )

    @abc.abstractmethod
    def init_model(self, **kwargs):
        """
        Initializes and compiles the model.
        """
        pass

    def _split_data(
            self,
            test_split,
            validation_split,
            seed: int = 1
    ):
        """
        Split data randomly into partitions.

        :param test_split: Fraction of total data to be in test set.
        :param validation_split: Fraction of train-eval data to be in validation split.
        :param seed: Seed for random selection of observations.
        :return:
        """
        # Do Test-Val-Train split by patients and put all images for a patient into the chosen partition
        np.random.seed(seed)

        patient_ids_unique = np.unique(list(self.img_to_patient_dict.values()))

        number_patients_test = round(len(patient_ids_unique) * test_split)
        patient_keys_test = patient_ids_unique[np.random.choice(
            a=np.arange(len(patient_ids_unique)),
            size=number_patients_test,
            replace=False
        )]
        patient_idx_train_eval = np.array([x for x in patient_ids_unique if x not in patient_keys_test])
        number_patients_eval = round(len(patient_idx_train_eval) * validation_split)
        patient_keys_eval = patient_idx_train_eval[np.random.choice(
            a=np.arange(len(patient_idx_train_eval)),
            size=number_patients_eval,
            replace=False
        )]
        patient_keys_train = np.array([x for x in patient_idx_train_eval if x not in patient_keys_eval])

        self._split_data_by_patients(
            patient_keys_test,
            patient_keys_eval,
            patient_keys_train
        )

    def _split_data_by_patients(
            self,
            patient_keys_test,
            patient_keys_eval,
            patient_keys_train,
    ):
        """
        Split data into partitions defined by user arguments.

        :param patient_idx_test:
        :param patient_idx_val:
        :param patient_idx_train:
        :return:
        """
        patient_to_imagelist = {}
        for patient in np.unique(list(self.img_to_patient_dict.values())):
            patient_to_imagelist[patient] = []
        for image, patient in self.img_to_patient_dict.items():
            patient_to_imagelist[patient].append(image)

        self.img_keys_train = np.concatenate([
            patient_to_imagelist[patient] for patient in patient_keys_train
        ])
        if len(patient_keys_test) > 0:
            self.img_keys_test = np.concatenate([
                patient_to_imagelist[patient] for patient in patient_keys_test
            ])
        else:
            self.img_keys_test = []
        if len(patient_keys_eval) > 0:
            self.img_keys_eval = np.concatenate([
                patient_to_imagelist[patient] for patient in patient_keys_eval
            ])
        else:
            self.img_keys_eval = []

        patient_ids_unique = np.unique(list(self.img_to_patient_dict.values()))
        print(
            f"\nWhole dataset: {len(list(self.complete_img_keys))} images from {len(patient_ids_unique)} patients.\n"
            f"Test dataset: {len(self.img_keys_test)} images from {len(patient_keys_test)} patients.\n"
            f"Training dataset: {len(self.img_keys_train)} images from {len(patient_keys_train)} patients.\n"
            f"Validation dataset: {len(self.img_keys_eval)} images from {len(patient_keys_eval)} patients.\n"
        )
        if len(self.img_keys_train) == 0:
            raise ValueError("The train dataset is empty.")

    def train(
            self,
            epochs: int = 1000,
            max_steps_per_epoch: Union[int, None] = 20,
            batch_size: int = 128,
            validation_batch_size: int = 256,
            max_validation_steps: Union[int, None] = 10,
            patience: int = 20,
            lr_schedule_min_lr: float = 1e-5,
            lr_schedule_factor: float = 0.2,
            lr_schedule_patience: int = 5,
            monitor_partition: str = "val",
            monitor_metric: str = "loss",
            early_stopping: bool = True,
            reduce_lr_on_plateau: bool = True,
            shuffle_buffer_size: int = int(1e4),
            log_dir: Union[str, None] = None
    ):
        """
        Train model.

        Uses validation loss and maximum number of epochs as termination criteria.

        :param epochs: refer to tf.keras.models.Model.fit() documentation
        :param max_steps_per_epoch: Maximum steps per epoch.
        :param batch_size: refer to tf.keras.models.Model.fit() documentation
        :param validation_batch_size: Number of validation data observations to evaluate evaluation metrics on.
        :param max_validation_steps: Maximum number of validation steps to perform.
        :param patience: refer to tf.keras.models.Model.fit() documentation
        :param lr_schedule_min_lr: Minimum learning rate for learning rate reduction schedule.
        :param lr_schedule_factor: Factor to reduce learning rate by within learning rate reduction schedule
            when plateau is reached.
        :param lr_schedule_patience: Patience for learning rate reduction in learning rate reduction schedule.
        :param monitor_partition: {"train", "val"} Partition to tie training callbacks to.
        :param monitor_partition: e.g. "loss" Metric to tie training callbacks to.
        :param shuffle_buffer_size: tf.Dataset.shuffle(): buffer_size argument.
        :param log_dir: Directory to save tensorboard callback to. Disabled if None.
        :return:
        """
        # Save training settings to allow model restoring.
        self.train_hyperparam = {
            "epochs": epochs,
            "max_steps_per_epoch": max_steps_per_epoch,
            "batch_size": batch_size,
            "validation_batch_size": validation_batch_size,
            "max_validation_steps": max_validation_steps,
            "patience": patience,
            "lr_schedule_min_lr": lr_schedule_min_lr,
            "lr_schedule_factor": lr_schedule_factor,
            "lr_schedule_patience": lr_schedule_patience,
            "monitor_partition": monitor_partition,
            "monitor_metric": monitor_metric,
            "log_dir": log_dir
        }

        # Set callbacks.
        cbs = []
        if early_stopping:
            cbs.append(tf.keras.callbacks.EarlyStopping(
                monitor=monitor_partition + '_' + monitor_metric if monitor_partition == "val" else monitor_metric,
                patience=patience,
                restore_best_weights=True
            ))
        if reduce_lr_on_plateau:
            cbs.append(tf.keras.callbacks.ReduceLROnPlateau(
                monitor=monitor_partition + '_' + monitor_metric if monitor_partition == "val" else monitor_metric,
                factor=lr_schedule_factor,
                patience=lr_schedule_patience,
                min_lr=lr_schedule_min_lr
            ))
        if log_dir is not None:
            cbs.append(tf.keras.callbacks.TensorBoard(
                log_dir=log_dir,
                histogram_freq=0,
                write_graph=False,
                write_grads=False,
                write_images=False,
                embeddings_freq=0,
                embeddings_layer_names=None,
                embeddings_metadata=None,
                embeddings_data=None,
                update_freq='epoch'
            ))

        train_dataset = self._get_dataset(
            keys=self.img_keys_train,
            batch_size=batch_size,
            shuffle_buffer_size=min(shuffle_buffer_size, len(self.img_keys_train))
        )
        # if len(self.img_keys_eval):
        eval_dataset = self._get_dataset(
            keys=self.img_keys_eval,
            batch_size=validation_batch_size,
            shuffle_buffer_size=min(shuffle_buffer_size, len(self.img_keys_eval))
        )
        # else:
        #     eval_dataset = None

        steps_per_epoch = min(max(len(self.img_keys_train) // batch_size, 1), max_steps_per_epoch)
        validation_steps = min(max(len(self.img_keys_eval) // validation_batch_size, 1), max_validation_steps)

        self.history = self.model.training_model.fit(
            x=train_dataset,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            callbacks=cbs,
            validation_data=eval_dataset,
            validation_steps=validation_steps,
            verbose=2
        ).history

    def _compile_model(
            self, optimizer,
            loss_weight_others,
            loss_weight_type: float = 0.,
            loss_weight_self_supervision: float = 1.,
    ):
        """
        Prepares the losses and metrics and compiles the model.

        :param optimizer:
        :param loss_weight_others: Weight of non-target graph-level supervised tasks.
        :param loss_weight_type: Weight of auxillary node-level supervised task.
        :return:
        """
        loss = {}
        metrics = {}
        loss_weights = {}
        # Auxillary self_supervision loss:
        if self.self_supervision:
            for label in self.self_supervision_label:
                if label == 'relative_cell_types':
                    loss['self_supervision_' + label] = rel_cell_type_mse
                    metrics['self_supervision_' + label] = [
                        rel_cell_type_mse,
                        # tf.keras.metrics.mse,
                        # tf.keras.metrics.mae,
                        # tf.keras.metrics.msle
                    ]
                    if loss_weight_self_supervision == 0:
                        loss_weight_self_supervision = 1
                    loss_weights["self_supervision_" + label] = loss_weight_self_supervision

        # Auxillary node-level supervision:
        if self.node_supervision:
            loss["node_labels"] = custom_categorical_crossentropy_nodes
            metrics["node_labels"] = custom_accuracy_nodes
            loss_weights["node_labels"] = loss_weight_type
        if loss_weight_type > 0 and not self.node_supervision:
            raise ValueError("For auxiliary node-level supervision set node_supervision=True in get_data()!")
        # Graph-level supervision:
        for feature in self.graph_label_selection:
            dt = self.label_data_types[feature]
            if dt == 'categorical':
                loss[feature] = custom_categorical_crossentropy
                metrics[feature] = [
                    custom_categorical_crossentropy,
                    custom_categorical_acc,
                    tf.keras.metrics.categorical_crossentropy
                ]
                loss_weights[feature] = loss_weight_others
            elif dt == 'continuous':
                loss[feature] = custom_mse
                metrics[feature] = [
                    custom_mse,
                    custom_mae,
                    tf.keras.metrics.mse,
                    tf.keras.metrics.msle
                ]
                loss_weights[feature] = loss_weight_others
            elif dt == 'percentage':
                loss[feature] = custom_mse
                metrics[feature] = [
                    custom_mse,
                    custom_mae,
                    tf.keras.metrics.mse,
                    tf.keras.metrics.msle
                ]
                loss_weights[feature] = loss_weight_others
            elif dt == 'survival':
                loss[feature] = custom_survival
                metrics[feature] = [
                    custom_survival,
                    custom_mae_survival
                ]
                loss_weights[feature] = loss_weight_others
            else:
                raise ValueError('Data type not recognized! Use \'categorical\', \'continuous\' or \'percentage\'.')

        if self.target_label != 'none':
            loss_weights[self.target_label] = 1
        self.model.training_model.compile(
            loss=loss,
            optimizer=optimizer,
            metrics=metrics,
            loss_weights=loss_weights
        )

    def evaluate(
            self,
            keys=None,
            seed: int = 1
    ):
        if keys is not None:
            pass
        elif 'img_keys_test' in dir(self) and len(self.img_keys_test) != 0:
            keys = self.img_keys_test
        else:
            warnings.warn("Test set empty. Evaluating on whole dataset!")
            keys = list(self.complete_img_keys)
        dataset = self._get_dataset(
            keys=keys,
            batch_size=1,
            shuffle_buffer_size=1,
            seed=seed
        )
        results = self.model.training_model.evaluate(dataset, steps=len(keys))
        return dict(zip(self.model.training_model.metrics_names, results))

    def predict(
            self,
            keys=None,
            seed: int = 1
    ):
        if keys is not None:
            pass
        elif 'img_keys_test' in dir(self) and len(self.img_keys_test) != 0:
            keys = self.img_keys_test
        else:
            warnings.warn("Test set empty. Predicting on whole dataset!")
            keys = list(self.complete_img_keys)
        dataset = self._get_dataset(
            keys=keys,
            batch_size=1,
            shuffle_buffer_size=1,
            seed=seed,
            train=False
        )

        preds = self.model.training_model.predict(dataset, steps=len(keys))
        if len(self.model.training_model.outputs) == 1:
            preds = [preds]

        out = {
            node.name.split('/')[0]: preds[i]
            for i, node in enumerate(self.model.training_model.outputs)
            if 'node_labels' not in node.name.split('/')[0]
               and 'final_embedding' not in node.name.split('/')[0]
        }
        return out


class EstimatorGraph(Estimator):
    """
    Estimator base class that exposes the feature matrix and the adjacency matrix to model.
    """

    def _get_dataset(
            self,
            keys,
            batch_size,
            shuffle_buffer_size,
            train: bool = True,
            seed: Union[int, None] = None
    ):
        """
        Prepares a dataset containing batched feature matrices and adjacency matrices as input and
        tissue metadata as labels for supervised models.

        :param keys: List of sample indices to use.
        :param batch_size:
        :param shuffle_buffer_size:
        :param seed: Seed to set for np.random for node downsampling.
        :return: A tf.data.Dataset for supervised models.
        """
        if len(keys) == 0:
            return None

        def generator():
            for key in keys:
                h = self.h[key]
                diff = self.max_nodes - h.shape[0]
                padding_zeros_h = np.zeros((diff, h.shape[1]))
                h = np.asarray(np.concatenate((h, padding_zeros_h)), dtype="float32")

                a = self.a[key]
                coo = a.tocoo()
                a_ind = np.asarray(np.mat([coo.row, coo.col]).transpose(), dtype="int64")
                a_val = np.asarray(coo.data, dtype="float32")
                a_shape = np.asarray((self.max_nodes, self.max_nodes), dtype="int64")
                a = tf.SparseTensor(indices=a_ind, values=a_val, dense_shape=a_shape)

                c = np.asarray(self.graph_covariates[key], dtype="float32")

                cluster = self.cluster_assignments[key]
                padding_zeros = np.zeros((diff, cluster.shape[1]))
                cluster = np.asarray(np.concatenate((cluster, padding_zeros)))

                y = [np.asarray(self.y[key][x], dtype="float32") for x in self.graph_label_selection]
                if self.self_supervision:
                    for label in reversed(self.self_supervision_label):
                        y = [self.y[key][label]] + y
                if self.node_supervision:
                    y_node_label = np.asarray(self.y[key]["node_labels"], dtype="float32")
                    padding_zeros_label = np.zeros((diff, y_node_label.shape[1]))
                    y_node_label = np.asarray(np.concatenate((y_node_label, padding_zeros_label)), dtype="float32")
                    y = [y_node_label] + y
                y = tuple(y)

                yield (h, a, c, cluster), y

        output_signatures_y = [tf.TensorSpec(shape=(None,), dtype=tf.float32)] * len(self.graph_label_selection)
        if self.self_supervision:
            for label in reversed(self.self_supervision_label):
                if label == 'relative_cell_types':
                    output_signatures_y = [
                                              tf.TensorSpec(shape=(self.n_cluster, self.n_types), dtype=tf.float32)
                                          ] + output_signatures_y
        if self.node_supervision:
            output_signatures_y = [
                                      tf.TensorSpec(shape=(self.max_nodes, self.n_types), dtype=tf.float32)
                                  ] + output_signatures_y
        output_signatures_y = tuple(output_signatures_y)

        dataset = tf.data.Dataset.from_generator(
            generator=generator,
            output_signature=(
                (
                    tf.TensorSpec(shape=(self.max_nodes, self.n_features), dtype=tf.float32),  # node features (h)
                    tf.SparseTensorSpec(shape=None, dtype=tf.float32),  # adjacency matrix (a)
                    tf.TensorSpec(shape=(self.n_graph_covariates,), dtype=tf.float32),  # graph covariates (c)
                    tf.TensorSpec(shape=(self.max_nodes, self.n_cluster), dtype=tf.float32)  # clusters (cluster)
                ),
                output_signatures_y  # labels (y)
            )
        )
        if train:
            dataset = dataset.shuffle(
                buffer_size=shuffle_buffer_size,
                seed=seed,
                reshuffle_each_iteration=True
            )
            dataset = dataset.repeat()
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(5)
        return dataset


class EstimatorNoGraph(Estimator):
    """
    Estimator base class that only exposes the node feature matrix but not the adjacency matrix to model.
    """

    def _get_dataset(
            self,
            keys,
            batch_size,
            shuffle_buffer_size,
            train: bool = True,
            seed: Union[int, None] = None
    ):
        """
        Prepares a dataset containing batched feature matrices as input and
        tissue metadata as labels for supervised models.

        :param keys: List of sample indices to use.
        :param batch_size:
        :param shuffle_buffer_size:
        :param seed: Seed to set for np.random for node downsampling.
        :return: A tf.data.Dataset for supervised models.
        """
        if len(keys) == 0:
            return None

        def generator():
            for i, key in enumerate(keys):
                h = self.h[key]
                diff = self.max_nodes - h.shape[0]
                # zeros = np.zeros((diff, h.shape[1]))
                # h = np.asarray(np.concatenate([h, zeros]), dtype="float32")
                padding_zeros_h = np.zeros((diff, h.shape[1]))
                h = np.asarray(np.concatenate((h, padding_zeros_h)), dtype="float32")

                c = np.asarray(self.graph_covariates[key], dtype="float32")

                y = [np.asarray(self.y[key][x], dtype="float32") for x in self.graph_label_selection]
                if self.node_supervision:
                    y_node_label = np.asarray(self.y[key]["node_labels"], dtype="float32")
                    padding_zeros_label = np.zeros((diff, y_node_label.shape[1]))
                    y_node_label = np.asarray(np.concatenate((y_node_label, padding_zeros_label)), dtype="float32")
                    y = [y_node_label] + y
                y = tuple(y)

                yield (h, c), y

        output_signatures_y = [tf.TensorSpec(shape=(None,), dtype=tf.float32)] * len(self.graph_label_selection)
        if self.node_supervision:
            output_signatures_y = [
                                      tf.TensorSpec(shape=(self.max_nodes, self.n_types), dtype=tf.float32)
                                  ] + output_signatures_y

        output_signatures_y = tuple(output_signatures_y)

        dataset = tf.data.Dataset.from_generator(
            generator=generator,
            output_signature=(
                (
                    tf.TensorSpec(shape=(self.max_nodes, self.n_features), dtype=tf.float32),  # features (h)
                    tf.TensorSpec(shape=(self.n_graph_covariates,), dtype=tf.float32),  # covariates (c)
                ),
                output_signatures_y  # labels (y)
            )
        )
        if train:
            dataset = dataset.shuffle(
                buffer_size=shuffle_buffer_size,
                seed=seed,
                reshuffle_each_iteration=True
            )
            dataset = dataset.repeat()
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(5)
        return dataset



class EstimatorDispersion(Estimator):
    """
    Estimator base class that only exposes the node feature matrix and the adjacency matrix to model - used for node type only (Baseline model).
    """

    def _get_dataset(
            self,
            keys,
            batch_size,
            shuffle_buffer_size,
            train: bool = True,
            seed: Union[int, None] = None
    ):
        """
        Prepares a dataset containing batched feature matrices as input and
        tissue metadata as labels for supervised models.

        :param keys: List of sample indices to use.
        :param batch_size:
        :param shuffle_buffer_size:
        :param seed: Seed to set for np.random for node downsampling.
        :return: A tf.data.Dataset for supervised models.
        """
        if len(keys) == 0:
            return None

        def generator():
            for i, key in enumerate(keys):
                h = self.h[key]
                diff = self.max_nodes - h.shape[0]
                # zeros = np.zeros((diff, h.shape[1]))
                # h = np.asarray(np.concatenate([h, zeros]), dtype="float32")
                padding_zeros_h = np.zeros((diff, h.shape[1]))
                h = np.asarray(np.concatenate((h, padding_zeros_h)), dtype="float32")

                a = self.a[key]
                coo = a.tocoo()
                a_ind = np.asarray(np.mat([coo.row, coo.col]).transpose(), dtype="int64")
                a_val = np.asarray(coo.data, dtype="float32")
                a_shape = np.asarray((self.max_nodes, self.max_nodes), dtype="int64")
                a = tf.SparseTensor(indices=a_ind, values=a_val, dense_shape=a_shape)

                # calculate relative cell types per node as input
                h = tf.sparse.sparse_dense_matmul(a, h)
                h = h / np.maximum(np.sum(h, axis=1,  keepdims=True), np.ones((h.shape[0] ,1)))

                c = np.asarray(self.graph_covariates[key], dtype="float32")

                y = [np.asarray(self.y[key][x], dtype="float32") for x in self.graph_label_selection]
                if self.node_supervision:
                    y_node_label = np.asarray(self.y[key]["node_labels"], dtype="float32")
                    padding_zeros_label = np.zeros((diff, y_node_label.shape[1]))
                    y_node_label = np.asarray(np.concatenate((y_node_label, padding_zeros_label)), dtype="float32")
                    y = [y_node_label] + y
                y = tuple(y)

                yield (h, c), y

        output_signatures_y = [tf.TensorSpec(shape=(None,), dtype=tf.float32)] * len(self.graph_label_selection)
        if self.node_supervision:
            output_signatures_y = [
                                      tf.TensorSpec(shape=(self.max_nodes, self.n_types), dtype=tf.float32)
                                  ] + output_signatures_y

        output_signatures_y = tuple(output_signatures_y)

        dataset = tf.data.Dataset.from_generator(
            generator=generator,
            output_signature=(
                (
                    tf.TensorSpec(shape=(self.max_nodes, self.n_features), dtype=tf.float32),  # features (h)
                    tf.TensorSpec(shape=(self.n_graph_covariates,), dtype=tf.float32),  # covariates (c)
                ),
                output_signatures_y  # labels (y)
            )
        )
        if train:
            dataset = dataset.shuffle(
                buffer_size=shuffle_buffer_size,
                seed=seed,
                reshuffle_each_iteration=True
            )
            dataset = dataset.repeat()
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(5)
        return dataset