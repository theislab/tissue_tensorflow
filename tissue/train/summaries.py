import os
import pickle
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd


class GridSearchContainer:
    """
    Loads, processes and depicts the results of one grid search run.

    Exposes the following processed data objects as attributes to user:
        - summary_table: pandas Dataframe
            Summarise all metrics in a single table with rows for each runs and CV partition.
        - run_ids_clean: list
            List of IDs of completed runs (all files present).
        - idx_test:
            Dictionary with cross-validation as key and list of test indices as values.
        - idx_train:
            Dictionary with cross-validation as key and list of train indices as values.
        - idx_val:
            Dictionary with cross-validation as key and list of validation indices as values.
        - patient_idx_test:
            Dictionary with cross-validation as key and list of patient ids used for testing.
        - patient_idx_train:
            Dictionary with cross-validation as key and list of patient ids used for training.
        - patient_idx_val:
            Dictionary with cross-validation as key and list of patient ids used for validation.
        - patient_dict:
            Dictionary with cross-validation as key and a dictionary of indices to patient id as values.
            # TODO what has cross-validation to do with this? The single dictionarys for the different cross-validations
            # TODO all seem to be the same
        - evals:
            Contains losses and metrics for all labels for all runs, all cross-validation-splits, all partitions.
        - true_labels:
            For all cross-validation parts, all partitions the true one-hot encoded target labels.
        - runparams:
            A dictionary containing all hyperparams for all runs like label_selection, diseased_only, ...
        - info:
            Contains full results of a single selected run.
    """
    summary_table: pd.DataFrame
    run_ids_clean: List
    idx_test: Dict
    idx_train: Dict
    idx_val: Dict
    patient_idx_test: Dict
    patient_idx_train: Dict
    patient_idx_val: Dict

    patient_dict: dict
    evals: dict
    true_labels: dict
    runparams: dict
    info: dict

    def __init__(
            self,
            source_path: Union[str, dict],
            gs_ids: List[str],
    ):
        """

        :param source_path: Path as string or dictionary of paths by gs_ids.
        :param gs_ids: List of grid search IDs to load.
        """
        if isinstance(source_path, str):
            source_path = dict([(x, source_path) for x in gs_ids])
        self.source_path = source_path
        self.gs_ids = gs_ids

    def load_gs(
            self,
            expected_files: list = [
                'evaluation.pickle',
                'history.pickle',
                'hyperparam.pickle',
                'get_data_args.pickle',
                'model_args.pickle',
            ]
    ):
        """
        Load all metrics from grid search output files.

        Core results are save in self.summary_table.

        :return:
        """
        self.summary_table = []
        self.runparams = {}
        self.run_ids_clean = {}
        self.evals = {}
        for gs_id in self.gs_ids:
            # Collect runs that belong to grid search by looping over file names in directory.
            indir = self.source_path[gs_id] + gs_id + "/results/"
            # runs_ids are the unique hyper-parameter settings, which are again subsetted by cross-validation.
            # These ids are present in all files names but are only collected from the history files here.
            run_ids = np.sort(np.unique([
                "_".join(".".join(x.split(".")[:-1]).split("_")[:-1])
                for x in os.listdir(indir)
                if 'history' in x
            ]))
            cv_ids = np.sort(np.unique([  # identifiers of cross-validation splits
                x.split("_")[-1]
                for x in run_ids
            ]))
            run_ids = np.sort(np.unique([  # identifiers of hyper-parameters settings
                "_".join(x.split("_")[:-1])
                for x in run_ids
            ]))
            run_ids_clean = []  # only IDs of completed runs (all files present)
            for r in run_ids:
                complete_run = True
                for cv in cv_ids:
                    # Check pickled files:
                    for end in expected_files:
                        fn = r + "_" + cv + "_" + end
                        if not os.path.isfile(indir + fn):
                            print("File %r missing" % fn)
                            complete_run = False
                # Check run parameter files (one per cross-validation set):
                fn = r + "_runparams.pickle"
                if not os.path.isfile(indir + fn):
                    print("File %r missing" % fn)
                    complete_run = False
                if not complete_run:
                    print("Run %r not successful" % (r + "_" + cv))
                    pass
                else:
                    run_ids_clean.append(r)

            # Load results and settings from completed runs:
            evals = {}  # Dictionary over runs with dictionary over cross-validations with results from model evaluation.
            runparams = {}  # Dictionary over runs with model settings.
            for x in run_ids_clean:
                # Load model settings (these are shared across all partitions).
                if os.path.isfile(
                        indir + x + "_runparams.pickle"):  # TODO depreceate and only keep the first case (newer commit)
                    fn_runparams = indir + x + "_runparams.pickle"
                    with open(fn_runparams, 'rb') as f:
                        runparams[x] = pickle.load(f)
                else:
                    fn_runparams = indir + x + "_" + cv_ids[0] + "_runparams.pickle"
                    with open(fn_runparams, 'rb') as f:
                        runparams[x] = pickle.load(f)
                evals[x] = {}
                for cv in cv_ids:
                    fn_eval = indir + x + "_" + cv + "_evaluation.pickle"
                    with open(fn_eval, 'rb') as f:
                        evals[x][cv] = pickle.load(f)
            # Summarise all metrics in a single table with rows for each runs and CV partition.
            self.runparams[gs_id] = runparams
            self.run_ids_clean[gs_id] = run_ids_clean
            self.evals[gs_id] = evals

            target_label = runparams[run_ids_clean[0]]['target_label']
            loss_keys = list(evals.values())[0]['cv0']['train'].keys()
            target_loss = target_label + '_loss'
            target_metric = [key for key in loss_keys
                             if target_label in key and key != target_loss]
            if len(target_metric) == 0:
                target_metric = [key for key in loss_keys
                                 if not 'loss' in key and key != 'mmd'][0]
                target_metric_name = target_label + '_' + target_metric
            else:
                target_metric_name = target_metric[0]
                target_metric = '_'.join(target_metric[0].split('_')[1:])

            metrics = np.unique(np.concatenate([
                np.asarray(list(evals[x][cv_ids[0]]["train"].keys()))
                for x in run_ids_clean
            ])).tolist()
            

            self.summary_table.append(pd.concat([
                pd.DataFrame(dict(list({
                                           "activation": [str(runparams[x]['activation']) for x in run_ids_clean]
                                           if 'activation' in list(runparams[x].keys()) else "none",
                                           "aggregation": [str(runparams[x]['aggregation']) for x in run_ids_clean]
                                           if 'aggregation' in list(runparams[x].keys()) else "none",
                                           "batch_size": [str(runparams[x]['batch_size']) for x in run_ids_clean]
                                           if 'batch_size' in list(runparams[x].keys()) else "none",
                                           "covariate_selection": [str(runparams[x]['covar_selection']) for x in
                                                                   run_ids_clean]
                                           if 'covariate_selection' in list(runparams[x].keys()) else "none",
                                           "depth_feature_embedding": [str(runparams[x]['depth_feature_embedding']) for
                                                                       x in run_ids_clean]
                                           if 'depth_feature_embedding' in list(runparams[x].keys()) else "none",
                                           "depth_final_dense": [str(runparams[x]['depth_final_dense']) for x in
                                                                 run_ids_clean]
                                           if 'depth_final_dense' in list(runparams[x].keys()) else "none",
                                           "depth": [str(runparams[x]['depth']) for x in run_ids_clean]
                                           if 'depth' in list(runparams[x].keys())
                                           else [str(runparams[x]['de']) for x in run_ids_clean]
                                           if 'de' in list(runparams[x].keys())
                                           else "none",
                                           "dropout": [str(runparams[x]['dropout_rate']) for x in run_ids_clean]
                                           if 'dropout_rate' in list(runparams[x].keys()) else "none",
                                           "dropout_rate_pooling": [str(runparams[x]['dropout_rate_pooling']) for x in
                                                                    run_ids_clean]
                                           if 'dropout_rate_pooling' in list(runparams[x].keys()) else "none",
                                           "feature_space": [str(runparams[x]['node_feature_space_id']) for x in
                                                             run_ids_clean]
                                           if 'node_feature_space_id' in list(runparams[x].keys())
                                           else 'type' if 'reg' in x
                                           else "none",
                                           "final_pooling": [str(runparams[x]['final_pooling']) for x in run_ids_clean]
                                           if 'final_pooling' in list(runparams[x].keys()) else "none",
                                           "l2": [str(runparams[x]['l2_reg']) for x in run_ids_clean]
                                           if 'l2_reg' in list(runparams[x].keys())
                                           else [str(runparams[x]['l2']) for x in run_ids_clean] if 'l2' in list(
                                               runparams[x].keys())
                                           else "none",
                                           "loss_weight": [str(runparams[x]['loss_weight_others']) for x in
                                                           run_ids_clean]
                                           if 'loss_weight_others' in list(runparams[x].keys())
                                           else [str(runparams[x]['lw']) for x in run_ids_clean] if 'lw' in list(
                                               runparams[x].keys())
                                           else "none",
                                           "loss_weight_types": [str(runparams[x]['loss_weight_type']) for x in
                                                                 run_ids_clean]
                                           if 'loss_weight_type' in list(runparams[x].keys()) else "none",
                                           "lr": [str(runparams[x]['learning_rate']) for x in run_ids_clean]
                                           if 'learning_rate' in list(runparams[x].keys())
                                           else [str(runparams[x]['lr']) for x in run_ids_clean] if 'lr' in list(
                                               runparams[x].keys())
                                           else "none",
                                           "max_dist": [str(runparams[x]['max_dist']) for x in run_ids_clean]
                                           if 'max_dist' in list(runparams[x].keys()) else "none",
                                           "mmd": [str(runparams[x]['lambda_mmd']) for x in run_ids_clean]
                                           if 'lambda_mmd' in list(runparams[x].keys()) else "none",
                                           "model_class": [str(runparams[x]['model_class']) for x in run_ids_clean]
                                           if 'model_class' in list(runparams[x].keys())
                                           else 'reg' if 'reg' in x
                                           else "none",
                                           "node_feature_transformation": [
                                               str(runparams[x]['node_feature_transformation']) for x in run_ids_clean]
                                           if 'node_feature_transformation' in list(runparams[x].keys()) else "none",
                                           "node_fraction": [str(runparams[x]['node_fraction']) for x in run_ids_clean]
                                           if 'node_fraction' in list(runparams[x].keys()) else "none",
                                           "radius_steps": [str(runparams[x]['radius_steps']) for x in run_ids_clean]
                                           if 'radius_steps' in list(runparams[x].keys()) else "none",
                                           "n_clusters": [str(runparams[x]['n_clusters']) for x in run_ids_clean]
                                           if 'n_clusters' in list(runparams[x].keys()) else "none",
                                           "entropy_weight": [str(runparams[x]['entropy_weight']) for x in
                                                              run_ids_clean]
                                           if 'entropy_weight' in list(runparams[x].keys()) else "none",
                                           "multitask_setting": [str(runparams[x]['multitask_setting']) for x in
                                                                 run_ids_clean]
                                           if 'multitask_setting' in list(runparams[x].keys()) else "none",
                                           "self_supervision_mode": [str(runparams[x]['self_supervision_mode']) for x in
                                                                     run_ids_clean]
                                           if 'self_supervision_mode' in list(runparams[x].keys()) else "none",
                                           "run_id": run_ids_clean,
                                           "target_label": [str(runparams[x]['target_label']) for x in run_ids_clean]
                                           if 'target_label' in list(runparams[x].keys()) else "none",
                                           "width": [str(runparams[x]['width']) for x in run_ids_clean]
                                           if 'width' in list(runparams[x].keys()) else "none",
                                           "cv": cv,
                                           "gs_id": gs_id
                                       }.items()) +
                                  list(dict([
                                      ("train_" + m, [
                                          evals[x][cv]["train"][m] if m in evals[x][cv]["train"].keys()
                                          else np.nan for x in run_ids_clean
                                      ]) for m in metrics
                                  ]).items()) +
                                  list(dict([
                                      ("test_" + m, [
                                          evals[x][cv]["test"][m] if evals[x][cv]["test"] is not None and m in evals[x][cv]["test"].keys()
                                          else np.nan for x in run_ids_clean
                                      ]) for m in metrics
                                  ]).items()) +
                                  list(dict([
                                  ("val_" + m, [
                                          evals[x][cv]["val"][m] if evals[x][cv]["val"] is not None and m in evals[x][cv]["val"].keys()
                                          else np.nan for x in run_ids_clean
                                      ]) for m in metrics
                                  ]).items()) +
                                  list(dict([
                                      ("all_" + m, [
                                          evals[x][cv]["all"][m] if m in evals[x][cv]["all"].keys()
                                          else np.nan for x in run_ids_clean
                                      ]) for m in metrics
                                  ]).items())
                                  )) for cv in cv_ids
            ]))

            # Summarise all metrics in a single table with rows for each runs and CV partition.
            if len(run_ids_clean) == 0:
                raise ValueError("no complete runs found")
            print("loaded %s: %i runs with %i-fold cross validation" %
                  (gs_id, len(self.run_ids_clean[gs_id]), len(cv_ids)))

        self.summary_table = pd.concat(self.summary_table)
        self.summary_table["feature_space"] = ['molecular' if c == 'standard' else
                                               'types' if c == 'type' else
                                               c for c in self.summary_table["feature_space"]]
        self.summary_table["covariate_selection"] = [
            "_".join(x) if x is not None else "None" for x in self.summary_table["covariate_selection"].values
        ]
        self.summary_table['adj_type'] = [
            'scaled' if elem == 'gcn' else 'spectral' if elem == 'gcnspectral' else 'none' for elem in
            self.summary_table['model_class'].values
        ]
        self.summary_table['model_type'] = [
            'graph' if 'gcn' in model or 'gat' in model else 'reference' for model in
            self.summary_table['model_class'].values
        ]
        self.summary_table["model_class"] = [
            'GCN' if c == 'gcn' else
            'GCNSS' if c == 'gcnss' else
            'GCNSSO' if c == 'gcnsso' else
            'GSUM' if c == 'gsum' else
            'GAT' if c == 'gatcomplex' else
            'MI' if c == 'mi' else
            'MLP' if c == 'reg' else
            'DISP' if c == 'regdisp' else
            c for c in self.summary_table["model_class"]
        ]

        self.summary_table['multitask'] = [
            'no' if setting == 'target' else 'yes' for setting in self.summary_table['multitask_setting'].values
        ]

    @property
    def cv_keys(self) -> List[str]:
        """
        Returns keys of cross-validation used in dictionaries in this class.

        :return: list of string keys
        """
        return np.unique(self.summary_table["cv"].values).tolist()

    def get_best_model_id(
            self,
            subset_hyperparameters: List[Tuple[str, str]] = [],
            metric_select: str = "loss",
            partition_select: str = "val",
            cv_mode: str = "mean"
    ):
        """
        :param subset_hyperparameters:
        :param metric_select: Metric to use for ranking models.
        :param cv_mode: Summary mode of distribution of metric over partitions in cross validation. Out of:

            - "mean"
            - "median"
            - "max"
            - "min"
        :param partition_select: "train" or "eval" or "test" partition of data to collect metric from.
        :return:
        """
        if metric_select.endswith('acc') or \
                metric_select.endswith('accuracy'):
            ascending = False
            if cv_mode == "min":
                raise Warning("selected cv_mode min with metric_id acc, likely not intended")
        elif metric_select.endswith('loss') or metric_select.endswith('crossentropy'):
            ascending = True
            if cv_mode == "max":
                raise Warning("selected cv_mode max with metric_id loss, likely not intended")
        elif metric_select.endswith('mse') or \
                metric_select.endswith('mae') or \
                metric_select.endswith('survival'):
            ascending = True
            if cv_mode == "max":
                raise Warning("selected cv_mode max with metric_id mae, likely not intended")
        else:
            raise ValueError("measure %s not recognized" % metric_select)
        if partition_select not in ["test", "val", "train"]:
            raise ValueError("partition %s not recognised" % partition_select)
        metric_select = partition_select + "_" + metric_select
        summary_table = self.summary_table.copy()
        for x, y in subset_hyperparameters:
            if np.sum(summary_table[x].values == y) == 0:
                print(
                    "subset was empty, available values for %s are %s, given was %s" %
                    (x, str(np.unique(summary_table[x].values).tolist()), str(y))
                )
            summary_table = summary_table.loc[summary_table[x].values == y, :]
        if cv_mode.lower() == "mean":
            best_model = summary_table.groupby("run_id", as_index=False)[metric_select].mean(). \
                sort_values([metric_select], ascending=ascending)
        elif cv_mode.lower() == "median":
            best_model = summary_table.groupby("run_id", as_index=False)[metric_select].median(). \
                sort_values([metric_select], ascending=ascending)
        elif cv_mode.lower() == "max":
            best_model = summary_table.groupby("run_id", as_index=False)[metric_select].max(). \
                sort_values([metric_select], ascending=ascending)
        elif cv_mode.lower() == "min":
            best_model = summary_table.groupby("run_id", as_index=False)[metric_select].min(). \
                sort_values([metric_select], ascending=ascending)
        else:
            raise ValueError("cv_mode %s not recognized" % cv_mode)
        if best_model.shape[0] > 0:
            pass
            # print(metric_select + ": " + str(best_model[metric_select].values[0]))
        best_model_id = best_model['run_id'].values[0] if best_model.shape[0] > 0 else None
        best_cv = summary_table[summary_table["run_id"] == best_model_id]. \
            sort_values([metric_select], ascending=ascending)['cv'].values[0] if best_model_id is not None else None
        best_gs_id = summary_table[summary_table["run_id"] == best_model_id]. \
            sort_values([metric_select], ascending=ascending)['gs_id'].values[0] if best_model_id is not None else None

        return best_gs_id, best_model_id, best_cv

    def copy_best_model(
            self,
            dst: str = "best",
            metric_select: str = "loss",
            partition_select: str = "val",
            cv_mode: str = "mean"
    ):
        """
        Extract all relavant files from grid search to re-execute best models.

        Copies best model saved as .h5 and data set partitions into target directory.

        :param dst: Target directory. Taken as relative path to grid search if does not start with "/".
        :param metric_select: Metric to use for ranking models.
        :param cv_mode: Summary mode of distribution of metric over partitions in cross validation. Out of:

            - "mean"
            - "median"
            - "max"
            - "min"
        :return:
        """
        from shutil import copyfile
        if dst[0] != "/":
            dst = self.source_path[self.gs_ids[0]] + self.gs_ids[0] + "/" + dst + "/"
        _, run_id, _ = self.get_best_model_id(  # _ is best cv, maybe use it?
            metric_select=metric_select,
            partition_select=partition_select,
            cv_mode=cv_mode
        )
        cvs = self.summary_table.loc[self.summary_table["run_id"].values == run_id, "cv"].values
        src = self.source_path[self.gs_ids[0]] + self.gs_ids[0] + "/results/"
        print("copying model files from %s to %s" % (src, dst))
        for cv in cvs:
            fn_model = run_id + "_" + cv + "_model.h5"
            copyfile(src + fn_model, dst + fn_model)
            fn_idx = run_id + "_" + cv + "_indices.pickle"
            copyfile(src + fn_idx, dst + fn_idx)

    def get_info(
            self,
            model_id,
            gs_id: str,
            expected_pickle: list = ['evaluation', 'history', 'hyperparam', 'predictions'],
            load_labels: List[str] = ["grade"]  # for ionpath "GRADE"
    ):
        indir = self.source_path[gs_id] + gs_id + "/results/"
        # Check that all files are present:
        cv_ids = np.sort(np.unique([
            x.split("_")[-2] for x in os.listdir(indir)
            if x.split("_")[-1].split(".")[0] == "history"
        ]))
        for cv in cv_ids:
            # Check pickled files:
            for suffix in expected_pickle:
                fn = model_id + "_" + cv + "_" + suffix + ".pickle"
                if not os.path.isfile(indir + fn):
                    raise ValueError("file %s missing" % suffix)
        info = {}
        for cv in cv_ids:
            info[cv] = {}
            for suffix in expected_pickle:
                fn = model_id + "_" + cv + "_" + suffix + ".pickle"
                with open(indir + fn, 'rb') as f:
                    data = pickle.load(f)
                    # TODO: for new data include the following
                    #                     if suffix == 'predictions':
                    #                         data = {key: {'_'.join(k.split('/')[0].split('_')[:-1]) if '_'.join(
                    #                             k.split('/')[0].split('_')[:-1]) != '' else k.split('/')[0]: v
                    #                                       for k, v in value.items()} for key, value in data.items()}
                    info[cv][suffix] = data
        fn = model_id + "_runparams.pickle"
        with open(indir + fn, 'rb') as f:
            info["runparams"] = pickle.load(f)

        self.info = info

        # Load files that are shared across a grid search.
        #idx_test = {}
        #idx_train = {}
        #idx_val = {}
        #patient_idx_test = {}
        #patient_idx_train = {}
        #patient_idx_val = {}
        #for cv in cv_ids:
        #    with open(indir + model_id + "_" + cv + '_img_indices.pickle', 'rb') as f:
        #        idx_partitions = pickle.load(f)
        #    idx_test[cv] = idx_partitions['test']
        #    idx_train[cv] = idx_partitions['train']
        #    idx_val[cv] = idx_partitions['val']
        #    if os.path.isfile(indir + model_id + "_" + cv + '_patient_indices.pickle'):
        #        with open(indir + model_id + "_" + cv + '_patient_indices.pickle', 'rb') as f:
        #            idx_patient_partitions = pickle.load(f)
        #        patient_idx_test[cv] = idx_patient_partitions['test']
        #        patient_idx_train[cv] = idx_patient_partitions['train']
        #        patient_idx_val[cv] = idx_patient_partitions['val']
        #self.idx_test = idx_test
        #self.idx_train = idx_train
        #self.idx_val = idx_val
        #self.patient_idx_test = patient_idx_test
        #self.patient_idx_train = patient_idx_train
        #self.patient_idx_val = patient_idx_val

        true_labels = {}
        patient_dict = {}
        for cv in cv_ids:
           with open(indir + model_id + "_" + cv + '_datainfo.pickle', 'rb') as f:
               datainfo = pickle.load(f)
           true_labels[cv] = {}
           true_labels[cv]['test'] = {}
           true_labels[cv]['train'] = {}
           true_labels[cv]['val'] = {}
           for l in load_labels:
               true_labels[cv]['test'][l] = np.concatenate(datainfo['true_targets']['test'][l], axis=0) if len(datainfo['true_targets']['test'][l]) > 0 else None
               true_labels[cv]['train'][l] = np.concatenate(datainfo['true_targets']['train'][l], axis=0)
               true_labels[cv]['val'][l] = np.concatenate(datainfo['true_targets']['val'][l], axis=0) if len(datainfo['true_targets']['val'][l]) > 0 else None
           patient_dict[cv] = datainfo['patient_dict']
        self.true_labels = true_labels
        self.patient_dict = patient_dict

    def _get_labels(
            self,
            cv_key,
            target_label: str = "grade",
            partition_show: str = "test"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns observed labels and predicted labels.

        :param cv_idx: Index of cross-validation to plot confusion matrix for.
        :param target_label:
        :param partition:
        :return: Tuple of predictions and labels. Each as numpy array
        """
        cv = cv_key
        
        predictions = self.info[cv]['predictions'][partition_show][target_label]
        labels = self.true_labels[cv][partition_show][target_label]
        return predictions, labels

    def _get_labels_survival(
            self,
            cv_key,
            target_label: str = "DFSmonth",
            partition_show: str = "test"
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns observed labels, predicted labels and censoring

        :param cv_idx: Index of cross-validation to plot confusion matrix for.
        :param metric_show:
        :param partition_show:
        :return: Tuple of predictions, labels and censoring. Each as numpy array
        """
        predictions, labels = self._get_labels(
            cv_key=cv_key,
            target_label=target_label,
            partition_show=partition_show
        )
        if len(labels.shape) != 2:
            raise ValueError("expected 2D label tensor for survival data, got %iD" % len(labels.shape))
        if labels.shape[1] != 2:
            raise ValueError("expected 2nd dimension of labels to be 2 for survival data, got %i" % labels.shape[1])
        censoring = labels[:, 1]
        labels = labels[:, 0]
        return predictions[:, 0], labels, censoring

    def _get_confusion_matrix(
            self,
            cv_key,
            target_label: str = "grade",
            partition_show: str = "test",
            small: bool = False
    ) -> np.ndarray:
        """
        Returns confusion matrix of categorical prediction problem.

        :param cv_idx: Index of cross-validation to plot confusion matrix for.
        :param metric_show:
        :param partition_show:
        :return: Confusion matrix
        """
        import sklearn
        predictions, labels = self._get_labels(
            cv_key=cv_key,
            target_label=target_label,
            partition_show=partition_show
        )
        nr_labels = predictions.shape[1]
        labels = np.reshape(labels, predictions.shape)
        labels = np.argmax(labels, axis=1)
        predictions = np.argmax(predictions, axis=1)

        if small:
            l = range(1, nr_labels)
            labels[labels == 0] = 1
            predictions[predictions == 0] = 1
        else:
            l = range(0, nr_labels)

        return sklearn.metrics.confusion_matrix(
            y_true=labels,
            y_pred=predictions,
            labels=l
        )

    def plot_confusion_matrix(
            self,
            target_label: str = "grade",
            partition_show: str = "test",
            sum_across_cv: bool = False,
            mean_across_cv: bool = False,
            save: Union[str, None] = None,
            suffix: str = "_confusion_matrix.pdf",
            show: bool = True,
            return_axs: bool = False,
            small: bool = False,
            x_ticks=['1', '2', '3']
    ):
        """
        Plots the confusion matrix between the observed and predicted labels for all images.
        Use to visualise categorical predictors.
        :param metric_show:
        :param partition_show:
        :param save: Whether (if not None) and where (path as string given as save) to save plot.
        :param suffix: Suffix of file name to save to.
        :param show: Whether to display plot.
        :param return_axs: Whether to return axis objects.
        :param small: For basel_zurich, whether to join grade 1 and 2.
        :return:
        """
        import matplotlib.pyplot as plt
        from sklearn.metrics import ConfusionMatrixDisplay

        n_cv = len(self.info.keys()) - 1

        if not return_axs:
            plt.ioff()

        if sum_across_cv:
            fig, ax = plt.subplots(
                nrows=1, ncols=1,
                figsize=(3, 3)
            )
            res = []
            acc = []
            for i, cv_key in enumerate(list(self.info.keys())[:-1]):
                res_i = self._get_confusion_matrix(
                    cv_key=cv_key,
                    target_label=target_label,
                    partition_show=partition_show,
                    small=small
                )
                res.append(res_i)
                acc.append(sum(res_i.diagonal()) / np.sum(res_i))
            res = sum(res)
            confusion_plot = ConfusionMatrixDisplay(res, display_labels=x_ticks)
            confusion_plot.plot(ax=ax)
            ax.set_title(", ".join([str(round(acc_i, 2)) for acc_i in acc]), y=1.15)
            ax.images[0].colorbar.remove()
        elif mean_across_cv:
            fig, ax = plt.subplots(
                nrows=1, ncols=1,
                figsize=(5, 5)
            )
            res = []
            acc = []
            for i, cv_key in enumerate(list(self.info.keys())[:-1]):
                res_i = self._get_confusion_matrix(
                    cv_key=cv_key,
                    target_label=target_label,
                    partition_show=partition_show,
                    small=small
                )
                res.append(res_i)
                acc.append(sum(res_i.diagonal()) / np.sum(res_i))
            res = sum(res) / (i+1)
            confusion_plot = ConfusionMatrixDisplay(np.round(res, 2), display_labels=x_ticks)
            confusion_plot.plot(ax=ax)
            acc_mean = sum(res.diagonal()) / np.sum(res)
            ax.set_title(str(round(acc_mean, 2)), y=1.15)
            ax.images[0].colorbar.remove()
        else:
            fig, ax = plt.subplots(
                nrows=1, ncols=n_cv,
                figsize=(3 * n_cv, 3),
            )
            for i, cv_key in enumerate(list(self.info.keys())[:-1]):
                res = self._get_confusion_matrix(
                    cv_key=cv_key,
                    target_label=target_label,
                    partition_show=partition_show,
                    small=small
                )
                acc = sum(res.diagonal()) / np.sum(res)
                confusion_plot = ConfusionMatrixDisplay(res, display_labels=x_ticks)
                confusion_plot.plot(ax=ax[i])
                ax[i].set_title(str(cv_key) + ": " + str(round(acc, 2)), y=1.15)
                ax[i].images[0].colorbar.remove()

        # Save, show and return figure.
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        if save is not None:
            plt.grid(False)
            plt.tight_layout()
            plt.savefig(save + "_" + partition_show + suffix)
        if show:
            plt.grid(False)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.show()
        if return_axs:
            return ax
        else:
            plt.close(fig)
            plt.ion()
            return None

    def _get_confusion_matrix_patients(
            self,
            cv_key,
            target_label: str = "grade",
            partition_show: str = "test"
    ):
        cv = cv_key
        patient_dict_test = {idx: patient for idx, patient in self.patient_dict[cv].items() if
                             idx in set(self.idx_test[cv])}
        true = np.array([np.where(r == 1) for r in self.true_labels[cv][partition_show][target_label]]).flatten()
        patient_true = {patient: -1 for patient in set(patient_dict_test.values())}
        # Find true label for a patient. If there are images for one patient with different labels, create a new patient.
        # (Mainly for basel_zurich, there are patients with images of grade 1/2/3 and images of metastasis.)
        for i, idx in enumerate(self.idx_test[cv]):
            if patient_true[patient_dict_test[idx]] == -1:
                patient_true[patient_dict_test[idx]] = true[i]
            elif patient_true[patient_dict_test[idx]] != true[i]:
                patient_true['new_' + patient_dict_test[idx]] = true[i]
                patient_dict_test[idx] = 'new_' + patient_dict_test[idx]
        probs = self.info[cv]['predictions'][partition_show][target_label]
        preds = np.argmax(probs, axis=1)
        # gather all predictions for one patient
        patient_pred = {patient: [] for patient in set(patient_dict_test.values())}
        for i, idx in enumerate(self.idx_test[cv]):
            patient_pred[patient_dict_test[idx]].append(preds[i])
        # If all images for one patient are correctly classified, the patient is counted as correctly classified,
        # if there is at least one of his images misclassified, the prediction for the patient is the one wrong
        # prediction that was made most often.
        res = np.zeros((4, 4))
        for patient in set(patient_dict_test.values()):
            tr = patient_true[patient]
            pred = patient_pred[patient]
            if sum(np.array(pred) == tr) == len(pred):
                res[tr][tr] += 1
            else:
                pred = [i for i in pred if i != tr]
                values, counts = np.unique(pred, return_counts=True)
                res[tr][values[np.argmax(counts)]] += 1
        res = np.ndarray.astype(res, int)
        return res

    def plot_confusion_matrix_patients(
            self,
            target_label: str = "grade",
            partition_show: str = "test",
            save: Union[str, None] = None,
            suffix: str = "_confusion_matrix_patients.pdf",
            show: bool = True,
            return_axs: bool = False
    ):
        """
        Plots the confusion matrix between the observed and predicted labels per patient.

        Use to visualise categorical predictors.

        :param metric_show:
        :param partition_show:
        :param save: Whether (if not None) and where (path as string given as save) to save plot.
        :param suffix: Suffix of file name to save to.
        :param show: Whether to display plot.
        :param return_axs: Whether to return axis objects.
        :return:
        """
        import matplotlib.pyplot as plt
        n_cv = len(self.cv_keys)

        plt.ioff()
        fig = plt.figure(figsize=(5 * n_cv, 5))
        for i, cv_key in enumerate(self.cv_keys):
            ax = fig.add_subplot(1, n_cv, i + 1)
            res = self._get_confusion_matrix_patients(
                cv_key=cv_key,
                target_label=target_label,
                partition_show=partition_show
            )
            ax.matshow(res)

            for (i, j), z in np.ndenumerate(res):
                ax.text(j, i, z, ha='center', va='center',
                        bbox=dict(facecolor='white', alpha=0.3, edgecolor='white'))
            ax.set_xticklabels([''] + ['1', '2', '3', 'M'])
            ax.set_yticklabels([''] + ['1', '2', '3', 'M'])
            plt.xlabel('Main Prediction', labelpad=20)
            plt.ylabel('True Label')
            plt.title('Confusion matrix for ' + str(cv_key))
        plt.suptitle('Confusion Matrix for prediction per Patient', y=1.1)

        # Save, show and return figure.
        plt.tight_layout()
        if save is not None:
            plt.savefig(save + suffix)

        if show:
            plt.show()

        plt.close(fig)
        plt.ion()

        if return_axs:
            return ax
        else:
            return None

    def _get_heatmap(
            self,
            cv_key,
            target_label: str = "grade",
            partition_show: str = "test"
    ):
        cv = cv_key
        patient_dict_test = {idx: patient for idx, patient in self.patient_dict[cv].items() if
                             idx in set(self.idx_test[cv])}
        true = np.array([np.where(r == 1) for r in self.true_labels[cv][partition_show][target_label]]).flatten()
        patient_true = {patient: -1 for patient in set(patient_dict_test.values())}
        for i, idx in enumerate(self.idx_test[cv]):
            if patient_true[patient_dict_test[idx]] == -1:
                patient_true[patient_dict_test[idx]] = true[i]
            elif patient_true[patient_dict_test[idx]] != true[i]:
                patient_true['new_' + patient_dict_test[idx]] = true[i]
                patient_dict_test[idx] = 'new_' + patient_dict_test[idx]

        prediction_per_patient = {patient: np.zeros(4) for patient in set(patient_dict_test.values())}
        probs = self.info[cv]['predictions'][partition_show][target_label]
        preds = np.argmax(probs, axis=1)
        preds = np.eye(4)[preds]
        for i, idx in enumerate(self.idx_test[cv]):
            prediction_per_patient[patient_dict_test[idx]] += np.array(preds[i])
        res = [np.concatenate([preds / sum(preds), [patient_true[patient]]]) for patient, preds in
               prediction_per_patient.items()]
        res = np.array(res)
        res = res[res[:, -1].argsort()]
        return res

    def plot_heatmap(
            self,
            target_label: str = "grade",
            partition_show: str = "test",
            save: Union[str, None] = None,
            suffix: str = "_confusion_matrix.pdf",
            show: bool = True,
            return_axs: bool = False
    ):
        """
        Plots the relative predictions per patient together with the main true label of the patient.

        :param metric_show:
        :param partition_show:
        :param save: Whether (if not None) and where (path as string given as save) to save plot.
        :param suffix: Suffix of file name to save to.
        :param show: Whether to display plot.
        :param return_axs: Whether to return axis objects.
        :return:
        """
        import matplotlib.pyplot as plt
        n_cv = len(self.cv_keys)

        plt.ioff()
        fig = plt.figure(figsize=(5 * n_cv, 10))
        for i, cv_key in enumerate(self.cv_keys):
            ax = fig.add_subplot(1, n_cv, i + 1)
            res = self._get_heatmap(
                cv_key=cv_key,
                target_label=target_label,
                partition_show=partition_show
            )
            ax.matshow(res)

            ax.set_xticklabels(['', '1', '2', '3', 'M', 'T'])
            plt.xlabel('Relative Prediction per Patient', labelpad=20)
            plt.ylabel('Patient')

        plt.suptitle('Heatmap for Relative Predictions per Patient', y=1.01)

        # Save, show and return figure.
        plt.tight_layout()
        if save is not None:
            plt.savefig(save + suffix)

        if show:
            plt.show()

        plt.close(fig)
        plt.ion()

        if return_axs:
            return ax
        else:
            return None

    def plot_scatter(
            self,
            target_label: str = "DFSmonth",
            partition: str = "test",
            save: Union[str, None] = None,
            suffix: str = "_scatter.pdf",
            show: bool = True,
            return_axs: bool = False
    ):
        """
        Plots scatter plot between observed and predicted labels.

        Use to visualise continuous predictors.

        :param target_label:
        :param partition:
        :param save: Whether (if not None) and where (path as string given as save) to save plot.
        :param suffix: Suffix of file name to save to.
        :param show: Whether to display plot.
        :param return_axs: Whether to return axis objects.
        :return:
        """
        import seaborn as sns
        import matplotlib.pyplot as plt
        n_cv = len(self.info.keys()) - 1

        plt.ioff()
        fig = plt.figure(figsize=(3 * n_cv, 3))
        for i, cv_key in enumerate(list(self.info.keys())[:n_cv]):
            ax = fig.add_subplot(1, n_cv, i + 1)
            predictions, labels, censoring = self._get_labels_survival(
                cv_key=cv_key,
                target_label=target_label,
                partition_show=partition
            )
            sns_data = pd.DataFrame({
                "pred": predictions,
                "obs": labels,
                "censored": ['yes' if c == 0 else 'no' for c in censoring]
            })
            sns.scatterplot(
                data=sns_data,
                x="obs",
                y="pred",
                hue="censored",
                ax=ax
            )
            ax.set_xlabel("observed %s" % target_label)
            ax.set_ylabel("predicted %s" % target_label)
            corr = np.corrcoef(
                sns_data["pred"].values[np.logical_not(np.isnan(sns_data["obs"].values.astype(np.float)))],
                sns_data["obs"].values[np.logical_not(np.isnan(sns_data["obs"].values.astype(np.float)))].astype(
                    np.float)
            )[0, 1]
            ax.set_title("%s: R^2=%s" % (cv_key, str(round(corr, 3))))
            if i > 0:
                ax.get_legend().remove()
            left, right = plt.xlim()
            plt.ylim(bottom=left)
            bottom, top = plt.ylim()
            if right > top:
                plt.ylim(top=right)
            if top > 4:
                plt.ylim(top=4)

        # Save, show and return figure.
        plt.tight_layout()
        if save is not None:
            plt.savefig(save + "_" + partition + suffix)

        if show:
            plt.show()

        plt.close(fig)
        plt.ion()

        if return_axs:
            return ax
        else:
            return None

    def plot_best_model_by_hyperparam(
            self,
            partition_show: str,
            metric_show: str,
            partition_select: str,
            metric_select: str,
            param_hue: str,
            param_x: str,
            cv_mode: Union[str, List[str]] = "mean",
            subset_hyperparam=[],
            show_swarm: bool = True,
            save: Union[str, None] = None,
            suffix: str = "_hyperparam.pdf",
            show: bool = True,
            return_axs: bool = False,
            panel_width: float = 5,
            panel_height: float = 3,
            xrot: float = 0,
            ttest: bool = False,
            types: bool = False,
            dispersion: bool = False,
            feature_space: bool = False,
            multitask: bool = False,
            return_summary_table: bool = False,
            
    ):
        """
        Produces boxplots for all hyperparameters with multiple values. For every value for that hyperparameter
        the best model is chosen and all cvs are plotted.

        :param partition_show: "train" or "eval" or "test" partition of data.
        :param metric_show: Metric to plot.
        :param param_x: Hyper-parameter for x-axis partition.
        :param param_hue: Hyper-parameter for hue-axis partition.
        :param cv_mode: Summary mode of distribution of metric over partitions in cross validation. Out of:

            - "mean"
            - "median"
            - "max"
            - "min":param save: Whether (if not None) and where (path as string given as save) to save plot.
        :param suffix: Suffix of file name to save to.
        :param show: Whether to display plot.
        :param return_axs: Whether to return axis objects.
        :param panel_width:
        :param panel_height:
        :return:
        """
        import seaborn as sns
        import matplotlib.pyplot as plt
        params = [
            param for param in param_x
            if len(np.unique(self.summary_table[param])) > 1 and param != param_hue
        ]
        print("selected %s" % str(params))

        if isinstance(cv_mode, str):
            cv_mode = [cv_mode]

        plt.ioff()
        fig, ax = plt.subplots(
            nrows=len(params), ncols=len(cv_mode),
            figsize=(panel_width * len(cv_mode), panel_height * len(params)),
            sharey='col'
        )
        for i, param in enumerate(params):
            # Plot each metric:
            params_x_unique = np.sort(np.unique(self.summary_table[param].values))
            params_hue_unique = np.sort(np.unique(self.summary_table[param_hue].values))
            for j, cvm in enumerate(cv_mode):
                run_ids = []
                summary_table = self.summary_table.copy()
                for x in params_x_unique:
                    empty = True
                    for hue in params_hue_unique:
                        _, run_id_temp, _ = self.get_best_model_id(
                            subset_hyperparameters=[(param, x), (param_hue, hue)] + subset_hyperparam,
                            partition_select=partition_select,
                            metric_select=metric_select,
                            cv_mode=cvm
                        )
                        if run_id_temp is not None:
                            run_ids.append(run_id_temp)
                            empty = False
                    if empty:
                        params_x_unique = [p for p in params_x_unique if p != x]
                if len(params_x_unique) > 1:
                    summary_table = summary_table.loc[np.array([x in run_ids for x in summary_table["run_id"].values]),
                                    :].copy()
                    summary_table.sort_values([param, param_hue], inplace=True)
                    #                     summary_table[param_hue] = pd.Categorical(
                    #                         summary_table[param_hue].values,
                    #                         categories=np.sort(np.unique(summary_table[param_hue].values))
                    #                     )
                    ycol = partition_show + "_" + metric_show
                    if len(cv_mode) == 1 and len(params) == 1:
                        ax = np.array([ax])
                    if param == 'depth':
                        order = ['1', '2', '3', '5', '10']
                    elif param == 'node_feature_transformation':
                        order = ['none', 'standardize_globally', 'standardize_per_image', 'rank_per_image']
                    elif param == 'multitask_setting':
                        order = ['target', 'small', 'large']
                    elif param == 'l2':
                        order = ['0.0', '1e-06', '0.001', '1.0']
                    elif param == 'model_class':
                        order = ['MLP', 'MI', 'DISP', 'GSUM', 'GCN', 'GCNSS', 'GCNADVANCED', 'GCNII', 'GAT', 'GATSINGLE', 'GATCOMPLEX']
                    elif param == 'mmd':
                        order = ['0.0', '1.0', '5.0', '10.0']
                    elif param == 'adj_type':
                        order = ['scaled', 'spectral']
                    else:
                        order = None
                    if order is not None:
                        order = [o for o in order if o in params_x_unique]
                    # order = ['MLP', 'MI', 'GCN', 'GCNSS']
                    # order = [o for o in order if o in params_x_unique]
                    # order = [o for o in order if o in params_x_unique]
                    if return_summary_table:
                        ax=ax[i, j] if len(ax.shape) == 2 else ax[i] if len(ax.shape) == 1 else ax
                        summary_table["partition_select"] = np.full(len(summary_table), partition_select)
                        return summary_table, param, ycol, order, ax
                    means = summary_table.groupby([param])[ycol].mean()
                    print(f"{means=}")
                    bp = sns.boxplot(
                        x=param, hue=param_hue, y=ycol,
                        order=order,
                        data=summary_table, ax=ax[i, j] if len(ax.shape) == 2 else ax[i] if len(ax.shape) == 1 else ax,
                        whis=1, 
                    )
                    if ttest:
                        from statannot import add_stat_annotation
                        if feature_space:
                            add_stat_annotation(data=summary_table, x=param, y=ycol, hue=param_hue,
                                            ax=ax[i, j] if len(ax.shape) == 2 else ax[i] if len(ax.shape) == 1 else ax, 
                                            box_pairs=[
                                                        (("MLP", "types"), ("MLP", "molecular")),
                                                        (("MI", "types"), ("MI", "molecular")),
                                                        (("GCN", "types"), ("GCN", "molecular")),
                                                        (("GCNSS", "types"), ("GCNSS", "molecular")),
                                                        ],
                                            order=order,
                                            test='t-test_ind', text_format='star', loc='inside', verbose=2, comparisons_correction=None)
                        elif types:
                            if dispersion:
                                    add_stat_annotation(data=summary_table, x=param, y=ycol, hue=param_hue,
                                                ax=ax[i, j] if len(ax.shape) == 2 else ax[i] if len(ax.shape) == 1 else ax, 
                                                box_pairs=[
                                                            (("MLP", "types"), ("MI", "types")),
                                                            (("MI", "types"), ("GCN", "types")),
                                                            (("MI", "types"), ("GCNSS", "types")),
                                                            (("MI", "types"), ("DISP", "types")),
                                                            ],
                                                order=order,
                                                test='t-test_ind', text_format='star', loc='inside', verbose=2, comparisons_correction=None)
                            else:
                                add_stat_annotation(data=summary_table, x=param, y=ycol, hue=param_hue,
                                                ax=ax[i, j] if len(ax.shape) == 2 else ax[i] if len(ax.shape) == 1 else ax, 
                                                box_pairs=[
                                                            (("MLP", "types"), ("MI", "types")),
                                                            (("MI", "types"), ("GCN", "types")),
                                                            (("MI", "types"), ("GCNSS", "types")),
                                                            ],
                                                order=order,
                                                test='t-test_ind', text_format='star', loc='inside', verbose=2, comparisons_correction=None)
                        elif multitask:
                            add_stat_annotation(data=summary_table, x=param, y=ycol, hue=param_hue,
                                            ax=ax[i, j] if len(ax.shape) == 2 else ax[i] if len(ax.shape) == 1 else ax, 
                                            box_pairs=[
                                                        (("MLP", "target"), ("MLP", "small")),
                                                        (("MI", "target"), ("MI", "small")),
                                                        (("GCN", "target"), ("GCN", "small")),
                                                        (("GCNSS", "target"), ("GCNSS", "small")),
                                                        ],
                                            order=order,
                                            test='t-test_ind', text_format='star', loc='inside', verbose=2, comparisons_correction=None)
                        else:
                            add_stat_annotation(data=summary_table, x=param, y=ycol, hue=param_hue,
                                                ax=ax[i, j] if len(ax.shape) == 2 else ax[i] if len(ax.shape) == 1 else ax, 
                                                box_pairs=[
                                                            (("MLP", "molecular"), ("MI", "molecular")),
                                                            (("MI", "molecular"), ("GCN", "molecular")),
                                                            (("MI", "molecular"), ("GCNSS", "molecular")),
                                                            #(("MLP", "molecular"), ("GCNSS", "molecular")),
                                                            #(("GCN", "molecular"), ("GCNSS", "molecular")),
                                                            ],
                                                order=order,
                                                test='t-test_ind', text_format='star', loc='inside', verbose=2, comparisons_correction=None)
                    handles, labels = bp.get_legend_handles_labels()
                    if show_swarm:
                        sns.swarmplot(
                            x=param, hue=param_hue, y=ycol, order=order,
                            data=summary_table,
                            ax=ax[i, j] if len(ax.shape) == 2 else ax[i] if len(ax.shape) == 1 else ax,
                            # palette=[sns.color_palette("gray_r")[0], sns.color_palette("gray_r")[1]]
                            palette=['black']
                        )
                    if 'acc' in metric_show:
                        y_label = partition_show + ' accuracy'
                    elif 'mae' in metric_show:
                        y_label = partition_show + ' mae'
                    else:
                        y_label = ycol
                    ax[i].set_ylabel(y_label, rotation=90)
                    ax[i].set_xticklabels(ax[i].get_xticklabels(), rotation=xrot)
                    box = ax[i].get_position()
                    ax[i].set_position([box.x0, box.y0, box.width * 0.95, box.height])
                    ax[i].legend(loc='center left', bbox_to_anchor=(1, 0.5), handles=handles, labels=labels,
                                 title=param_hue)
        plt.ylim(bottom=0.0, top=1.1)
        plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

        # if param_x == ['model_class']:
        #     plt.legend(bbox_to_anchor=(1., 0.9), title=param_hue)

        # Save, show and return figure.
        plt.tight_layout()
        if save is not None:
            plt.tight_layout()
            plt.savefig(save + suffix)

        if show:
            plt.show()

        plt.close(fig)
        plt.ion()

        if return_axs:
            return ax
        else:
            return None

    def plot_best_model_by_hyperparam_metric(
            self,
            partitions_show_hue: str,
            metric_show: str,
            partition_select: str,
            metric_select: str,
            param_x: str,
            cv_mode: Union[str, List[str]] = "mean",
            subset_hyperparam=[],
            show_swarm: bool = True,
            save: Union[str, None] = None,
            suffix: str = "_hyperparam.pdf",
            show: bool = True,
            return_axs: bool = False,
            panel_width: float = 5,
            panel_height: float = 3,
            xrot: float = 0
    ):
        """
        Produces boxplots for all hyperparameters with multiple values. For every value for that hyperparameter
        the best model is chosen and all cvs are plotted.

        :param partition_show: "train" or "eval" or "test" partition of data.
        :param metric_show: Metric to plot.
        :param param_x: Hyper-parameter for x-axis partition.
        :param param_hue: Hyper-parameter for hue-axis partition.
        :param cv_mode: Summary mode of distribution of metric over partitions in cross validation. Out of:

            - "mean"
            - "median"
            - "max"
            - "min":param save: Whether (if not None) and where (path as string given as save) to save plot.
        :param suffix: Suffix of file name to save to.
        :param show: Whether to display plot.
        :param return_axs: Whether to return axis objects.
        :param panel_width:
        :param panel_height:
        :return:
        """
        import seaborn as sns
        import matplotlib.pyplot as plt
        params = [
            param for param in param_x
            if len(np.unique(self.summary_table[param])) > 1
        ]
        print("selected %s" % str(params))

        if isinstance(cv_mode, str):
            cv_mode = [cv_mode]

        plt.ioff()
        fig, ax = plt.subplots(
            nrows=len(params), ncols=len(cv_mode),
            figsize=(panel_width * len(cv_mode), panel_height * len(params)),
            sharey='col'
        )
        for i, param in enumerate(params):
            # Plot each metric:
            params_x_unique = np.sort(np.unique(self.summary_table[param].values))
            for j, cvm in enumerate(cv_mode):
                run_ids = []
                summary_table = self.summary_table.copy()
                for x in params_x_unique:
                    empty = True
                    _, run_id_temp, _ = self.get_best_model_id(
                        subset_hyperparameters=[(param, x)] + subset_hyperparam,
                        partition_select=partition_select,
                        metric_select=metric_select,
                        cv_mode=cvm
                    )
                    if run_id_temp is not None:
                        run_ids.append(run_id_temp)
                        empty = False
                    if empty:
                        params_x_unique = [p for p in params_x_unique if p != x]
                if len(params_x_unique) > 1:
                    summary_table = summary_table.loc[np.array([x in run_ids for x in summary_table["run_id"].values]),
                                    :].copy()
                    summary_table.sort_values([param], inplace=True)
                    value_vars = [partition + '_' + metric_show for partition in partitions_show_hue]
                    summary_table = pd.melt(summary_table, value_vars=value_vars, id_vars=param)
                    if len(cv_mode) == 1 and len(params) == 1:
                        ax = np.array([ax])
                    # if param == 'depth':
                    #     order = ['1', '2', '3', '5', '10']
                    # elif param == 'node_feature_transformation':
                    #     order = ['none', 'standardize_globally', 'standardize_per_image', 'rank_per_image']
                    # elif param == 'multitask_setting':
                    #     order = ['target', 'small', 'large']
                    # elif param == 'l2':
                    #     order = ['0.0', '1e-06', '0.001', '1.0']
                    # elif param == 'model_class':
                    #     order = ['MLP', 'MI', 'GSUM', 'GCN', 'GCNADVANCED', 'GCNII', 'GAT', 'GATSINGLE', 'GATCOMPLEX']
                    # elif param == 'mmd':
                    #     order = ['0.0', '1.0', '5.0', '10.0']
                    # elif param == 'adj_type':
                    #     order = ['scaled', 'spectral']
                    # else:
                    #     order = None
                    # if order is not None:
                    # order = [o for o in order if o in params_x_unique]
                    bp = sns.boxplot(
                        x=param, hue='variable', y='value',
                        data=summary_table, ax=ax[i, j] if len(ax.shape) == 2 else ax[i] if len(ax.shape) == 1 else ax,

                    )
                    handles, labels = bp.get_legend_handles_labels()
                    if show_swarm:
                        sns.swarmplot(
                            x=param, hue='variable', y='value',
                            data=summary_table,
                            ax=ax[i, j] if len(ax.shape) == 2 else ax[i] if len(ax.shape) == 1 else ax,
                        )
                    ax[i].set_ylabel(metric_show, rotation=90)
                    ax[i].set_xticklabels(ax[i].get_xticklabels(), rotation=xrot)
                    box = ax[i].get_position()
                    ax[i].set_position([box.x0, box.y0, box.width * 0.95, box.height])
                    ax[i].legend(loc='center left', bbox_to_anchor=(1, 0.5), handles=handles,
                                 labels=partitions_show_hue, title='partition')

        # if param_x == ['model_class']:
        #     plt.legend(bbox_to_anchor=(1., 0.9), title=param_hue)

        # Save, show and return figure.
        plt.tight_layout()
        if save is not None:
            plt.savefig(save + suffix)

        if show:
            plt.show()

        plt.close(fig)
        plt.ion()

        if return_axs:
            return ax
        else:
            return None

    def plot_training_history(
            self,
            cv_keys: Union[str, None] = None,
            log_loss: bool = False,
            plot_metrics=["loss", "val_loss"],
            plot_lr=False,
            ada_scale: float = 0.1,
            panel_width=10.,
            panel_height=6.,
            rename_metrics: dict = {},
            save: Union[str, None] = None,
            suffix: str = "_training_history.pdf",
            show: bool = True,
            ax=None
    ):
        """
        Plot train and validation loss during training and learning rate reduction.

        :param cv_key: Index of cross-validation to plot training history for.
        :param log_loss:
        :param save: Whether (if not None) and where (path as string given as save) to save plot.
        :param suffix: Suffix of file name to save to.
        :param show: Whether to display plot.
        :param return_axs: Whether to return axis objects.
        :return:
        """
        import seaborn as sns
        import matplotlib.pyplot as plt
        if plot_lr:
            plts = 2
        else:
            plts = 1
        input_axs = ax is not None
        if not input_axs:
            plt.ioff()
            fig, ax = plt.subplots(
                nrows=1, ncols=plts,
                figsize=(panel_width * plts, panel_height)
            )
        if cv_keys is None:
            cv_keys = list(self.info.keys())[:-1]
        sns_data = []
        for cv in cv_keys:
            sns_data_temp = pd.DataFrame(self.info[cv]["history"])
            for k, v in sns_data_temp.items():
                if "ada_" in k.lower():
                    sns_data_temp[k] = v * ada_scale
            sns_data_temp["epoch"] = np.arange(0, sns_data_temp.shape[0])
            sns_data_temp["cv"] = cv
            sns_data.append(sns_data_temp)
        sns_data = pd.concat(sns_data, axis=0)

        sns_data_loss = pd.concat([pd.DataFrame({
            "epoch": sns_data["epoch"].values,
            "cv": sns_data["cv"].values,
            "loss": np.log(sns_data[x].values) if log_loss else sns_data[x].values,
            "partition": x
        }) for i, x in enumerate(plot_metrics)], ignore_index=True)
        if len(rename_metrics) > 0:
            sns_data_loss["partition"] = [
                x if x not in rename_metrics.keys() else rename_metrics[x]
                for x in sns_data_loss["partition"].values
            ]
        if plot_lr:
            ax[0] = sns.lineplot(
                x="epoch", y="loss", hue="partition", style="cv",
                data=sns_data_loss, ax=ax[0]
            )
            if log_loss:
                ax[0].set_ylabel("log loss")
            sns_data_lr = pd.DataFrame({
                "epoch": sns_data["epoch"].values,
                "cv": sns_data["cv"].values,
                "lr": np.log(sns_data["lr"].values) / np.log(10)
            })
            ax[1] = sns.lineplot(
                x="epoch", y="lr", style="cv",
                data=sns_data_lr, ax=ax[1]
            )
            ax[1].set_ylabel("log10 learning rate")
        else:
            ax = sns.lineplot(
                x="epoch", y="loss", hue="partition", style="cv",
                data=sns_data_loss, ax=ax
            )
            if log_loss:
                ax.set_ylabel("log loss")
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.95, box.height])
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        # Save, show and return figure.
        if input_axs:
            pass
        else:
            plt.tight_layout()
            if save is not None:
                plt.tight_layout()
                plt.savefig(save + suffix)
            if show:
                plt.show()
            plt.close(fig)
            plt.ion()

    def plot_training_history_big(
            self,
            cv_keys: Union[str, None] = None,
            log_loss: bool = False,
            plot_metrics=["loss", "val_loss"],
            save: Union[str, None] = None,
            suffix: str = "_training_history_mt.pdf",
            show: bool = True,
            return_axs: bool = False
    ):
        """
        Plot train and validation loss during training and learning rate reduction.

        :param cv_key: Index of cross-validation to plot training history for.
        :param log_loss:
        :param save: Whether (if not None) and where (path as string given as save) to save plot.
        :param suffix: Suffix of file name to save to.
        :param show: Whether to display plot.
        :param return_axs: Whether to return axis objects.
        :return:
        """
        import seaborn as sns
        import matplotlib.pyplot as plt
        panel_width = 5
        panel_height = 3

        plt.ioff()
        if cv_keys is None:
            cv_keys = list(self.info.keys())[:-1]
        fig, ax = plt.subplots(
            nrows=len(cv_keys), ncols=2,
            figsize=(panel_width * 2, panel_height * len(cv_keys))
        )

        # if cv_key is None:
        sns_data = []
        for cv in cv_keys:
            sns_data_temp = pd.DataFrame(self.info[cv]["history"])
            sns_data_temp["epoch"] = np.arange(0, sns_data_temp.shape[0])
            sns_data_temp["cv"] = cv
            sns_data.append(sns_data_temp)
        sns_data = pd.concat(sns_data, axis=0)
        if 'node_labels_loss' in sns_data.columns:
            print('Note that node_label_loss is scaled by 1e5!!!')
            sns_data['node_labels_loss'] *= 1e5
            sns_data['val_node_labels_loss'] *= 1e5

        sns_data_loss = pd.concat([pd.DataFrame({
            "epoch": sns_data["epoch"].values,
            "cv": sns_data["cv"].values,
            "loss": np.log(sns_data[x].values) if log_loss else sns_data[x].values,
            "partition": x
        }) for i, x in enumerate(plot_metrics)])
        for i, cv in enumerate(cv_keys):
            data = sns_data_loss[sns_data_loss['cv'] == cv]
            sns.lineplot(
                x="epoch", y="loss", hue="partition",
                data=data[~data['partition'].str.contains('val')], ax=ax[i, 0]
            )
            sns.lineplot(
                x="epoch", y="loss", hue="partition",
                data=data[data['partition'].str.contains('val')], ax=ax[i, 1]
            )
            if cv != 'cv0':
                ax[i, 0].get_legend().remove()
                ax[i, 1].get_legend().remove()
            else:
                ax[i, 0].set_title('train loss')
                ax[i, 1].set_title('val_loss')
            ax[i, 0].set_ylim(bottom=0)
            ax[i, 1].set_ylim(bottom=0)

        # Save, show and return figure.
        plt.tight_layout()
        if save is not None:
            plt.savefig(save + suffix)

        if show:
            plt.show()

        plt.close(fig)
        plt.ion()

        if return_axs:
            return ax
        else:
            return None

    def plot_abs_error_vs_number_cells(self):
        import matplotlib.pyplot as plt
        nr_nodes = self.info['cv0']['statistics']['test']['number_nodes']
        nr_nodes_r = [round(a / 300) * 300 for a in nr_nodes]
        preds = self.info['cv0']['predictions']['test']['grade'].argmax(axis=1)
        true = self.true_labels['cv0']['test']['grade'].argmax(axis=1)
        error = preds == true
        p = pd.DataFrame({
            'number nodes': nr_nodes_r,
            'mean accuracy': error
        })
        mp = p.groupby('number nodes').mean()
        mp.plot(figsize=(10, 8))
        plt.scatter(nr_nodes, error, s=4, color='black', label='1 = correct, 0 = wrong')
        plt.legend()

    def plot_abs_error_vs_mean_degree(self):
        import matplotlib.pyplot as plt
        mean_degree = self.info['cv0']['statistics']['test']['mean_node_degree']
        mean_degree_r = [round(a / 5, 0) * 5 for a in mean_degree]
        preds = self.info['cv0']['predictions']['test']['grade'].argmax(axis=1)
        true = self.true_labels['cv0']['test']['grade'].argmax(axis=1)
        error = preds == true
        p = pd.DataFrame({
            'mean degree': mean_degree_r,
            'mean accuracy': error
        })
        mp = p.groupby('mean degree').mean()
        mp.plot(figsize=(10, 8))
        plt.scatter(mean_degree, error, s=4, color='black', label='1 = correct, 0 = wrong')
        plt.legend()

    def plot_abs_error_vs_median_degree(self):
        import matplotlib.pyplot as plt
        median_degree = self.info['statistics']['test']['median_node_degree']
        median_degree_r = [round(a / 5, 0) * 5 for a in median_degree]
        preds = self.info['predictions']['test']['diseasestatus_tumor'].squeeze()
        true = self.true_labels
        abs_error = np.abs(preds - true)
        acc = [1 - round(x) for x in abs_error]
        p = pd.DataFrame({
            'median degree': median_degree_r,
            'mean accuracy': acc
        })
        mp = p.groupby('median degree').mean()
        mp.plot(figsize=(10, 8))
        plt.scatter(median_degree, abs_error, s=4, color='black', label='absolute error')
        plt.legend()

    def plot_roc(
            self,
            partition='test',
            save: Union[str, None] = None,
            suffix: str = "_roc_curve.pdf",
            show: bool = True,
            return_axs: bool = False,
            load_labels=['grade']
    ):
        import matplotlib.pyplot as plt
        from sklearn.metrics import roc_curve
        import sklearn
        fig = plt.figure(figsize=(10, 2))
        for j, cv_id in enumerate(list(self.info.keys())[:-1]):
            ax = fig.add_subplot(1, 6, (j + 1))
            y_hat = np.transpose(self.info[cv_id]['predictions'][partition][load_labels[0]])
            y_obs = np.transpose(self.true_labels[cv_id][partition][load_labels[0]])
            for i in range(len(y_hat)):
                a, b, _ = sklearn.metrics.roc_curve(
                    y_obs[i], y_hat[i]
                )
                ax.plot(a, b, label=f'{[load_labels[0]]} ' + str(i + 1))
                ax.set(adjustable='box', aspect='equal')
                ax.ylabel = 'true positive rate'
                ax.xlabel = 'false positive rate'
            if j == 0:
                plt.legend(bbox_to_anchor=(-0.5, 1.05))
        # Save, show and return figure.
        plt.tight_layout()
        if save is not None:
            plt.tight_layout()
            plt.savefig(save + suffix)

        if show:
            plt.show()

        plt.close(fig)
        plt.ion()

        if return_axs:
            return ax
        else:
            return None

    def roc_auc_score(
            self,
            partition='test',
            load_labels=['grade']
    ):
        import sklearn
        scores = {}
        
        for cv_id in list(self.info.keys())[:-1]:
            y_hat = np.transpose(self.info[cv_id]['predictions'][partition][load_labels[0]])
            y_obs = np.transpose(self.true_labels[cv_id][partition][load_labels[0]])
            auc_roc = np.expand_dims(np.array([
                sklearn.metrics.roc_auc_score(
                    y_true=y_obs[i],
                    y_score=y_hat[i]
                )
                for i in range(len(y_hat))
            ]), axis=-1)
            scores[cv_id] = auc_roc
            # print(f'{cv_id=}')
            # import seaborn as sns
            # import matplotlib.pyplot as plt
            # sns.distplot(y_hat[0][y_obs[0]==1], hist=False)
            # sns.distplot(y_hat[1][y_obs[1]==1], hist=False)
            # plt.plot([0.5,0.5], [0,6])
            # plt.show()
        return scores

    def plot_best_model_roc_auc(
            self,
            partition_show: str,
            partition_select: str,
            metric_select: str,
            cv_mode: Union[str, List[str]] = "mean",
            show_swarm: bool = True,
            save: Union[str, None] = None,
            suffix: str = "_roc_auc_score.pdf",
            show: bool = True,
            return_axs: bool = False,
            panel_width: float = 5,
            panel_height: float = 3,
            load_labels=["grade"],
            plot_all_groups = False,
            ttest: bool =False,
            dispersion: bool = False,
            feature_space: bool =False,
            data_key: str=None,
    ):
        """
        Produces boxplots for all hyperparameters with multiple values. For every value for that hyperparameter
        the best model is chosen and all cvs are plotted.

        :param partition_show: "train" or "eval" or "test" partition of data.
        :param metric_show: Metric to plot.
        :param param_x: Hyper-parameter for x-axis partition.
        :param param_hue: Hyper-parameter for hue-axis partition.
        :param cv_mode: Summary mode of distribution of metric over partitions in cross validation. Out of:

            - "mean"
            - "median"
            - "max"
            - "min":param save: Whether (if not None) and where (path as string given as save) to save plot.
        :param suffix: Suffix of file name to save to.
        :param show: Whether to display plot.
        :param return_axs: Whether to return axis objects.
        :param panel_width:
        :param panel_height:
        :return:
        """
        import seaborn as sns
        import matplotlib.pyplot as plt
        if isinstance(cv_mode, str):
            cv_mode = [cv_mode]

        plt.ioff()
        if plot_all_groups:
            fig, ax = plt.subplots(
                nrows=2, ncols=1,
                figsize=(panel_width, panel_height),
            )
        else:
            fig, ax = plt.subplots(
                nrows=1, ncols=1,
                figsize=(panel_width, panel_height),
            )
        # Plot each metric:
        params_x_unique = np.sort(np.unique(self.summary_table['model_class'].values))
        model_class = []
        auc = []
        grade = []
        # params_x_unique=['GCN']
        for j, cvm in enumerate(cv_mode):
            run_ids = []
            for x in params_x_unique:
                gs_id, run_id, cv_id = self.get_best_model_id(
                    subset_hyperparameters=[('model_class', x)],
                    partition_select=partition_select,
                    metric_select=metric_select,
                    cv_mode=cvm
                )
                if run_id is not None:
                    run_ids.append(run_id)
                    self.get_info(gs_id=gs_id, model_id=run_id, load_labels=load_labels)
                    score = self.roc_auc_score(partition=partition_show, load_labels=load_labels)
                    score = np.squeeze(np.concatenate(list(score.values())))
                    auc.append(score)
                    if data_key == "sch":
                        cvs = len(score) / 2
                        model_class.append([x] * int(cvs) * 2)
                        grade.append(['CLR', 'DII'] * int(cvs))
                    else:
                        cvs = len(score) / 3
                        model_class.append([x] * int(cvs) * 3)
                        grade.append(['grade 1', 'grade 2', 'grade 3'] * int(cvs))
                    
            model_class = np.concatenate(np.array(model_class))
            grade = np.concatenate(np.array(grade))
            auc = np.concatenate(np.array(auc))
            order = ['MLP', 'MI', 'DISP', 'GCN', 'GCNSS', 'GCNSSO', 'GCNspectral', 'GAT']
            order = [o for o in order if o in params_x_unique]
            
            if plot_all_groups:
                bp = sns.boxplot(
                    x=model_class, y=auc, hue=grade, order=order, ax=ax[0]
                )
                handles, labels = bp.get_legend_handles_labels()
                if show_swarm:
                    sns.swarmplot(
                        x=model_class, y=auc, hue=grade, order=order, ax=ax[0]
                    )
                ax[0].set_ylabel('auc roc score', rotation=90)
                ax[0].legend(handles=handles, labels=labels)
                # ax.get_legend().remove()

                ax[0].set_ylim(bottom=0.0, top=1.0)
                ax[0].set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

                bp = sns.boxplot(
                    x=model_class, y=auc, order=order, color='lightgray', ax=ax[1]
                )
                ax[1].set_ylabel('auc roc score', rotation=90)
                # ax.get_legend().remove()

                ax[1].set_ylim(bottom=0.0, top=1.0)
                ax[1].set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

                if ttest:
                        from statannot import add_stat_annotation
                        if feature_space:
                            add_stat_annotation(x=model_class, y=auc,
                                            ax=ax[1],
                                            box_pairs=[
                                                        (("MLP", "types"), ("MLP", "molecular")),
                                                        (("MI", "types"), ("MI", "molecular")),
                                                        (("GCN", "types"), ("GCN", "molecular")),
                                                        (("GCNSS", "types"), ("GCNSS", "molecular")),
                                                        ],
                                            order=order,
                                            test='t-test_ind', text_format='star', loc='inside', verbose=2, comparisons_correction=None)
                        else:
                            if dispersion:
                                    add_stat_annotation(x=model_class, y=auc,
                                                    ax=ax[1],
                                                    box_pairs=[
                                                                ("MLP", "MI"),
                                                                ("MI", "GCN"),
                                                                ("MI", "GCNSS"),
                                                                ("MI", "DISP"),
                                                                #(("MI", "molecular"), ("GCNSS", "molecular")),
                                                                #(("MLP", "molecular"), ("GCNSS", "molecular")),
                                                                #(("GCN", "molecular"), ("GCNSS", "molecular")),
                                                                ],
                                                    order=order,
                                                    test='t-test_ind', text_format='star', loc='inside', verbose=2, comparisons_correction=None)
                            else:
                                add_stat_annotation(x=model_class, y=auc,
                                                    ax=ax[1],
                                                    box_pairs=[
                                                                ("MLP", "MI"),
                                                                ("MI", "GCN"),
                                                                ("MI", "GCNSS"),
                                                                #(("MI", "molecular"), ("GCNSS", "molecular")),
                                                                #(("MLP", "molecular"), ("GCNSS", "molecular")),
                                                                #(("GCN", "molecular"), ("GCNSS", "molecular")),
                                                                ],
                                                    order=order,
                                                    test='t-test_ind', text_format='star', loc='inside', verbose=2, comparisons_correction=None)

            else:
                bp = sns.boxplot(
                    x=model_class, y=auc, hue=grade, order=order
                )
                handles, labels = bp.get_legend_handles_labels()
                if show_swarm:
                    sns.swarmplot(
                        x=model_class, y=auc, hue=grade, order=order
                    )
                ax.set_ylabel('auc roc score', rotation=90)
                ax.legend(handles=handles, labels=labels)
                # ax.get_legend().remove()

                plt.ylim(bottom=0.0, top=1.0)
                plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])



        # Save, show and return figure.
        plt.tight_layout()
        if save is not None:
            plt.tight_layout()
            plt.savefig(save + suffix)

        if show:
            plt.show()

        plt.close(fig)
        plt.ion()

        if return_axs:
            return ax
        else:
            return None

    def compute_merged_acc(
            self,
            partition_show='test'
    ):
        accuracies = []
        for cv_id in list(self.info.keys())[:-1]:
            pred = self.info[cv_id]['predictions'][partition_show]['grade'].argmax(axis=1)
            pred[pred == 0] = 1
            true = self.true_labels[cv_id][partition_show]['grade'].argmax(axis=1)
            true[true == 0] = 1
            acc = np.mean(true == pred)
            accuracies.append(acc)
        mean_acc = np.mean(accuracies)
        return accuracies, mean_acc
