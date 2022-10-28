import pickle
import sys

import numpy as np
import tensorflow as tf

import tissue.api as tissue

print(tf.__version__)

# manual inputs
data_set = sys.argv[1].lower()
optimizer = sys.argv[2].lower()
lr_keys = sys.argv[3]
l2_keys = sys.argv[4]
depth_keys = sys.argv[5]
width_keys = sys.argv[6]
loss_weight_others_keys = sys.argv[7]
batch_size_key = sys.argv[8]
target_label = sys.argv[9].lower()
feature_space_id = sys.argv[10].lower()
multitask_setting = sys.argv[11].lower()    # either 'target', 'small' or 'large'
gs_id = sys.argv[12].lower()
data_path_base = sys.argv[13]
out_path = sys.argv[14]

# data
if data_set == 'basel_zurich':
    data_path = data_path_base + '/zenodo/'
    buffered_data_path = data_path + 'refactored/buffer/'
    label_selection_target = ['grade']
    label_selection_survival = ['DFSmonth']
    labels_1 = [
        'tumor_size',
        'ERStatus',
        'PRStatus',
        'HER2Status'
    ]
    labels_2 = [
        'Lymphaticinvasion',
        'microinvasion'
    ]
    covar_selection = [
        'age',
        'Count_Cells',
        'location'
    ]
    cell_type_coarseness = 'fine'
    radius = 10
elif data_set == 'metabric':
    data_path = data_path_base + '/metabric/'
    buffered_data_path = data_path + 'refactored/buffer/'
    label_selection_target = ['grade']
    label_selection_survival = ['time_last_seen']
    labels_1 = [
        'tumor_size',
        'ERstatus'
    ]
    labels_2 = [
        'hist_type',
        'stage',
        'lymph_pos'
    ]
    covar_selection = [
        'age',
        'menopausal'
    ]
    cell_type_coarseness = 'fine'
    radius = 20
elif data_set == "schuerch":
    data_path = data_path_base + '/schuerch/'
    buffered_data_path = data_path + 'refactored/buffer/'
    label_selection_target = ['Group']
    label_selection_survival = ['DFS']
    labels_1 = [
        'DFS',
        'Group',
        'Diffuse',
        'Klintrup_Makinen',  # slightly finer than Group
        'Sex',
        'Age',
    ]
    labels_2 = []
    covar_selection = [
        'Sex',
        'Age'
    ]
    cell_type_coarseness = 'binary'
    radius = 25
else:
    raise ValueError('data_origin not recognized')

if target_label == "survival":
    label_selection = label_selection_survival
else:
    label_selection = label_selection_target
target_label = label_selection[0]
drop_when_missing = [target_label]

labels_1 = [label for label in labels_1 if label != target_label]
labels_2 = [label for label in labels_2 if label != target_label]

if multitask_setting == 'target':
    pass
elif multitask_setting == 'small':
    label_selection += labels_1
    if target_label == 'survival':
        label_selection += label_selection_target
    else:
        label_selection += label_selection_survival
elif multitask_setting == 'large':
    label_selection += labels_1
    label_selection += labels_2
    label_selection += label_selection_target
    label_selection += label_selection_survival
else:
    raise ValueError("multitask setting %s not recognized" % multitask_setting)
label_selection = list(np.unique(label_selection))

if 'schuerch' in data_set:
    monitor_partition = "train"
    monitor_metric = "loss"
    early_stopping = False
    reduce_lr_on_plateau = False
    epochs = 200
    validation_split = 0.1
    test_split = 0.
else:
    monitor_partition = "val"
    monitor_metric = "loss"
    early_stopping = True
    reduce_lr_on_plateau = True
    epochs = 500
    validation_split = 0.1
    test_split = 0.1


# model and training
ncv = 3
activation = "relu"

lr_dict = {
    "1": 0.05,
    "2": 0.005,
    "3": 0.0005,
    "4": 0.00005
}
l2_dict = {
    "1": 0.,
    "2": 1e-6,
    "3": 1e-3,
    "4": 1e0
}
depth_dict = {
    "1": 1,
    "2": 2,
    "3": 3,
    "4": 5,
    "5": 10
}
width_dict = {
    "1": 16,
    "2": 32,
    "3": 64
}
lw_dict = {
    "1": 0.,
    "2": 1e-2,
    "3": 1e-1,
    "4": 1.
}
bs_dict = {
    "1": 16,
    "2": 32,
    "3": 64,
    "4": 256,
}

# Grid serach sub grid here so that more grid search points can be handled in a single job:
for lr_key in lr_keys.split("+"):
    for l2_key in l2_keys.split("+"):
        for depth_key in depth_keys.split("+"):
            for width_key in width_keys.split("_+"):
                for loss_weight_others_key in loss_weight_others_keys.split("+"):
                    # Set ID of output
                    model_id = gs_id + "_regression_" + \
                        data_set + "_" + target_label + "_" + \
                        optimizer + "_lr" + str(lr_key) + \
                        "_l2" + str(l2_key) + \
                        "_de" + str(depth_key) + "_wi" + str(width_key) + \
                        "_lw" + str(loss_weight_others_key) + \
                        "_bs" + str(batch_size_key)+ \
                        "_fs" + str(feature_space_id) + "_mt" + str(multitask_setting)

                    run_params = {
                        'model_id': model_id,
                        'model_class': 'reg',
                        'gs_id': gs_id,

                        'data_set': data_set,
                        'optimizer': optimizer,
                        'learning_rate': lr_dict[lr_key],
                        'l2_reg': l2_dict[l2_key],
                        'depth': depth_dict[depth_key],
                        'width': width_dict[width_key],
                        'loss_weight_others': lw_dict[loss_weight_others_key],
                        'batch_size': bs_dict[batch_size_key],
                        'target_label': target_label,
                        'graph_label_selection': label_selection,
                        'node_feature_space_id': feature_space_id,
                        'multitask_setting': multitask_setting,
                    }

                    fn_out = out_path + "/results/" + model_id
                    with open(fn_out + '_runparams.pickle', 'wb') as f:
                        pickle.dump(obj=run_params, file=f)

                    for i in range(ncv):
                        print("cv %i" % i)
                        model_id_cv = model_id + "_cv" + str(i)
                        fn_out = out_path + "/results/" + model_id_cv

                        trainer = tissue.train.TrainModelREG()
                        trainer.init_estim()
                        trainer.estimator.get_data(
                            data_origin=data_set,
                            data_path=data_path,
                            buffered_data_path=buffered_data_path,
                            write_buffer=False,
                            radius=radius, # doesn't matter just for loading from buffer
                            target_label=target_label,
                            graph_label_selection=label_selection,
                            graph_covar_selection=None,
                            node_label_space_id=feature_space_id,
                            node_feature_transformation='none',
                            drop_when_missing=drop_when_missing,
                            cell_type_coarseness=cell_type_coarseness,
                            test_split=.1,
                            validation_split=validation_split,
                            seed=7 # same seed for all models
                        )
                        trainer.estimator.init_model(
                            optimizer=optimizer,
                            learning_rate=lr_dict[lr_key],
                            depth=depth_dict[depth_key],
                            activation=activation,
                            l2_reg=l2_dict[l2_key],
                            loss_weight_others=lw_dict[loss_weight_others_key],
                            width=width_dict[width_key]
                        )
                        trainer.estimator.train(
                            epochs=epochs,
                            max_steps_per_epoch=20,
                            batch_size=bs_dict[batch_size_key],
                            validation_batch_size=16,
                            max_validation_steps=10,
                            patience=20,
                            lr_schedule_min_lr=1e-5,
                            lr_schedule_factor=0.2,
                            lr_schedule_patience=10,
                            monitor_partition=monitor_partition,
                            monitor_metric=monitor_metric,
                            early_stopping=early_stopping,
                            reduce_lr_on_plateau=reduce_lr_on_plateau,
                            shuffle_buffer_size=int(100),
                        )
                        trainer.save(fn=fn_out, save_weights=True)
