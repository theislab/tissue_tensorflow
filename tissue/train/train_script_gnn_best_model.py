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
dropout_rate_key = sys.argv[4]
l2_keys = sys.argv[5]
depth_keys = sys.argv[6]
width_keys = sys.argv[7]
loss_weight_others_keys = sys.argv[8]    # weight factor for all losses but target
loss_weight_type_key = sys.argv[9]

batch_size_key = sys.argv[11]

max_dist_key = sys.argv[13]             # max euclidean distance for an edge


transform_key = sys.argv[16]
covar_key = sys.argv[17]
target_label = sys.argv[18].lower()   # 'survival' or other
node_label_space_id = sys.argv[19].lower()
feature_embedding_key = sys.argv[20]
multitask_setting = sys.argv[21].lower()    # either 'target', 'small' or 'large'
n_clusters_key = sys.argv[22]
entropy_weight_key = sys.argv[23]
number_heads_key = sys.argv[24]
self_supervision_mode = sys.argv[25].lower()  # 'none', 'multitask'
model_class = sys.argv[26].lower()     # 'gcn' or 'gat' or 'gatcomplex' or 'gatsingle' or 'mi'
aggregation = sys.argv[27].lower()         # 'none' or 'diffpool'
final_pooling = sys.argv[28].lower()    # 'mean', 'max', 'attention'

gs_id = sys.argv[31].lower()
data_path_base = sys.argv[32]
out_path = sys.argv[33]

if self_supervision_mode == 'multitask':
    self_supervision_label = ['relative_cell_types']
    self_supervision = True
else:
    self_supervision_label = []
    self_supervision = False

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
    md_dict = {
        "1": 10,
        "2": 20,
        "3": 50
    }
    cell_type_coarseness = 'fine'
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
    md_dict = {
        "1": 10,
        "2": 20,
        "3": 55
    }
    cell_type_coarseness = 'fine'
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
    md_dict = {     # avg node degree:
        "1": 25,    # 2.6
        "2": 50,    # 8.2
        "3": 120    # 40.3
    }
    cell_type_coarseness = 'binary'
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

if model_class == "gsum" or 'schuerch' in data_set:
    monitor_partition = "train"
    monitor_metric = "loss"
    early_stopping = True
    reduce_lr_on_plateau = True
else:
    monitor_partition = "val"
    monitor_metric = "loss"
    early_stopping = True
    reduce_lr_on_plateau = True
if monitor_partition == "val":
    validation_split = 0.1
else:
    validation_split = 0.
if early_stopping:
    epochs = 2000
else:
    epochs = 200
epochs = epochs if "test" not in gs_id else 10  # short run if GS is labeled test

# model and training
ncv = 3
step_len_gat_loop = 1000
batched_gat = True
attention_dim = 4
aggregation_depth = 1
depth_final_dense = 1
activation = 'leakyrelu_0.1'

transformation_dict = {
    "1": "standardize_globally",
    "2": "standardize_per_image",
    "3": "rank_per_image",
    "4": "none"
}
covar_dict = {
    "1": None,
    "2": covar_selection
}
lr_dict = {
    "1": 0.05,
    "2": 0.005,
    "3": 0.0005,
    "4": 0.00005
}
depth_dict = {
    "1": 1,
    "2": 2,
    "3": 3,
    "4": 5,
    "5": 10,
    "6": 15,
}
width_dict = {
    "0": 4,
    "1": 8,
    "2": 16,
    "3": 32,
    "4": 64
}
dr_dict = {
    "1": 0.,
    "2": 0.2,
    "3": 0.5
}
dr_pool_dict = {
    "1": 0.,
    "2": 0.2,
    "3": 0.5
}
l2_dict = {
    "1": 0.,
    "2": 1e-6,
    "3": 1e-3,
    "4": 1e0
}
feature_embedding_dict = {
    "0": 0,
    "1": 1,
    "2": 2,
    "3": 3
}
lw_dict = {
    "1": 0.,
    "2": 1e-2,
    "3": 1e-1,
    "4": 1.,
    "5": 1e1,
    "6": 1e2,
    "7": 1e3,
}
lw_type_dict = {    # grade loss is about 1, raw type about 2e-6
    "1": 0.,
    "2": 1e1,
    "3": 1e3,
    "4": 1e5,
}
bs_dict = {
    "0": 8,
    "1": 16,
    "2": 32,
    "3": 64,
    "4": 62, # for Schuerch
}
kn_dict = {
    "1": 1,
    "2": 2,
    "3": 5,
    "4": 10
}
nc_dict = {
    "1": 5,
    "2": 10,
    "3": 20
}
ew_dict = {
    "1": 1e-4,
    "2": 1e-2,
    "3": 1.,
    "4": 1e2
}
nh_dict = {
    "1": 1,
    "2": 2,
    "3": 4
}

if lw_type_dict[loss_weight_type_key] > 0:
    node_labels = True
else:
    node_labels = False
if node_label_space_id == 'type':
    dr_key = "1"
    dr_pooling_key = dropout_rate_key
    patience = 80
    lr_schedule_patience = 40
else:
    dr_key = "1"
    dr_pooling_key = dropout_rate_key
    patience = 70
    lr_schedule_patience = 20
max_steps_per_epoch = 20
max_validation_steps = 10
if model_class == "gatsingle":
    batch_size = 1
    validation_batch_size = 1
    max_steps_per_epoch *= bs_dict[batch_size_key]
    max_validation_steps *= 16
else:
    batch_size = bs_dict[batch_size_key]
    validation_batch_size = 16

# Grid serach sub grid here so that more grid search points can be handled in a single job:
for lr_key in lr_keys.split("+"):
    for l2_key in l2_keys.split("+"):
        for depth_key in depth_keys.split("+"):
            for width_key in width_keys.split("+"):
                for loss_weight_others_key in loss_weight_others_keys.split("+"):
                    if width_dict[width_key] % nh_dict[number_heads_key] != 0:
                        raise ValueError("Width has to be a multiple of number heads!")
                    # Set ID of output
                    model_id = model_class + "_" + aggregation + "_" +\
                        data_set + "_" + target_label + \
                        optimizer + "_lr" + str(lr_key) + \
                        "_dr" + str(dr_key) + "_drp" + str(dr_pooling_key) + \
                        "_l2" + str(l2_key) + \
                        "_de" + str(depth_key) + "_wi" + str(width_key) + \
                        "_lw" + str(loss_weight_others_key) + "_lt" + str(loss_weight_type_key) + \
                        "_bs" + str(batch_size_key) + \
                        "_md" + str(max_dist_key) + \
                        "_tk" + str(transform_key) + "_ck" + str(covar_key) + \
                        "_fs" + str(node_label_space_id) + "_fe" + str(feature_embedding_key) + \
                        "_fp" + str(final_pooling) + "_mt" + str(multitask_setting) + \
                        "_nc" + str(n_clusters_key) + "_ew" + str(entropy_weight_key) + \
                        '_nh' + str(number_heads_key) + '_ss' + str(self_supervision_mode)

                    run_params = {
                        'model_id': model_id,
                        'model_class': model_class,
                        'gs_id': gs_id,

                        'data_set': data_set,
                        'max_dist': md_dict[max_dist_key],
                        'target_label': target_label,
                        'graph_label_selection': label_selection,
                        'graph_covar_selection': covar_dict[covar_key],
                        'node_feature_space_id': node_label_space_id,
                        'node_feature_transformation': transformation_dict[transform_key],

                        'optimizer': optimizer,
                        'learning_rate': lr_dict[lr_key],
                        'depth_feature_embedding': feature_embedding_dict[feature_embedding_key],
                        'depth': depth_dict[depth_key],
                        'activation': activation,
                        'dropout_rate': dr_dict[dr_key],
                        'dropout_rate_pooling': dr_pool_dict[dr_pooling_key],
                        'l2_reg': l2_dict[l2_key],
                        'loss_weight_others': lw_dict[loss_weight_others_key],
                        'loss_weight_type': lw_type_dict[loss_weight_type_key],
                        'batch_size': bs_dict[batch_size_key],
                        'aggregation': aggregation,
                        'aggregation_depth': aggregation_depth,
                        'final_pooling': final_pooling,
                        'depth_final_dense': depth_final_dense,
                        'multitask_setting': multitask_setting,
                        'n_clusters': nc_dict[n_clusters_key],
                        'entropy_weight': ew_dict[entropy_weight_key],
                        'number_heads': nh_dict[number_heads_key],
                        'self_supervision_mode': self_supervision_mode,
                    }
                    if 'gcn' in model_class:
                        kwargs_init = {"width": width_dict[width_key]}
                    elif model_class == "gsum":
                        kwargs_init = {}
                    elif model_class == "mi":
                        kwargs_init = {"width": width_dict[width_key]}
                    elif model_class == "gat":
                        kwargs_init = {
                            "complex": False,
                            "batched_gat": batched_gat,
                            "step_len_gat_loop": step_len_gat_loop,
                            "number_heads": nh_dict[number_heads_key],
                            "attention_dim": attention_dim,
                            "width": width_dict[width_key] // nh_dict[number_heads_key],
                        }
                    elif model_class == "gatcomplex":
                        kwargs_init = {
                            "complex": True,
                            "batched_gat": batched_gat,
                            "step_len_gat_loop": step_len_gat_loop,
                            "number_heads": nh_dict[number_heads_key],
                            "attention_dim": attention_dim,
                            "width": width_dict[width_key] // nh_dict[number_heads_key],
                        }
                    elif model_class == "gatsingle":
                        kwargs_init = {
                            "complex": True,
                            "batched_gat": batched_gat,
                            "step_len_gat_loop": step_len_gat_loop,
                            "number_heads": nh_dict[number_heads_key],
                            "attention_dim": attention_dim,
                            "width": width_dict[width_key] // nh_dict[number_heads_key],
                            "single_batched": True,
                            "update_freq": bs_dict[batch_size_key],
                        }
                    else:
                        raise ValueError("model class %s not recognized" % model_class)
                    run_params.update(kwargs_init)

                    fn_out = out_path + "/results/" + model_id
                    with open(fn_out + '_runparams.pickle', 'wb') as f:
                        pickle.dump(obj=run_params, file=f)

                    for i in range(ncv):
                        print("cv %i" % i)
                        model_id_cv = model_id + "_cv" + str(i)
                        fn_out = out_path + "/results/" + model_id_cv

                        if 'gcn' in model_class:
                            trainer = tissue.train.TrainModelGCN()
                        elif model_class == "gsum":
                            trainer = tissue.train.TrainModelGsum()
                        elif model_class == "mi":
                            trainer = tissue.train.TrainModelMI()
                        elif 'gat' in model_class:
                            trainer = tissue.train.TrainModelGAT()
                        else:
                            raise ValueError("model class %s not recognized" % model_class)
                        trainer.init_estim()
                        trainer.estimator.get_data(
                            data_origin=data_set,
                            data_path=data_path,
                            buffered_data_path=buffered_data_path,
                            write_buffer=False,
                            radius=md_dict[max_dist_key],
                            target_label=target_label,
                            node_supervision=node_labels,
                            cell_type_coarseness=cell_type_coarseness,
                            graph_label_selection=label_selection,
                            graph_covar_selection=covar_dict[covar_key],
                            node_label_space_id=node_label_space_id,
                            node_feature_transformation=transformation_dict[transform_key],
                            adj_type=None,  # will be selected automatically
                            drop_when_missing=drop_when_missing,
                            n_cluster=nc_dict[n_clusters_key],
                            self_supervision=self_supervision,
                            self_supervision_label=self_supervision_label,
                            test_split=.1,
                            validation_split=validation_split,
                            seed=7 # same seed for all models
                        )
                        trainer.estimator.init_model(
                            optimizer=optimizer,
                            learning_rate=lr_dict[lr_key],
                            depth_feature_embedding=feature_embedding_dict[feature_embedding_key],
                            depth=depth_dict[depth_key],
                            activation=activation,
                            dropout_rate=dr_dict[dr_key],
                            dropout_rate_pooling=dr_pool_dict[dr_pooling_key],
                            l2_reg=l2_dict[l2_key],
                            add_covar_at_nodes=True,
                            add_covar_at_latent=False,
                            loss_weight_others=lw_dict[loss_weight_others_key],
                            loss_weight_type=lw_type_dict[loss_weight_type_key],
                            loss_weight_self_supervision=lw_dict[loss_weight_others_key],
                            aggregation=aggregation,
                            aggregation_n_clusters=nc_dict[n_clusters_key],
                            aggregation_depth=aggregation_depth,
                            entropy_weight=ew_dict[entropy_weight_key],
                            final_pooling=final_pooling,
                            depth_final_dense=depth_final_dense,
                            **kwargs_init
                        )
                        trainer.estimator.train(
                            epochs=epochs,
                            max_steps_per_epoch=max_steps_per_epoch,
                            batch_size=batch_size,
                            validation_batch_size=validation_batch_size,
                            max_validation_steps=max_validation_steps,
                            patience=patience,
                            lr_schedule_min_lr=1e-5,
                            lr_schedule_factor=0.2,
                            lr_schedule_patience=lr_schedule_patience,
                            monitor_partition=monitor_partition,
                            monitor_metric=monitor_metric,
                            early_stopping=early_stopping,
                            reduce_lr_on_plateau=reduce_lr_on_plateau,
                            shuffle_buffer_size=int(100),
                        )
                        trainer.save(fn=fn_out, save_weights=True)