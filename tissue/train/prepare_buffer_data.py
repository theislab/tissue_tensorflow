import sys
import time
import tensorflow as tf
import tissue.api as tissue
import pickle
print(tf.__version__)

start = time.time()

# manual inputs
data_set = sys.argv[1].lower()
max_dist_key = sys.argv[2]           
data_path_base = sys.argv[3]
cell_type_coarseness = sys.argv[4]

# data
if data_set == 'basel_zurich':
    data_path = data_path_base + '/zenodo/'
    data_buffer_path = data_path + 'refactored/buffer/'
    label_selection = [
        'location',
        'grade',
        'tumor_size',
        'diseasestatus',
        'tumor_type',
        'age',
        'Patientstatus',
        'OSmonth',
        'DFSmonth',
        'clinical_type',
        'Subtype',
        'PTNM_M',
        'PTNM_T',
        'PTNM_N',
        'PTNM_Radicality',
        'Lymphaticinvasion',
        'Venousinvasion',
        'ERStatus',
        'PRStatus',
        'HER2Status',
        'TripleNegDuctal',
        'microinvasion',
        'I_plus_neg',
        'SN',
        'Pre-surgeryTx',
        'Post-surgeryTx',
        'TMABlocklabel',
        'Yearofsamplecollection',
        '%tumorcells',
        '%normalepithelialcells',
        '%stroma',
        '%inflammatorycells',
        'Count_Cells'
    ]
    target_label = "grade"
    md_dict = {
        "1": 10,
        "2": 20,
        "3": 50
    }
    cell_type_coarseness=cell_type_coarseness
elif data_set == 'metabric':
    data_path = data_path_base + '/metabric/'
    data_buffer_path = data_path + 'refactored/buffer/'
    label_selection = [
        'grade',
        'tumor_size',
        'hist_type',
        'stage',
        'age',
        'menopausal',
        'time_last_seen',
        'ERstatus',
        'lymph_pos',
        'CT',
        'HT',
        'RT',
        'surgery',
        'NPI'
    ]
    target_label = "grade"
    md_dict = {
        "1": 10,
        "2": 20,
        "3": 55
    }
    cell_type_coarseness=cell_type_coarseness
elif data_set == 'schuerch':
    data_path = data_path_base + '/schuerch/'
    data_buffer_path = data_path + 'refactored/buffer/'
    label_selection = [
        'DFS',
        'Group',
        'LA',
        'Diffuse',
        'Klintrup_Makinen',
        'CLR_Graham_Appelman',
        'Sex',
        'Age',
    ]
    target_label = "DFS"
    md_dict = {
        "1": 25,
        "2": 50,
        "3": 120
    }
    cell_type_coarseness=cell_type_coarseness
else:
    raise ValueError('data_origin not recognized')

rs_dict = {
    "1": 1,
    "2": 2,
    "3": 3
}

trainer = tissue.train.TrainModelGCN()

# estim.get_data(
#     data_origin=data_set,
#     data_path=data_path,
#     buffered_data_path=data_buffer_path,
#     write_buffer=True,
#     max_dist=md_dict[max_dist_key],
#     steps=rs_dict[radius_steps_key],
#     graph_label_selection=select_labels,
#     target_label=target_label,
#     node_feature_transformation='none',
#     diseased_only=True,
#     diseased_as_paired=False
# )

trainer.init_estim()
trainer.estimator.get_data(
    data_origin=data_set,
    data_path=data_path,
    buffered_data_path=data_buffer_path,
    write_buffer=True,
    radius=md_dict[max_dist_key],
    cell_type_coarseness=cell_type_coarseness,
    target_label=target_label,
    graph_label_selection=label_selection,
    node_feature_transformation='none',
)
