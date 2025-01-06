import pandas as pd
import os
import sys
sys.path.append(os.path.dirname(sys.path[0]))
import time
import random
import string
import numpy as np
import torch
import datetime
import json
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool, global_max_pool
from utils import data_helper as dh
import gnn_models.gnn_model as gm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset_name = 'drs3v6'
gravity_filter = 1.0
fire_filter = 100.0

train_ratio = 0.8
random_seed = 1207
batch_size = 256

# modes: 'strawman2', 'proposed', 'strawman1'
modelkey_mode_dict = {
    # '': 'strawman1',
    # '': 'strawman2',
    # '': 'proposed',
}

##################### backup the script ############################
timestamp_str = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
keystring = timestamp_str + ''.join(random.choices(string.ascii_letters, k=6))
script_name = os.path.basename(__file__)
dh.backup_script(keystring, __file__)

torch.manual_seed(random_seed), np.random.seed(random_seed), random.seed(random_seed)

######################## define model loading function ##############################
def load_gnn_only_model(checkpoint_keystring, midrp_mlp_checkpoint_keystring=None):
    midrp_gnn_runtime_params = dh.load_params_by_keystring(checkpoint_keystring)
    gnn_model = gm.CustomGNN1(**midrp_gnn_runtime_params['gnn_params']).to(device)
    
    
    if midrp_mlp_checkpoint_keystring is not None:
        gnn_model = dh.load_torch_model_by_keystring(midrp_mlp_checkpoint_keystring, gnn_model, 'midrp_gnn_model', device)
    else:
        gnn_model = dh.load_torch_model_by_keystring(checkpoint_keystring, gnn_model, 'gnn_model', device)
    gnn_model.eval()
    return gnn_model

def load_gnn_mlp_seq_model(checkpoint_keystring):
    midrp_mlp_runtime_params = dh.load_params_by_keystring(checkpoint_keystring)
    midrp_mlp_model = gm.MIDRPredictor1(layers=midrp_mlp_runtime_params['midrp_mlp_layers'], dropouts=midrp_mlp_runtime_params['midrp_mlp_dropouts']).to(device)
    midrp_mlp_model = dh.load_torch_model_by_keystring(checkpoint_keystring, midrp_mlp_model, 'midrp_mlp_model', device)

    midrp_gnn_checkpoint_keystring = midrp_mlp_runtime_params['midrp_GNN_checkpoint_keystring']
    if midrp_mlp_runtime_params.get('post_train_together_epoch_num', 0) > 0:
        midrp_gnn_model = load_gnn_only_model(midrp_gnn_checkpoint_keystring, checkpoint_keystring)
    else:
        midrp_gnn_model = load_gnn_only_model(midrp_gnn_checkpoint_keystring)
    midrp_gnn_model.eval(), midrp_mlp_model.eval()
    return midrp_gnn_model, midrp_mlp_model


def load_gnn_mlp_tog_model(checkpoint_keystring):
    midrp_tog_runtime_params = dh.load_params_by_keystring(checkpoint_keystring)
    midrp_mlp_model = gm.MIDRPredictor1(layers=midrp_tog_runtime_params['midrp_mlp_layers'], dropouts=midrp_tog_runtime_params['midrp_mlp_dropouts']).to(device)
    gnn_model = gm.CustomGNN1(**midrp_tog_runtime_params['gnn_params']).to(device)
    midrp_mlp_model = dh.load_torch_model_by_keystring(checkpoint_keystring, midrp_mlp_model, 'midrp_mlp_model', device)
    gnn_model = dh.load_torch_model_by_keystring(checkpoint_keystring, gnn_model, 'gnn_model', device)
    gnn_model.eval(), midrp_mlp_model.eval()
    return gnn_model, midrp_mlp_model

def load_models(checkpoint_keystring, mode):
    if mode == 'strawman2':
        return load_gnn_only_model(checkpoint_keystring)
    elif mode == 'proposed':
        return load_gnn_mlp_seq_model(checkpoint_keystring)
    elif mode == 'strawman1':
        return load_gnn_mlp_tog_model(checkpoint_keystring)
    else:
        raise ValueError(f"Unsupported mode: {mode}")

########################### define the inference functions ############################
def midrp_infer_gnn_only(gnn_model, data, N_forward_layers):
    ## all data should be with the same h_max
    out = gnn_model(data.x, data.edge_index, data.edge_attr, N_forward_layers=N_forward_layers, graph_embedding=False, batch=data.batch)
    out = global_max_pool(out, data.batch)
    return torch.clamp(out, min=0)
def midrp_infer_gnn_mlp(gnn_model, mlp_model, data, N_forward_layers):
    with torch.no_grad():
        graph_embedding = gnn_model(data.x, data.edge_index, data.edge_attr, N_forward_layers=N_forward_layers, graph_embedding=True, batch=data.batch, embedding_position=0)
    pred = mlp_model(graph_embedding)
    return torch.clamp(pred, min=0)

def midrp_infer(models, data, N_forward_layers, mode):
    if mode == 'strawman2':
        return midrp_infer_gnn_only(models, data, N_forward_layers)
    elif mode == 'proposed' or mode == 'strawman1':
        return midrp_infer_gnn_mlp(models[0], models[1], data, N_forward_layers)
    else:
        raise ValueError(f"Unsupported mode: {mode}")

########################### load model and then dataset ############################

keystring_model_dict = {
    model_keystring: load_models(model_keystring, mode, checkpoint_epoch=modelkey_checkpoint_epoch_dict.get(model_keystring, None)) for model_keystring, mode in modelkey_mode_dict.items()
}

########################### load the dataset ############################
train_dataset_list, test_dataset_list = dh.load_gf_train_test_normalized_labeled_dataset(dataset_name, gravity_filter, fire_filter, device)
print(f"Train dataset size: {len(train_dataset_list)}, Test dataset size: {len(test_dataset_list)}")

train_loaders = dh.create_loaders(train_dataset_list, batch_size, shuffle=False)
test_loaders = dh.create_loaders(test_dataset_list, batch_size, shuffle=False)
train_batch_num, test_batch_num = sum([len(loader) for loader in train_loaders.values()]), sum([len(loader) for loader in test_loaders.values()])
train_h_max_counts, test_h_max_counts= dh.count_from_loaders(train_loaders), dh.count_from_loaders(test_loaders)
print('Training:',json.dumps(train_h_max_counts, indent=4))
print('Testing:',json.dumps(test_h_max_counts, indent=4))

############################## evaluate the model ############################
###############  first about mse and mae ############################
mse_loss_func = torch.nn.MSELoss()
mae_loss_func = torch.nn.L1Loss()

modelkey_train_mse_dict, modelkey_test_mse_dict = {}, {}
modelkey_train_mae_dict, modelkey_test_mae_dict = {}, {}

##first, on the training dataest
for model_keystring, model in keystring_model_dict.items():
    mode = modelkey_mode_dict[model_keystring]
    print(f"evaluating model: {model_keystring}, mode: {mode}")
    train_mse, train_mae = 0, 0 
    for h_max, loader in train_loaders.items():
        for data in loader:
            pred = midrp_infer(model, data, h_max, mode).squeeze()
            mse_loss = mse_loss_func(pred, data.y_max)
            mae_loss = mae_loss_func(pred, data.y_max)
            train_mse += mse_loss.item()
            train_mae += mae_loss.item()
    modelkey_train_mse_dict[model_keystring] = train_mse / train_batch_num
    modelkey_train_mae_dict[model_keystring] = train_mae / train_batch_num

    ## second, on the testing dataset
    test_mse, test_mae = 0, 0
    for h_max, loader in test_loaders.items():
        for data in loader:
            pred = midrp_infer(model, data, h_max, mode).squeeze()
            mse_loss = mse_loss_func(pred, data.y_max)
            mae_loss = mae_loss_func(pred, data.y_max)
            test_mse += mse_loss.item()
            test_mae += mae_loss.item()
    modelkey_test_mse_dict[model_keystring] = test_mse / test_batch_num
    modelkey_test_mae_dict[model_keystring] = test_mae / test_batch_num

#################### print result and save result ############################
print("Training MSE:")
print(json.dumps(modelkey_train_mse_dict, indent=4))
print("Testing MSE:")
print(json.dumps(modelkey_test_mse_dict, indent=4))


####################### then about the spearmans rank correlation ############################


for data in train_dataset_list:
    for key, model in keystring_model_dict.items():
        prediction = midrp_infer(model, data, N_forward_layers=data.h_max, mode=modelkey_mode_dict[key]).squeeze().item()
        setattr(data, f'y_pred_{key}', prediction)
for data in test_dataset_list:
    for key, model in keystring_model_dict.items():
        prediction = midrp_infer(model, data, N_forward_layers=data.h_max, mode=modelkey_mode_dict[key]).squeeze().item()
        setattr(data, f'y_pred_{key}', prediction)


################# calculate the rank of each data within each model_number ############################
##### first group the data by model_number ########
train_model_groups = defaultdict(list)
for data in train_dataset_list:
    train_model_groups[data.model_number].append(data)
test_model_groups = defaultdict(list)
for data in test_dataset_list:
    test_model_groups[data.model_number].append(data)
#### then define the rank function for each grouped dataset #######
def sort_and_rank(model_groups):
    for model_number, data_group in model_groups.items():
        pred_keys = [f'y_pred_{key}' for key in modelkey_mode_dict.keys()]
        for pred_key in pred_keys:
            sorted_group = sorted(data_group, key=lambda d: getattr(d, pred_key), reverse=True)
            for rank, data in enumerate(sorted_group, start=1):
                setattr(data, f'pred_fire_rank_{pred_key}', rank)

sort_and_rank(train_model_groups)
sort_and_rank(test_model_groups)

################# calculate the spearman correlation between the ranks of each model_number ############################

h_max_list = list(set([data.h_max.item() for data in train_dataset_list]))
train_h_max_r_dict = {key: {h_max: [] for h_max in h_max_list} for key in modelkey_mode_dict.keys()}
test_h_max_r_dict = {key: {h_max: [] for h_max in h_max_list} for key in modelkey_mode_dict.keys()}

train_spearman_correlation = {key: {} for key in modelkey_mode_dict.keys()}
test_spearman_correlation = {key: {} for key in modelkey_mode_dict.keys()}
for model_number, data_group in train_model_groups.items():
    gt_fire_ranks = [data.drift_ratio_ranking.item() for data in data_group]
    for model_keystring in modelkey_mode_dict.keys():
        pred_key = f'y_pred_{model_keystring}'

        fire_ranks = [getattr(data, f'pred_fire_rank_{pred_key}') for data in data_group]
        spearman_corr = spearmanr(gt_fire_ranks, fire_ranks).correlation
        train_spearman_correlation[model_keystring][model_number] = spearman_corr
        
        h_max = data_group[0].h_max.item()
        train_h_max_r_dict[model_keystring][h_max].append(spearman_corr)

for model_number, data_group in test_model_groups.items():
    gt_fire_ranks = [data.drift_ratio_ranking.item() for data in data_group]
    for model_keystring in modelkey_mode_dict.keys():
        pred_key = f'y_pred_{model_keystring}'

        fire_ranks = [getattr(data, f'pred_fire_rank_{pred_key}') for data in data_group]
        spearman_corr = spearmanr(gt_fire_ranks, fire_ranks).correlation
        test_spearman_correlation[model_keystring][model_number] = spearman_corr

        h_max = data_group[0].h_max.item()
        test_h_max_r_dict[model_keystring][h_max].append(spearman_corr)


################# calculate the average spearman correlation ############################
train_avg_spearman_correlation = {key: np.mean(list(val.values())) for key, val in train_spearman_correlation.items()}
test_avg_spearman_correlation = {key: np.mean(list(val.values())) for key, val in test_spearman_correlation.items()}   
print('Train avg r_s:', train_avg_spearman_correlation)
print('Test avg r_s:', test_avg_spearman_correlation)

train_avg_spearman_correlation_h_max = {key: {h_max: np.mean(val) for h_max, val in h_max_dict.items()} for key, h_max_dict in train_h_max_r_dict.items()}
test_avg_spearman_correlation_h_max = {key: {h_max: np.mean(val) for h_max, val in h_max_dict.items()} for key, h_max_dict in test_h_max_r_dict.items()}
print('Train avg r_s by level:', json.dumps(train_avg_spearman_correlation_h_max, indent=4))
print('Test avg r_s by level:', json.dumps(test_avg_spearman_correlation, indent=4))


################# save the results ############################
param_name_list = [
    'gravity_filter',
    'fire_filter',
    'dataset_name',
    'train_ratio',
    'random_seed',
    'modelkey_mode_dict',
    'timestamp_str',
    'keystring',
    'script_name',
    'modelkey_train_mse_dict',
    'modelkey_test_mse_dict',
    'modelkey_train_mae_dict',
    'modelkey_test_mae_dict',
    'train_avg_spearman_correlation',
    'test_avg_spearman_correlation',
    'train_avg_spearman_correlation_h_max',
    'test_avg_spearman_correlation_h_max',
]

dh.save_runtime_params(keystring, locals(), param_name_list)

