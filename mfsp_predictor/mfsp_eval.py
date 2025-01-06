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
from collections import defaultdict
from utils import data_helper as dh
import matplotlib.pyplot as plt
from torch_geometric.nn import global_mean_pool, global_max_pool
import gnn_models.gnn_model as gm
import gc

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset_name = 'drs4v2'
gravity_filter_threshold = 1.0
train_ratio = 0.8
random_seed = 1207

# mfspp_checkpoint_keystring = '20241227211301SfGJQB'


mfspp_runtime_params = dh.load_params_by_keystring(mfspp_checkpoint_keystring)
midrp_checkpoint_keystring = mfspp_runtime_params['midrp_checkpoint_keystring']
midrp_runtime_params = dh.load_params_by_keystring(midrp_checkpoint_keystring)

mfspp_mode = mfspp_runtime_params['mfspp_mode']
midrp_mode = mfspp_runtime_params['midrp_mode']
virtual_fire_01coord = [0.5, 0.5, 0.5]

##################### backup the script ############################
timestamp_str = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
keystring = timestamp_str + ''.join(random.choices(string.ascii_letters, k=6))
script_name = os.path.basename(__file__)
dh.backup_script(keystring, __file__)

torch.manual_seed(random_seed), np.random.seed(random_seed), random.seed(random_seed)

######################## define model loading function ##############################
def load_midrp_gnn_only_model(checkpoint_keystring, midrp_mlp_checkpoint_keystring=None):
    midrp_gnn_runtime_params = dh.load_params_by_keystring(checkpoint_keystring)
    gnn_model = gm.CustomGNN1(**midrp_gnn_runtime_params['gnn_params']).to(device)
    if midrp_mlp_checkpoint_keystring is not None:
        gnn_model = dh.load_torch_model_by_keystring(midrp_mlp_checkpoint_keystring, gnn_model, 'midrp_gnn_model', device)
    else:
        gnn_model = dh.load_torch_model_by_keystring(checkpoint_keystring, gnn_model, 'gnn_model', device)
    gnn_model.eval()
    return gnn_model

def load_midrp_gnn_mlp_seq_model(checkpoint_keystring):
    midrp_mlp_runtime_params = dh.load_params_by_keystring(checkpoint_keystring)
    midrp_mlp_model = gm.MIDRPredictor1(layers=midrp_mlp_runtime_params['midrp_mlp_layers'], dropouts=midrp_mlp_runtime_params['midrp_mlp_dropouts']).to(device)
    midrp_mlp_model = dh.load_torch_model_by_keystring(checkpoint_keystring, midrp_mlp_model, 'midrp_mlp_model', device)

    midrp_gnn_checkpoint_keystring = midrp_mlp_runtime_params['midrp_GNN_checkpoint_keystring']
    if midrp_mlp_runtime_params.get('post_train_together_epoch_num', 0) > 0:
        midrp_gnn_model = load_midrp_gnn_only_model(midrp_gnn_checkpoint_keystring, checkpoint_keystring)
    else:
        midrp_gnn_model = load_midrp_gnn_only_model(midrp_gnn_checkpoint_keystring)
    midrp_gnn_model.eval(), midrp_mlp_model.eval()
    return midrp_gnn_model, midrp_mlp_model

def load_midrp_gnn_mlp_tog_model(checkpoint_keystring):
    midrp_tog_runtime_params = dh.load_params_by_keystring(checkpoint_keystring)
    midrp_mlp_model = gm.MIDRPredictor1(layers=midrp_tog_runtime_params['midrp_mlp_layers'], dropouts=midrp_tog_runtime_params['midrp_mlp_dropouts']).to(device)
    gnn_model = gm.CustomGNN1(**midrp_tog_runtime_params['gnn_params']).to(device)
    midrp_mlp_model = dh.load_torch_model_by_keystring(checkpoint_keystring, midrp_mlp_model, 'midrp_mlp_model', device)
    gnn_model = dh.load_torch_model_by_keystring(checkpoint_keystring, gnn_model, 'gnn_model', device)
    gnn_model.eval(), midrp_mlp_model.eval()
    return gnn_model, midrp_mlp_model

def load_midrp_models(checkpoint_keystring, mode):
    if mode == 'strawman2':
        return load_midrp_gnn_only_model(checkpoint_keystring)
    elif mode == 'proposed':
        return load_midrp_gnn_mlp_seq_model(checkpoint_keystring)
    elif mode == 'strawman1':
        return load_midrp_gnn_mlp_tog_model(checkpoint_keystring)
    else:
        raise ValueError(f"Unsupported mode: {mode}")

def load_mfspp_models(checkpoint_keystring, mode, midrp_models, midrp_mode):
    # mfspp_mode = ['fix-gnn', 'tl_w_gnn', 'denovo']
    mfspp_runtime_params = dh.load_params_by_keystring(checkpoint_keystring)
    if mode == 'tl_w_gnn':
        ## the model init params is the same as the midrp model
        if midrp_mode == 'strawman2':
            mfspp_gnn_model = copy.deepcopy(midrp_models)
        else:
            mfspp_gnn_model = copy.deepcopy(midrp_models[0])
    elif mode == 'denovo':
        ## if tl_w_gnn, then the gnn model is different from the midrp gnn model
        mfspp_gnn_model = gm.CustomGNN1(**mfspp_runtime_params['mfspp_gnn_params']).to(device)
    mfspp_mlp_model = gm.MFSPPredictor1(layers=mfspp_runtime_params['mfspp_mlp_layers'], dropouts=mfspp_runtime_params['mfspp_mlp_dropouts']).to(device)
    mfspp_gnn_model = dh.load_torch_model_by_keystring(checkpoint_keystring, mfspp_gnn_model, 'mfspp_gnn_model', device)
    mfspp_mlp_model = dh.load_torch_model_by_keystring(checkpoint_keystring, mfspp_mlp_model, 'mfspp_mlp_model', device)
    mfspp_gnn_model.eval(), mfspp_mlp_model.eval()
    return mfspp_gnn_model, mfspp_mlp_model

########################### define the inference functions ############################
def midrp_infer_gnn_only(gnn_model, data, N_forward_layers):
    ## all data should be with the same h_max
    out = gnn_model(data.x, data.edge_index, data.edge_attr, N_forward_layers=N_forward_layers, graph_embedding=False, batch=data.batch)
    out = global_max_pool(out, data.batch)
    return torch.clamp(out, min=0)
def midrp_infer_gnn_mlp(gnn_model, mlp_model, data, N_forward_layers):
    # with torch.no_grad():
    graph_embedding = gnn_model(data.x, data.edge_index, data.edge_attr, N_forward_layers=N_forward_layers, graph_embedding=True, batch=data.batch)
    pred = mlp_model(graph_embedding)
    return torch.clamp(pred, min=0)

def midrp_infer(models, data, N_forward_layers, mode):
    if mode == 'strawman2':
        return midrp_infer_gnn_only(models, data, N_forward_layers)
    elif mode == 'proposed' or mode == 'strawman1':
        return midrp_infer_gnn_mlp(models[0], models[1], data, N_forward_layers)
    else:
        raise ValueError(f"Unsupported mode: {mode}")


###################### define the MFSP Predictor prediction function ###########################
def predict_01_mfsp(mfspp_gnn, mfspp_mlp, data, N_forward_layers, mfspp_mode):
    if mfspp_mode == 'denovo':
        embedding = mfspp_gnn(data.x[:,:3], data.edge_index, data.edge_attr, N_forward_layers=N_forward_layers, graph_embedding=True, batch=data.batch)
    else:
        embedding = mfspp_gnn(data.x, data.edge_index, data.edge_attr, N_forward_layers=N_forward_layers, graph_embedding=True, batch=data.batch)
    pred = mfspp_mlp(embedding)
    return torch.clamp(pred, min=0)


def modify_fire_point(pred_01_fire_point, data, random_virtual_fire_point=False):
    if isinstance(data, list): # this means set virtual fire point at the very beginning (for MFSP Predictor)
        for d in data:
            if random_virtual_fire_point:
                pred_01_fire_point = torch.rand(3, device=device)
            boundary = d.x[:, 0:3].max(dim=0)[0].squeeze() # [x_max, y_max, z_max]
            fire_point = pred_01_fire_point * boundary
            fire_point_level = torch.floor(fire_point[2] / d.normalized_unit_height).reshape(1) / 7
            fire_point_expanded = torch.cat([fire_point, fire_point_level]).unsqueeze(0).repeat(d.x.size(0), 1)
            
            diff = d.x[:, 0:4] - fire_point_expanded
            diff[:, :3] = torch.abs(diff[:, :3])
            distance = torch.norm(diff[:, 0:3], dim=1)
            d.x = torch.cat([d.x[:, 0:4], fire_point_expanded, diff, distance.reshape(-1, 1)], dim=1)
        return data
    elif data.batch is None:
        boundary = data.x[:, 0:3].max(dim=0)[0].squeeze() # [x_max, y_max, z_max]
        fire_point = pred_01_fire_point.squeeze() * boundary
        fire_point_level = torch.floor(fire_point[2] / data.normalized_unit_height).reshape(1) / 7
        fire_point_expanded = torch.cat([fire_point, fire_point_level]).unsqueeze(0).repeat(data.x.size(0), 1)
        
        diff = data.x[:, 0:4] - fire_point_expanded
        diff[:, :3] = torch.abs(diff[:, :3])
        distance = torch.norm(diff[:, 0:3], dim=1)
        data.x = torch.cat([data.x[:, 0:4], fire_point_expanded, diff, distance.reshape(-1, 1)], dim=1)
        return data

    else: # this means modify the fire point (for midr Predictor), retain the compuation graph is important
        assert not random_virtual_fire_point 
        boundary = scatter(data.x[:, 0:3], data.batch, dim=0, reduce='max')
        fire_point = (pred_01_fire_point * boundary)
        fire_point_level = (fire_point[:, 2] / data.normalized_unit_height).floor() / 7
        fire_point_expanded = torch.cat([fire_point, fire_point_level.unsqueeze(1)], dim=1)
        coord_diff = (data.x[:, 0:3] - fire_point_expanded[:, 0:3]).abs()
        height_diff = data.x[:, 3] - fire_point_expanded[:, 3]
        distance = torch.norm(coord_diff, dim=1)
        data.x = torch.cat([data.x[:, :4], fire_point_expanded, coord_diff, height_diff.reshape(-1,1), distance.reshape(-1,1)], dim=1)
        return data


def normalized_fire_to_real_fire_coord(fire_point):
    coord_min = 0
    coord_max = 35000
    return fire_point * (coord_max - coord_min) + coord_min

def real_fire_coord_to_normalized_fire_coord(fire_point):
    coord_min = 0
    coord_max = 35000
    return (fire_point - coord_min) / (coord_max - coord_min)

def _01_fire_coord_to_real_fire_coord(fire_point, data):
    if data.batch is None:
        boundary = data.x[:, 0:3].max(dim=0)[0].squeeze()
    else:
        boundary = scatter(data.x[:, 0:3], data.batch, dim=0, reduce='max')
    fire_point = (fire_point * boundary)
    return normalized_fire_to_real_fire_coord(fire_point)

def mfspp_infer(models, data, N_forward_layers, mode, virtual_fire_01coord):
    if mode != 'denovo':
        data = modify_fire_point(torch.tensor(virtual_fire_01coord, device=device), data, random_virtual_fire_point=False)
    pred_01_fire_point = predict_01_mfsp(models[0], models[1], data, N_forward_layers, mode)
    pred_real_fire_point = _01_fire_coord_to_real_fire_coord(pred_01_fire_point, data)
    return pred_01_fire_point, pred_real_fire_point


########################### load the dataset ###################################################
## if the dataset is already processed, then just load the labeled data directly: 
midrp_models = load_midrp_models(midrp_checkpoint_keystring, midrp_mode)
mfspp_models = load_mfspp_models(mfspp_checkpoint_keystring, mfspp_mode, midrp_models, midrp_mode)


train_dataset_list, test_dataset_list = dh.check_and_load_midrp_processed_full_info_train_test_dataset(mdrp_checkpoint_keystring=mdrp_checkpoint_keystring, dataset_name=dataset_name, gravity_filter_threshold=gravity_filter_threshold, device=device)
if train_dataset_list is None or test_dataset_list is None:
    print("Dataset not found, loading the raw dataset and processing it...")

    train_dataset_list, test_dataset_list = dh.load_unlabeled_train_test_normalized_dataset(dataset_name=dataset_name, gravity_filter_threshold=gravity_filter_threshold, device=device) 
    print(f"Train dataset size: {len(train_dataset_list)}, Test dataset size: {len(test_dataset_list)}")

    def pred_all_room_midr(data, midrp_models, midrp_mode):
        # shoud add:  fire_point_list,  fire_01_point_list, fire_real_point_list, pred_midr_list
        expanded_dataset_list = []
        h_max = data.h_max.item()
        normalized_unit_lengths = data.normalized_unit_lengths
        boundary = data.x[:, 0:3].max(dim=0)[0].squeeze()
        for wx_ in range(data.w_max.item()):
            for dy_ in range(data.d_max.item()):
                for hz_ in range(data.h_max.item()):
                    data_ = copy.deepcopy(data)
                    # data_ = copy.copy(data)
                    data_.fire_room_idx = torch.tensor([wx_, dy_, hz_], device=device, dtype=torch.int)

                    fire_point = torch.zeros(4, device=device)
                    fire_point[0:3] = torch.tensor([(wx_ + 0.5), (dy_ + 0.5), (hz_ + 0.5)], device=device) * normalized_unit_lengths
                    fire_point[3] = torch.tensor(hz_, device=device, dtype=torch.float32) / 7

                    data_.fire_point = fire_point
                    data_.gt_real_mfsp = normalized_fire_to_real_fire_coord(fire_point[0:3]).reshape(1,3)
                    fire_point_expand = fire_point.unsqueeze(0).repeat(data_.x.size(0), 1)
                    diff = data_.x[:, 0:4] - fire_point_expand
                    distance = torch.norm(diff[:, 0:3], dim=1)
                    data_.x = torch.cat([data_.x[:, :4], fire_point_expand, diff, distance.reshape(-1, 1)], dim=1)
                    expanded_dataset_list.append(data_)
        loader = DataLoader(expanded_dataset_list, batch_size=len(expanded_dataset_list), shuffle=False)
        with torch.no_grad():
            for data_ in loader:
                pred_midr = midrp_infer(midrp_models, data_, h_max, midrp_mode).squeeze()

        # ## only finding the max is OK
        for i, data_ in enumerate(expanded_dataset_list):
            data_.pred_midr = pred_midr[i]

        fire_point_list = torch.stack([data_.fire_point[0:3] for data_ in expanded_dataset_list], dim=0).squeeze()
        fire_01_point_list = torch.stack([(data_.fire_room_idx + 0.5) / torch.tensor([data_.w_max.item(), data_.d_max.item(), data_.h_max.item()], device=device, dtype=torch.float32) for data_ in expanded_dataset_list], dim=0).squeeze()
        fire_real_point_list = torch.stack([data_.gt_real_mfsp for data_ in expanded_dataset_list], dim=0).squeeze()
        fire_room_idx_list = torch.stack([data_.fire_room_idx for data_ in expanded_dataset_list], dim=0).squeeze().reshape(-1, 3)
        pred_midr_list = torch.stack([data_.pred_midr for data_ in expanded_dataset_list], dim=0).squeeze()
        ## we also need room idx list!
        

        # 根据 pred_midr_list 排序并直接赋值
        sorted_indices = torch.argsort(pred_midr_list, descending=True)

        data.fire_point_list = fire_point_list[sorted_indices, :]
        data.fire_01_point_list = fire_01_point_list[sorted_indices, :]
        data.fire_real_point_list = fire_real_point_list[sorted_indices, :]
        data.fire_room_idx_list = fire_room_idx_list[sorted_indices, :]
        data.pred_midr_list = pred_midr_list[sorted_indices]
        return 

        
    for data in train_dataset_list:
        pred_all_room_midr(data, midrp_models, midrp_mode)

    for data in test_dataset_list:
        pred_all_room_midr(data, midrp_models, midrp_mode)

    ############################## then save the processed dataset ############################
    dh.save_midrp_processed_full_info_train_test_dataset(train_dataset_list, test_dataset_list, mdrp_checkpoint_keystring, dataset_name=dataset_name,gravity_filter_threshold=gravity_filter_threshold)
############################# predict the MFSP ############################################

for data in train_dataset_list:
    with torch.no_grad():
        pred_01_fire_point, pred_real_fire_point = mfspp_infer(mfspp_models, data, data.h_max.item(), mfspp_mode, virtual_fire_01coord) 

        pred_mfsp_room_idx = torch.floor(pred_real_fire_point / data.unit_lengths).squeeze().int() 
        pred_mfsp_room_idx = torch.min(pred_mfsp_room_idx, torch.tensor([data.w_max.item()-1, data.d_max.item()-1, data.h_max.item()-1], device=device, dtype=torch.int))
        
        data.pred_mfsp_room_idx = pred_mfsp_room_idx
        data.pred_mfsp_real_coord = pred_real_fire_point
        
        modified_data = modify_fire_point(pred_01_fire_point, data)
        pred_midr = midrp_infer(midrp_models, modified_data, N_forward_layers=data.h_max.item(), mode=midrp_mode).squeeze()
        data.pred_midr = pred_midr
        

for data in test_dataset_list:
    with torch.no_grad():
        pred_01_fire_point, pred_real_fire_point = mfspp_infer(mfspp_models, data, data.h_max.item(), mfspp_mode, virtual_fire_01coord) 

        pred_mfsp_room_idx = torch.floor(pred_real_fire_point / data.unit_lengths).squeeze().int() 
        pred_mfsp_room_idx = torch.min(pred_mfsp_room_idx, torch.tensor([data.w_max.item()-1, data.d_max.item()-1, data.h_max.item()-1], device=device, dtype=torch.int))
        
        data.pred_mfsp_room_idx = pred_mfsp_room_idx
        data.pred_mfsp_real_coord = pred_real_fire_point

        modified_data = modify_fire_point(pred_01_fire_point, data)
        pred_midr = midrp_infer(midrp_models, modified_data, N_forward_layers=data.h_max.item(), mode=midrp_mode).squeeze()
        data.pred_midr = pred_midr


################################# compare with ground truth and visualize the results ########################################

train_real_distance_list = [torch.norm(data.pred_mfsp_real_coord - data.fire_real_point_list[0, :].reshape(1,3)).item() for data in train_dataset_list]
test_real_distance_list = [torch.norm(data.pred_mfsp_real_coord - data.fire_real_point_list[0, :].reshape(1,3)).item() for data in test_dataset_list]
train_room_distance_list = [torch.norm(data.pred_mfsp_room_idx.float() - data.fire_room_idx_list[0, :].float()).item() for data in train_dataset_list]
test_room_distance_list = [torch.norm(data.pred_mfsp_room_idx.float() - data.fire_room_idx_list[0, :].float()).item() for data in test_dataset_list]
train_room_rank_list = [torch.nonzero(torch.all(data.pred_mfsp_room_idx.int() == data.fire_room_idx_list.int(), dim=1))[0].item() for data in train_dataset_list]
test_room_rank_list = [torch.nonzero(torch.all(data.pred_mfsp_room_idx.int() == data.fire_room_idx_list.int(), dim=1))[0].item() for data in test_dataset_list]
train_midr_list = [data.pred_midr.item() for data in train_dataset_list]
test_midr_list = [data.pred_midr.item() for data in test_dataset_list]


train_avg_real_distance = np.mean(train_real_distance_list)
test_avg_real_distance = np.mean(test_real_distance_list)
train_avg_room_distance = np.mean(train_room_distance_list)
test_avg_room_distance = np.mean(test_room_distance_list)
train_avg_room_rank = np.mean(train_room_rank_list)
test_avg_room_rank = np.mean(test_room_rank_list)
train_avg_midr = np.mean(train_midr_list)
test_avg_midr = np.mean(test_midr_list)

print(f"Train Avg Real Distance: {train_avg_real_distance}, Test Avg Real Distance: {test_avg_real_distance}")
print(f"Train Avg Room Distance: {train_avg_room_distance}, Test Avg Room Distance: {test_avg_room_distance}")
print(f"Train Avg Room Rank: {train_avg_room_rank}, Test Avg Room Rank: {test_avg_room_rank}")
print(f"Train Avg midr: {train_avg_midr}, Test Avg midr: {test_avg_midr}")

######################## plot the cdf of the real distance, room distance and room rank ############################

def plot_cdf_of_list(ax, data_list, label):
    sorted_data = np.sort(data_list)
    cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    ax.plot(sorted_data, cdf, marker='none', linestyle='-', label=label)

fig, ax = plt.subplots(1,1, figsize=(8, 6))
plot_cdf_of_list(ax, train_real_distance_list, 'Train Real Distance')
plot_cdf_of_list(ax, test_real_distance_list, 'Test Real Distance')
ax.set_xlabel('Real Distance')
ax.set_ylabel('CDF')
ax.set_title('CDF of Real Distance')
ax.grid(True)
ax.legend()
dh.save_runtime_fig(keystring, fig, 'real_distance_cdf')

fig, ax = plt.subplots(1,1, figsize=(8, 6))
plot_cdf_of_list(ax, train_room_distance_list, 'Train Room Distance')
plot_cdf_of_list(ax, test_room_distance_list, 'Test Room Distance')
ax.set_xlabel('Room Distance')
ax.set_ylabel('CDF')
ax.set_title('CDF of Room Distance')
ax.grid(True)
ax.legend()
dh.save_runtime_fig(keystring, fig, 'room_distance_cdf')

fig, ax = plt.subplots(1,1, figsize=(8, 6))
plot_cdf_of_list(ax, train_room_rank_list, 'Train Room Rank')
plot_cdf_of_list(ax, test_room_rank_list, 'Test Room Rank')
ax.set_xlabel('Room Rank')
ax.set_ylabel('CDF')
ax.set_title('CDF of Room Rank')
ax.grid(True)
ax.legend()
dh.save_runtime_fig(keystring, fig, 'room_rank_cdf')

########################### finally save the results ########################################
param_name_list = [
    'dataset_name',
    'train_ratio',
    'random_seed',
    'mfspp_checkpoint_keystring',
    'mfspp_checkpoint_epoch',
    'midrp_checkpoint_keystring',
    'midrp_checkpoint_epoch',
    'mfspp_mode',
    'midrp_mode',
    'test_num',
    'timestamp_str',
    'keystring',
    'script_name',
    'train_avg_real_distance',
    'test_avg_real_distance',
    'train_avg_room_distance',
    'test_avg_room_distance',
    'train_avg_room_rank',
    'test_avg_room_rank',
    'train_avg_midr',
    'virtual_fire_01coord',
    'test_avg_midr',
]

dh.save_runtime_params(keystring, locals(), param_name_list)

result_dict = {
    'train_real_distance_list': np.array(train_real_distance_list),
    'test_real_distance_list': np.array(test_real_distance_list),
    'train_room_distance_list': np.array(train_room_distance_list),
    'test_room_distance_list': np.array(test_room_distance_list),
    'train_room_rank_list': np.array(train_room_rank_list),
    'test_room_rank_list': np.array(test_room_rank_list),
    'train_midr_list': np.array(train_midr_list),
    'test_midr_list': np.array(test_midr_list),
}

dh.save_runtime_csv_results(keystring, result_dict)
