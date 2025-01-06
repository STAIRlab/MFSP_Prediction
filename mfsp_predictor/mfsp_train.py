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
from collections import defaultdict
from utils import data_helper as dh
import matplotlib.pyplot as plt
import copy
import gnn_models.gnn_model as gm
from torch_scatter import scatter

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset_name = 'drs4v2'
train_ratio = 0.8
random_seed = 1207
gravity_filter_threshold = 1.0

midrp_dim = 64
midrp_mode = 'proposed'
midrp_checkpoint_keystring = '20241223151005bPJuxs'


########### modes: 'tl_w_gnn', 'denovo'
# mfspp_mode = 'denovo'
mfspp_mode = 'tl_w_gnn'

learning_rate = 1e-3
batch_size = 128
epoch_num_total = 520
# loss_method = 'real-dist-mse'
loss_method = 'hybrid-real-dist-mse'

hybrid_midr_weight = - 10
hybrid_real_dist_weight = 1e-6 

mfspp_mlp_layers = [midrp_dim * 2, midrp_dim * 2, 3]
mfspp_mlp_dropouts = 0.0
mfspp_mlp_class = gm.MFSPPredictor1


if  mfspp_mode == 'tl_w_gnn':
    random_virtual_fire_point = True
    virtual_fire_01coord = [0.5, 0.5, 0.5]

elif mfspp_mode == 'denovo':
    mfspp_gnn_params = dict(
        node_attr_dim=3, 
        edge_attr_dim=9, 
        node_emb_dim=64,
        final_out_dim=1,
        num_gnn_layers=7,
        shared_hidden=False,
        aggr='max', # max seems better
        edge_emb_flag=True, # seems not very useful to set to True
        edge_emb_dim=32,
        inter_layer_pypass_flag=True,
        extended_message_in_flag=False, # try both
        message_mlp_hidden_layers=[128,],
        message_dropout_rates=0.1,
        intra_layer_pypass_flag=True,
        update_with_target_flag=False, ## try both True and False
        update_mlp_hidden_layers=[128,],
        update_dropout_rates=0.1,
        node_init_mlp_hidden_layers=[64,],
        node_init_mlp_dropout_rates=0.1,
        edge_init_mlp_hidden_layers=[64,],
        edge_init_mlp_dropout_rates=0.1,
        post_mlp_hidden_layers=[],
        post_mlp_dropout_rates=0.1,
        update_edge_flag=True,
        edge_update_mlp_hidden_layers=[64],
        edge_update_mlp_dropout_rates=0,
    )
    mfspp_gnn_class = gm.CustomGNN1



##################### backup the script ############################
timestamp_str = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
keystring = timestamp_str + ''.join(random.choices(string.ascii_letters, k=6))
script_name = os.path.basename(__file__)
dh.backup_script(keystring, __file__)

torch.manual_seed(random_seed), np.random.seed(random_seed), random.seed(random_seed)

mfspp_mlp_model = mfspp_mlp_class(layers=mfspp_mlp_layers, dropouts=mfspp_mlp_dropouts).to(device)
mfspp_mlp_model_name = str(type(mfspp_mlp_model))
mfspp_gnn_model_name = None
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

########################## load the midr Predictor model #######################################
midrp_models = load_models(midrp_checkpoint_keystring, midrp_mode)


########################## define the MFSP Predictor model #######################################
if mfspp_mode == 'tl_w_gnn':
    if midrp_mode == 'strawman2':
        mfspp_gnn_model = copy.deepcopy(midrp_models)
    else:
        mfspp_gnn_model = copy.deepcopy(midrp_models[0])
elif mfspp_mode == 'denovo':
    mfspp_gnn_model = mfspp_gnn_class(**mfspp_gnn_params).to(device)
    mfspp_gnn_model_name = str(type(mfspp_gnn_model))
else:
    raise ValueError(f"Unsupported mode: {mfspp_mode}")    

###################### define the MFSP Predictor prediction function ###########################
def predict_01_mfsp(mfspp_gnn, mfspp_mlp, data, N_forward_layers, mfspp_mode):
    if mfspp_mode == 'denovo':
        embedding = mfspp_gnn(data.x[:,:3], data.edge_index, data.edge_attr, N_forward_layers=N_forward_layers, graph_embedding=True, batch=data.batch)
    elif mfspp_mode == 'tl_w_gnn':
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
    else: # this means modify the fire point (for midr Predictor), retain the compuation graph is important
        assert not random_virtual_fire_point 
        boundary = scatter(data.x[:, 0:3], data.batch, dim=0, reduce='max')
        fire_point = (pred_01_fire_point * boundary)
        fire_point_level = (fire_point[:, 2] / data.normalized_unit_height).floor() / 7
        fire_point_expanded = torch.cat([fire_point, fire_point_level.unsqueeze(1)], dim=1)
        fire_point_expanded = fire_point_expanded.repeat_interleave(data.batch.bincount(), dim=0)

        coord_diff = (data.x[:, 0:3] - fire_point_expanded[:, 0:3]).abs()
        height_diff = data.x[:, 3] - fire_point_expanded[:, 3]
        distance = torch.norm(coord_diff, dim=1)
        data.x = torch.cat([data.x[:, :4], fire_point_expanded, coord_diff, height_diff.reshape(-1,1), distance.reshape(-1,1)], dim=1)
        return data

def normalized_fire_to_real_fire_coord(fire_point):
    coord_min = 0
    coord_max = 35000
    return fire_point * (coord_max - coord_min) + coord_min

def _01_fire_coord_to_real_fire_coord(fire_point, data):
    boundary = scatter(data.x[:, 0:3], data.batch, dim=0, reduce='max')
    fire_point = (pred_01_fire_point * boundary)
    return normalized_fire_to_real_fire_coord(fire_point)

########################### load the dataset ###################################################
## if the dataset is already processed, then just load the labeled data directly: 
train_dataset_list, test_dataset_list = dh.check_and_load_midrp_processed_train_test_dataset(midrp_checkpoint_keystring=midrp_checkpoint_keystring, dataset_name=dataset_name, gravity_filter_threshold=gravity_filter_threshold, device=device)
if train_dataset_list is None or test_dataset_list is None:
    print("Dataset not found, loading the raw dataset and processing it...")

    train_dataset_list, test_dataset_list = dh.load_unlabeled_train_test_normalized_dataset(dataset_name=dataset_name, gravity_filter_threshold=gravity_filter_threshold, device=device)
    print(f"Train dataset size: {len(train_dataset_list)}, Test dataset size: {len(test_dataset_list)}")

    def find_gt_mfsp(data, midrp_models, midrp_mode):
        # suppose to add 1) 
        expanded_dataset_list = []
        h_max = data.h_max.item()
        normalized_unit_lengths = data.normalized_unit_lengths
        boundary = data.x[:, 0:3].max(dim=0)[0].squeeze()
        for wx_ in range(data.w_max.item()):
            for dy_ in range(data.d_max.item()):
                for hz_ in range(data.h_max.item()):
                    data_ = copy.deepcopy(data)
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
        for data_ in loader:
            pred_midr = midrp_infer(midrp_models, data_, h_max, midrp_mode).squeeze()

        # ## only finding the max is OK
        for i, data_ in enumerate(expanded_dataset_list):
            data_.pred_midr = pred_midr[i]
        data_ = max(expanded_dataset_list, key=lambda x: x.pred_midr)
        data.fire_room_idx = data_.fire_room_idx.reshape(1,3)
        data.gt_01_mfsp = ((data.fire_room_idx + 0.5) / torch.tensor([data.w_max.item(), data.d_max.item(), data.h_max.item()], device=device, dtype=torch.float32) ).reshape(1, 3)
        data.gt_real_mfsp = data_.gt_real_mfsp
        data.x = data_.x


    for data in train_dataset_list:
        find_gt_mfsp(data, midrp_models, midrp_mode)

    for data in test_dataset_list:
        find_gt_mfsp(data, midrp_models, midrp_mode)

    ############################## then save the processed dataset ############################
    dh.save_midrp_processed_train_test_dataset(train_dataset_list, test_dataset_list, midrp_checkpoint_keystring, dataset_name=dataset_name, gravity_filter_threshold=gravity_filter_threshold)

########################### if needed, set the virtual fire point ############################
if mfspp_mode == 'tl_w_gnn' :
    train_dataset_list = modify_fire_point(torch.tensor(virtual_fire_01coord,device=device), train_dataset_list, random_virtual_fire_point)
    test_dataset_list = modify_fire_point(torch.tensor(virtual_fire_01coord,device=device), test_dataset_list, random_virtual_fire_point=False)

train_loaders = dh.create_loaders(train_dataset_list, batch_size, shuffle=True)
test_loaders = dh.create_loaders(test_dataset_list, batch_size, shuffle=False)
train_batch_num, test_batch_num = sum([len(loader) for loader in train_loaders.values()]), sum([len(loader) for loader in test_loaders.values()])
train_h_max_counts, test_h_max_counts= dh.count_from_loaders(train_loaders), dh.count_from_loaders(test_loaders)
print('Training:',json.dumps(train_h_max_counts, indent=4))
print('Testing:',json.dumps(test_h_max_counts, indent=4))


########################### define the optimizer #################################
mfspp_mlp_optimizer = torch.optim.Adam(mfspp_mlp_model.parameters(), lr=learning_rate)
mfspp_gnn_optimizer = torch.optim.Adam(mfspp_gnn_model.parameters(), lr=learning_rate)

loss_func_mse = torch.nn.MSELoss()


train_avg_pred_midr_record, test_avg_pred_midr_record = [], []
train_avg_real_dist_mse_record, test_avg_real_dist_mse_record = [], []
train_avg_real_dist_mae_record, test_avg_real_dist_mae_record = [], []
train_hybird_loss_record, test_hybird_loss_record = [], []

for epoch in range(epoch_num_total):
    if (epoch > 0) \
        and (mfspp_mode == 'tl_w_gnn') \
        and (random_virtual_fire_point):
        train_dataset_list = modify_fire_point(torch.tensor(virtual_fire_01coord, device=device), train_dataset_list, random_virtual_fire_point)
        train_loaders = dh.create_loaders(train_dataset_list, batch_size, shuffle=True)

    mfspp_mlp_model.train()
    mfspp_gnn_model.train()

    train_hybrid_loss, train_real_dist_mse_loss, train_real_dist_mae_loss, train_midr_loss = 0, 0, 0, 0
    remaining_counts = train_h_max_counts.copy()
    while sum(remaining_counts.values()) > 0:
        probabilities = [count / sum(remaining_counts.values()) for count in remaining_counts.values()]
        selected_h_max = random.choices(list(train_loaders.keys()), probabilities, k=1)[0]
        loader = train_loaders[selected_h_max]

        ########### train the model with the selected h_max ###########
        for data in loader:
            remaining_counts[selected_h_max] = max(0, remaining_counts[selected_h_max] - batch_size)

            ################## first predict the MFSP ##################
            pred_01_fire_point = predict_01_mfsp(mfspp_gnn_model, mfspp_mlp_model, data, N_forward_layers=selected_h_max, mfspp_mode=mfspp_mode)
            pred_real_fire_point = _01_fire_coord_to_real_fire_coord(pred_01_fire_point, data)

            #################### then backpropagate the loss ##################
            mfspp_mlp_optimizer.zero_grad()
            if (mfspp_mode == 'denovo') \
                or ((mfspp_mode == 'tl_w_gnn') and (epoch > epoch_num_total - epoch_num_tog)):
                mfspp_gnn_optimizer.zero_grad()
            

            if loss_method == 'real-dist-mse':
                loss = loss_func_mse(pred_real_fire_point, data.gt_real_mfsp)
                train_real_dist_mse_loss += loss.item()
            elif loss_method == 'hybrid-real-dist-mse':
                modified_data = modify_fire_point(pred_01_fire_point, data)
                pred_midr = midrp_infer(midrp_models, modified_data, N_forward_layers=selected_h_max, mode=midrp_mode).squeeze()
                loss = hybrid_midr_weight * pred_midr.mean() + hybrid_real_dist_weight * loss_func_mse(pred_real_fire_point, data.gt_real_mfsp)
                train_hybrid_loss += loss.item()

            ##################### update the model with optimizer ##################
            if epoch != 0:
                loss.backward()
                mfspp_mlp_optimizer.step()
                if (mfspp_mode == 'denovo') \
                    or ((mfspp_mode == 'tl_w_gnn') and (epoch > epoch_num_total - epoch_num_tog)):
                    mfspp_gnn_optimizer.step()
            
            ##################### finally, record the losses #####################
            with torch.no_grad():
                if loss_method != 'real-dist-mse':
                    train_real_dist_mse_loss += loss_func_mse(pred_real_fire_point, data.gt_real_mfsp).item()
                if loss_method != 'real-dist-mae':
                    train_real_dist_mae_loss += loss_func_mae(pred_real_fire_point, data.gt_real_mfsp).item()
                if loss_method != 'hybrid-real-dist-mse':
                    modified_data = modify_fire_point(pred_01_fire_point, data)
                    pred_midr = midrp_infer(midrp_models, modified_data, N_forward_layers=selected_h_max, mode=midrp_mode).squeeze()
                    train_hybrid_loss += hybrid_midr_weight * pred_midr.mean().item() + hybrid_real_dist_weight * loss_func_mse(pred_real_fire_point, data.gt_real_mfsp).item()
                
                train_midr_loss += pred_midr.mean().item()
            break
        
    mfspp_mlp_model.eval()
    mfspp_gnn_model.eval()
    test_01_dist_mse_loss, test_real_dist_mse_loss, test_01_dist_mae_loss, test_real_dist_mae_loss, test_midr_loss = 0, 0, 0, 0, 0
    test_hybird_loss = 0
    with torch.no_grad():
        for h_max, loader in test_loaders.items():
            for data in loader:
                pred_01_fire_point = predict_01_mfsp(mfspp_gnn_model, mfspp_mlp_model, data, N_forward_layers=h_max, mfspp_mode=mfspp_mode)
                pred_real_fire_point = _01_fire_coord_to_real_fire_coord(pred_01_fire_point, data)

                test_real_dist_mse_loss += loss_func_mse(pred_real_fire_point, data.gt_real_mfsp).item()
                test_real_dist_mae_loss += loss_func_mae(pred_real_fire_point, data.gt_real_mfsp).item()

                modified_data = modify_fire_point(pred_01_fire_point, data)
                pred_midr = midrp_infer(midrp_models, modified_data, N_forward_layers=h_max, mode=midrp_mode).squeeze()
                test_midr_loss += pred_midr.mean().item()
                test_hybird_loss += hybrid_midr_weight * pred_midr.mean().item() + hybrid_real_dist_weight * loss_func_mse(pred_real_fire_point, data.gt_real_mfsp).item()
                
    train_avg_real_dist_mse_record.append(train_real_dist_mse_loss / train_batch_num)
    test_avg_real_dist_mse_record.append(test_real_dist_mse_loss / test_batch_num)
    train_avg_real_dist_mae_record.append(train_real_dist_mae_loss / train_batch_num)
    test_avg_real_dist_mae_record.append(test_real_dist_mae_loss / test_batch_num)
    train_avg_pred_midr_record.append(train_midr_loss / train_batch_num) 
    test_avg_pred_midr_record.append(test_midr_loss / test_batch_num) 
    train_hybird_loss_record.append(train_hybrid_loss / train_batch_num) 
    test_hybird_loss_record.append(test_hybird_loss / test_batch_num)

    print(f"Epoch: {epoch}, Train 01 Dist Loss: {train_avg_01_dist_mse_record[-1]}, Test 01 Dist Loss: {test_avg_01_dist_mse_record[-1]}")

final_train_avg_real_dist_mse, final_test_avg_real_dist_mse = train_avg_real_dist_mse_record[-1], test_avg_real_dist_mse_record[-1]
final_train_avg_real_dist_mae, final_test_avg_real_dist_mae = train_avg_real_dist_mae_record[-1], test_avg_real_dist_mae_record[-1]
final_train_avg_pred_midr, final_test_avg_pred_midr = train_avg_pred_midr_record[-1], test_avg_pred_midr_record[-1]
final_train_hybrid_loss, final_test_hybrid_loss = train_hybird_loss_record[-1], test_hybird_loss_record[-1]


################################ plot the loss curve ########################################
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
ax.plot(train_avg_real_dist_mse_record, label='Train Real Dist Loss')
ax.plot(test_avg_real_dist_mse_record, label='Test Real Dist Loss')
ax.set_xlabel('Epoch')
ax.set_ylabel('Average Distance')
ax.legend()
ax.grid()
ax.set_title(f"{keystring}, {dataset_name}")
dh.save_runtime_fig(keystring, fig, 'loss_real_dist_mse')

fig, ax = plt.subplots(1, 1, figsize=(8, 6))
ax.plot(train_avg_01_dist_mae_record, label='Train 01 Dist MAE')
ax.plot(test_avg_01_dist_mae_record, label='Test 01 Dist MAE')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.legend()
ax.grid()
ax.set_title(f"{keystring}, {dataset_name}")
dh.save_runtime_fig(keystring, fig, 'loss_01_dist_mae')

fig, ax = plt.subplots(1, 1, figsize=(8, 6))
ax.plot(train_avg_real_dist_mae_record, label='Train Real Dist MAE')
ax.plot(test_avg_real_dist_mae_record, label='Test Real Dist MAE')
ax.set_xlabel('Epoch')
ax.set_ylabel('Average Distance')
ax.legend()
ax.grid()
ax.set_title(f"{keystring}, {dataset_name}")
dh.save_runtime_fig(keystring, fig, 'loss_real_dist_mae')


fig, ax = plt.subplots(1, 1, figsize=(8, 6))
ax.plot(train_avg_pred_midr_record, label='Train midr')
ax.plot(test_avg_pred_midr_record, label='Test midr')
ax.set_xlabel('Epoch')
ax.set_ylabel('Avg Maximum Drift Ratio')
ax.legend()
ax.grid()
ax.set_title(f"{keystring}, {dataset_name}")
dh.save_runtime_fig(keystring, fig, 'midr')

fig, ax = plt.subplots(1, 1, figsize=(8, 6))
ax.plot(train_hybird_loss_record, label='Train Hybrid Loss')
ax.plot(test_hybird_loss_record, label='Test Hybrid Loss')
ax.set_xlabel('Epoch')
ax.set_ylabel('Hybrid Loss')
ax.legend()
ax.grid()
ax.set_title(f"{keystring}, {dataset_name}, {hybrid_midr_weight}, {hybrid_real_dist_weight}")
dh.save_runtime_fig(keystring, fig, 'hybrid_loss')

################## save the runtime params ############################
param_name_list = [
    'dataset_name',
    'train_ratio',
    'random_seed',
    'gravity_filter_threshold',
    'midrp_mode',
    'midrp_checkpoint_keystring',
    'mfspp_mode',
    'midrp_dim',
    'random_virtual_fire_point',
    'virtual_fire_01coord',
    'mfspp_gnn_params',
    'hybrid_midr_weight',
    'hybrid_real_dist_weight',
    'mfspp_mlp_layers',
    'mfspp_mlp_dropouts',
    'learning_rate',
    'batch_size',
    'epoch_num_total',
    'epoch_num_tog',
    'loss_method',
    'timestamp_str',
    'keystring',
    'script_name',
    'mfspp_mlp_model_name',
    'mfspp_gnn_model_name',
    'final_train_avg_real_dist_mse',
    'final_test_avg_real_dist_mse',
    'final_train_avg_real_dist_mae',
    'final_test_avg_real_dist_mae',
    'final_train_avg_pred_midr',
    'final_test_avg_pred_midr',
    'final_train_hybrid_loss',
    'final_test_hybrid_loss',
]
dh.save_runtime_params(keystring, locals(), param_name_list)

result_dict = {
    'train_avg_real_dist_mse_record': train_avg_real_dist_mse_record,
    'test_avg_real_dist_mse_record': test_avg_real_dist_mse_record,
    'train_avg_real_dist_mae_record': train_avg_real_dist_mae_record,
    'test_avg_real_dist_mae_record': test_avg_real_dist_mae_record,
    'train_avg_pred_midr_record': train_avg_pred_midr_record,
    'test_avg_pred_midr_record': test_avg_pred_midr_record,
    'train_hybird_loss_record': train_hybird_loss_record,
    'test_hybird_loss_record': test_hybird_loss_record,
}

dh.save_runtime_csv_results(keystring, result_dict)
dh.save_runtime_torch_model(keystring, mfspp_mlp_model, 'mfspp_mlp_model')
dh.save_runtime_torch_model(keystring, mfspp_gnn_model, 'mfspp_gnn_model')