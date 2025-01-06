import os
import time
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import argparse
import numpy as np
import pandas as pd
import sys
import opensees.openseespy as ops
import random
sys.path.append(os.path.dirname(sys.path[0]))
from utils import data_helper as dh
import re
import torch 
import torch_geometric as pyg
from torch_geometric.data import Data
import subprocess
import shutil
import copy

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset_name = 'drs3v6'
dataset_pt_folder = os.path.join(dh.get_datasets_dir(), dataset_name, 'pt-data').replace("\\", "/")

###################### load the origin data ##############################
dataset_w_gravity_whole_file = os.path.join(dataset_pt_folder, f'{dataset_name}_w_gravity_dr.pt').replace("\\", "/")
if not os.path.exists(dataset_w_gravity_whole_file):
    dataset_w_gravity_list = []
    w_gravity_file_list = os.listdir(dataset_pt_folder)
    w_gravity_file_list = [f  for f in w_gravity_file_list if (f.endswith('_w_gravity_dr.pt') and f.startswith('m')) ]
    w_gravity_file_list.sort()
    for model_pt_w_gravity_file in w_gravity_file_list:
        print(f"loading {model_pt_w_gravity_file}")
        torch_model_data_list = torch.load(os.path.join(dataset_pt_folder, model_pt_w_gravity_file).replace("\\", "/"))
        dataset_w_gravity_list.extend(torch_model_data_list)
    torch.save(dataset_w_gravity_list, dataset_w_gravity_whole_file)
else:
    dataset_w_gravity_list = torch.load(dataset_w_gravity_whole_file)

valid_dataset_list =  [data for data in dataset_w_gravity_list if data.max_drift_ratio < 100]
valid_model_number_count_dict = {model_number:0 for model_number in range(1,3001)}
for data in valid_dataset_list:
    valid_model_number_count_dict[data.model_number] += 1
valid_model_number_list = [model_number for model_number, count in valid_model_number_count_dict.items() if count == 30]
print(f"valid model count: {len(valid_model_number_list)}")

valid_dataset_list = [data for data in valid_dataset_list if data.model_number in valid_model_number_list]
print(f"valid case count: {len(valid_dataset_list)}")


############################## filter and save ##############################
gravity_filter_threshold = 1.0
fire_filter_threshold = 100

dataset_remaining_list = [data for data in valid_dataset_list if data.gravity_max_drift_ratio < gravity_filter_threshold and data.max_drift_ratio < fire_filter_threshold]
model_number_count_dict = {model_number:0 for model_number in range(1,3001)}
for data in dataset_remaining_list:
    model_number_count_dict[data.model_number] += 1
model_number_remaining = [model_number for model_number, count in model_number_count_dict.items() if count == 30]
print(f"{len(model_number_remaining)} structures left" )
dataset_remaining_list = [data for data in dataset_remaining_list if data.model_number in model_number_remaining]

filtered_filename = f"{dataset_name}_filtered_g{gravity_filter_threshold:.2f}_gf{fire_filter_threshold:.2f}.pt"
filtered_filepath = os.path.join(dataset_pt_folder, filtered_filename)
torch.save(dataset_remaining_list, filtered_filepath)
train_ratio = 0.8
random_split = True
random.shuffle(model_number_remaining)
print(model_number_remaining)
train_model_number_list, test_model_number_list = model_number_remaining[:int(train_ratio * len(model_number_remaining))], model_number_remaining[int(train_ratio * len(model_number_remaining)):]
train_dataset_list = [data for data in dataset_remaining_list if data.model_number in train_model_number_list]
test_dataset_list = [data for data in dataset_remaining_list if data.model_number in test_model_number_list]
train_dataset_filepath = filtered_filepath.replace("_filtered", "_filtered_train")
test_dataset_filepath = filtered_filepath.replace("_filtered", "_filtered_test")
torch.save(train_dataset_list, train_dataset_filepath)
torch.save(test_dataset_list, test_dataset_filepath)

################## normalize the dataset and save the normalized version ############################
node_coord_min = 0
node_coord_max = 35000
level_max = 7
edge_attr_norm_idx = torch.tensor([0, 1, 2, 3, 4, 5],  dtype=torch.long)
edge_attr_norm_min = torch.tensor([210000*0.8, 250*0.8, 0.001*0.8, 3000, 0, -7.5], dtype=torch.float)
edge_attr_norm_max = torch.tensor([210000*1.2, 250*1.2, 0.001*1.2, 5000, 7, -0.5], dtype=torch.float)
def normalize_dataset_list(dataset_list):
    new_dataset_list = []
    for data in dataset_list:
        data = copy.deepcopy(data)
        data.node_indices = (data.x[:,:3] / data.unit_lengths).long()
        data.w_max = data.node_indices[:, 0].max().squeeze()
        data.d_max = data.node_indices[:, 1].max().squeeze()
        data.h_max = data.node_indices[:, 2].max().squeeze()
        data.normalized_unit_lengths = ((data.unit_lengths - node_coord_min) / (node_coord_max - node_coord_min)).squeeze()
        data.normalized_unit_height = ((data.unit_lengths[2] - node_coord_min) / (node_coord_max - node_coord_min)).squeeze()

        data.x[:, :3] = (data.x[:, :3] - node_coord_min) / (node_coord_max - node_coord_min) 
        data.x[:, 3] = data.x[:, 3] / level_max
        data.edge_attr[:, edge_attr_norm_idx] = (data.edge_attr[:, edge_attr_norm_idx] - edge_attr_norm_min) / (edge_attr_norm_max - edge_attr_norm_min)

        data.y_max = data.max_drift_ratio

        data.fire_point[:3] = (data.fire_point[:3] - node_coord_min) / (node_coord_max - node_coord_min)
        data.fire_point[3] = data.fire_point[3] / level_max
        fire_point_expanded = data.fire_point.unsqueeze(0).repeat(data.x.shape[0], 1)
        diff = data.x[:, :4] - fire_point_expanded
        diff[:, :3] = torch.abs(diff[:, :3])
        distance = torch.norm(diff[:, :3], dim=1, keepdim=True)
        data.x = torch.cat([data.x, fire_point_expanded, diff, distance], dim=1)
        new_dataset_list.append(data)
    return new_dataset_list

train_dataset_list_normalized = normalize_dataset_list(train_dataset_list)
test_dataset_list_normalized = normalize_dataset_list(test_dataset_list)
train_dataset_normalized_filepath = train_dataset_filepath.replace("_filtered", "_filtered_normalized")
test_dataset_normalized_filepath = test_dataset_filepath.replace("_filtered", "_filtered_normalized")
torch.save(train_dataset_list_normalized, train_dataset_normalized_filepath)
torch.save(test_dataset_list_normalized, test_dataset_normalized_filepath)