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
import matplotlib.pyplot as plt
import gnn_models.gnn_model as gm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

############################ tunable settings #################################
midrp_GNN_checkpoint_keystring = ''
comment = ''
dataset_name = 'drs3v6'

midrp_mlp_layers = [64, 1]
midrp_mlp_dropouts = 0.0
learning_rate = 1e-3
batch_size = 128
epoch_num = 10
post_train_together_epoch_num = 100
post_train_gnn_learning_rate = 1e-4

################################# load the params #################################
midrp_gnn_runtime_params = dh.load_params_by_keystring(midrp_GNN_checkpoint_keystring)
random_seed = midrp_gnn_runtime_params['random_seed']
gravity_filter = midrp_gnn_runtime_params.get('gravity_filter', 1.0)
fire_filter = midrp_gnn_runtime_params.get('fire_filter', 100.0)

############################### define the mdr mlp model #################################
midrp_mlp_model = gm.MIDRPredictor1(layers=midrp_mlp_layers, dropouts=midrp_mlp_dropouts).to(device)
total_params = sum(p.numel() for p in midrp_mlp_model.parameters())
midrp_mlp_name = str(type(midrp_mlp_model))

############################## set the random seed #################################
timestamp_str = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
keystring = timestamp_str + ''.join(random.choices(string.ascii_letters, k=6))
script_name = os.path.basename(__file__)
dh.backup_script(keystring, __file__)

torch.manual_seed(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)
############################## load the mdr gnn model #################################
midrp_gnn_model = gm.CustomGNN1(**midrp_gnn_runtime_params['gnn_params']).to(device)

## if midrp_GNN_checkpoint_epoch exists, load the model from the checkpoint
midrp_gnn_model = dh.load_torch_model_by_keystring(midrp_GNN_checkpoint_keystring, midrp_gnn_model, 'gnn_model', device)
midrp_gnn_model.eval()

############################### load the dataset #################################
train_dataset_list, test_dataset_list = dh.load_gf_train_test_normalized_labeled_dataset(dataset_name, gravity_filter, fire_filter, device)
print(f"Train dataset size: {len(train_dataset_list)}, Test dataset size: {len(test_dataset_list)}")

train_loaders = dh.create_loaders(train_dataset_list, batch_size, shuffle=True)
test_loaders = dh.create_loaders(test_dataset_list, batch_size, shuffle=False)
train_batch_num, test_batch_num = sum([len(loader) for loader in train_loaders.values()]), sum([len(loader) for loader in test_loaders.values()])
train_h_max_counts, test_h_max_counts= dh.count_from_loaders(train_loaders), dh.count_from_loaders(test_loaders)
print('Training:',json.dumps(train_h_max_counts, indent=4))
print('Testing:',json.dumps(test_h_max_counts, indent=4))

############################## define the optimizer #################################
optimizer = torch.optim.Adam(midrp_mlp_model.parameters(), lr=learning_rate)
if post_train_together_epoch_num > 0:
    optimizer_gnn = torch.optim.Adam(midrp_gnn_model.parameters(), lr=post_train_gnn_learning_rate)

torch_mse_loss = torch.nn.MSELoss()

####################### define the whole mdr predictor model ###########################
def mdr_predictor_model(midrp_gnn_model, midrp_mlp_model, data, N_forward_layers=1, gnn_grad_flag=False):
    if gnn_grad_flag:
        graph_embedding = midrp_gnn_model(data.x, data.edge_index, data.edge_attr, N_forward_layers=N_forward_layers, graph_embedding=True, batch=data.batch)
        return midrp_mlp_model(graph_embedding)

    with torch.no_grad():
        graph_embedding = midrp_gnn_model(data.x, data.edge_index, data.edge_attr, N_forward_layers=N_forward_layers, graph_embedding=True, batch=data.batch)
    return midrp_mlp_model(graph_embedding)

############################## train the model #################################
train_mse_loss_record, test_mse_loss_record = [], []

for epoch in range(epoch_num + post_train_together_epoch_num):
    midrp_mlp_model.train()
    if epoch >= epoch_num:
        midrp_gnn_model.train()

    train_mse_loss = 0
    remaining_counts = train_h_max_counts.copy()
    while sum(remaining_counts.values()) > 0:
        probabilities = [count / sum(remaining_counts.values()) for count in remaining_counts.values()]
        selected_h_max = random.choices(list(train_loaders.keys()), probabilities, k=1)[0]
        loader = train_loaders[selected_h_max]

        ########### train the model with the selected h_max ###########
        for data in loader:
            remaining_counts[selected_h_max] -= batch_size
            
            gnn_grad_flag = False if epoch < epoch_num else True
            pred = mdr_predictor_model(midrp_gnn_model, midrp_mlp_model, data, N_forward_layers=selected_h_max, gnn_grad_flag=gnn_grad_flag).squeeze()

            optimizer.zero_grad()
            loss = torch_mse_loss(pred, data.y_max)

            if epoch != 0:
                loss.backward()
                optimizer.step()
                if epoch >= epoch_num:
                    optimizer_gnn.step()

            ########### record the loss ###########
            train_mse_loss += loss.item()
            break # only train one batch for each h_max


    train_mse_loss_record.append(train_mse_loss / train_batch_num)

    midrp_mlp_model.eval()
    test_mse_loss = 0
    with torch.no_grad():
        for h_max, loader in test_loaders.items():
            for data in loader:
                pred = mdr_predictor_model(midrp_gnn_model, midrp_mlp_model, data, N_forward_layers=h_max, gnn_grad_flag=True).squeeze()
                test_mse_loss += torch_mse_loss(pred, data.y_max).item()
    test_mse_loss_record.append(test_mse_loss / test_batch_num)

################################## record and save the last epoch performance ##################################
final_train_mse_loss, final_test_mse_loss = train_mse_loss_record[-1], test_mse_loss_record[-1]
print(f"Final train MSE loss: {final_train_mse_loss:.4f}, Final test MSE loss: {final_test_mse_loss:.4f}")


################################## plot two figures ##################################
fig, ax = plt.subplots()
ax.semilogy(train_mse_loss_record, label='Train')
ax.semilogy(test_mse_loss_record, label='Test')
ax.set_xlabel('Epoch')
ax.set_ylabel('MSE Loss')
ax.set_title('MSE Loss')
ax.legend()
ax.grid()
dh.save_runtime_fig(keystring, fig, 'mse_loss')

################################## save the runtime params and model ##################################
param_name_list = [
    'midrp_GNN_checkpoint_keystring',
    'gravity_filter',
    'fire_filter',
    'comment',
    'dataset_name',
    'midrp_mlp_layers',
    'midrp_mlp_dropouts',
    'learning_rate',
    'batch_size',
    'epoch_num',
    'post_train_together_epoch_num',
    'post_train_gnn_learning_rate',
    'random_seed',
    'total_params',
    'midrp_mlp_name',
    'timestamp_str',
    'keystring',
    'script_name',
    'final_train_mse_loss',
    'final_test_mse_loss',
]

dh.save_runtime_params(keystring, locals(), param_name_list)

result_dict = {
    'train_mse_loss_record': train_mse_loss_record,
    'test_mse_loss_record': test_mse_loss_record,
}

dh.save_runtime_csv_results(keystring, result_dict)
dh.save_runtime_torch_model(keystring, midrp_mlp_model, 'midrp_mlp_model')
if post_train_together_epoch_num > 0:
    dh.save_runtime_torch_model(keystring, midrp_gnn_model, 'midrp_gnn_model')