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

######################### tunaable settings #########################
dataset_name = 'drs3v6'
gravity_filter=1.0
fire_filter=100.0
gnn_class = gm.CustomGNN1

random_seed = 0

loss_weight_scale = 1
learning_rate = 1e-3
epoch_num = 200
batch_size = 128
mask_nonzero = False
gnn_params = dict(
    node_attr_dim=0, # TO be determined later
    edge_attr_dim=0, # TO be determined later
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

######## place holder variables #################
total_params, gnn_model_name = None, None

##################### backup the script and set the device #####################
timestamp_str = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
keystring = timestamp_str + ''.join(random.choices(string.ascii_letters, k=6))
script_name = os.path.basename(__file__)
dh.backup_script(keystring, __file__)

torch.manual_seed(random_seed), np.random.seed(random_seed), random.seed(random_seed)
############################## data pre-processing (normaliaztion, train-test split) ##############################

train_dataset_list, test_dataset_list = dh.load_gf_train_test_normalized_dataset(dataset_name, gravity_filter, fire_filter, device)

train_loaders = dh.create_loaders(train_dataset_list, batch_size, shuffle=True)
test_loaders = dh.create_loaders(test_dataset_list, batch_size, shuffle=False)
train_batch_num, test_batch_num = sum([len(loader) for loader in train_loaders.values()]), sum([len(loader) for loader in test_loaders.values()])
train_h_max_counts, test_h_max_counts= dh.count_from_loaders(train_loaders), dh.count_from_loaders(test_loaders)
print('Training:',json.dumps(train_h_max_counts, indent=4))
print('Testing:',json.dumps(test_h_max_counts, indent=4))

############################# define GNN model and optimizer ##########################################
node_attr_dim = train_dataset_list[0].x.shape[1]
edge_attr_dim = train_dataset_list[0].edge_attr.shape[1]
gnn_params.update(dict(
    node_attr_dim=node_attr_dim,
    edge_attr_dim=edge_attr_dim,
))
gnn_model = gnn_class(**gnn_params).to(device)
gnn_model_name = str(type(gnn_model))
total_params = sum(p.numel() for p in gnn_model.parameters() if p.requires_grad)
print(f"GNN model: {gnn_model_name}, Total number of trainable parameters: {total_params}")
print(f"GNN model params: {json.dumps(gnn_params, indent=4)}")
print(f"Loss method: {loss_method}")

############################# define optimizer and loss function ##########################################
optimizer = torch.optim.Adam(gnn_model.parameters(), lr=learning_rate)
torch_mse_loss = torch.nn.MSELoss()
torch_mse_loss_nored = torch.nn.MSELoss(reduction='none')
def calculate_loss(pred, gt, gt_origin=None,  mask_nonzero=True):
    if mask_nonzero:
        nonzero_mask = gt_origin > 1e-5
        return torch_mse_loss(pred[nonzero_mask], gt[nonzero_mask])
    else:
        return torch_mse_loss(pred, gt)

########################### define records for training and testing ########################################
train_mse_loss_record, test_mse_loss_record = [], []

####################################### training the model ##################################################
for epoch in range(epoch_num):
    gnn_model.train()
    train_mse_loss = 0
    remaining_counts = train_h_max_counts.copy()
    while sum(remaining_counts.values()) > 0:
        ################# randomly select a h_max group to train #####################
        probabilities = [count / sum(remaining_counts.values()) for count in remaining_counts.values()]
        selected_h_max = random.choices(list(train_loaders.keys()), probabilities, k=1)[0]
        loader = train_loaders[selected_h_max]
        for data in loader:
            remaining_counts[selected_h_max] -= batch_size

            optimizer.zero_grad()
            out = gnn_model(data.x, data.edge_index, data.edge_attr, N_forward_layers=selected_h_max)

            loss = calculate_loss(out, data.y, data.y, mask_nonzero=mask_nonzero)
            
            loss.backward()
            optimizer.step()

            train_mse_loss += loss.item()
            break ### only one batch for now
    
    train_mse_loss_record.append(train_mse_loss / train_batch_num)
    ################################ evaluate the model with test dataset ############################################
    test_mse_loss = 0
    gnn_model.eval()
    with torch.no_grad():
        for h_max, loader in test_loaders.items():
            for data in loader:
                data = data.to(device)
                out = gnn_model(data.x, data.edge_index, data.edge_attr, N_forward_layers=h_max)

                ## calculate all the losses
                test_mse_loss += calculate_loss(out, data.y, data.y, mask_nonzero=mask_nonzero).item()

    test_mse_loss_record.append(test_mse_loss / test_batch_num)
    print("---" * 16)

################################  record and save the last epoch performance #################################
final_train_mse_loss, final_test_mse_loss = train_mse_loss_record[-1], test_mse_loss_record[-1]
print(f"Final train mse loss: {final_train_mse_loss:.6f}, Final test mse loss: {final_test_mse_loss:.6f}")

############### normalized mse and log2w loss in one figure ######################
fig, ax = plt.subplots(1,1, figsize=(8,6))
ax.semilogy(train_mse_loss_record, label='train mse loss')
ax.semilogy(test_mse_loss_record, label='test mse loss', linestyle='-.')
ax.legend()
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.grid()
ax.set_title(f"Train and Test Loss, {gnn_model_name},\n{dataset_name}")
dh.save_runtime_fig(keystring, fig, 'normalized_loss_record')


param_name_list = [
    'dataset_name',
    'random_seed',
    'train_ratio',
    'learning_rate',
    'epoch_num',
    'batch_size',
    'mask_nonzero',
    'gnn_params',
    'loss_method',
    'total_params',
    'gnn_model_name',
    'timestamp_str',
    'keystring',
    'script_name',
    'gravity_filter',
    'fire_filter',
    'final_train_mse_loss',
    'final_test_mse_loss',
]
dh.save_runtime_params(keystring, locals(), param_name_list)

result_dict = {
    'train_mse_loss_record': train_mse_loss_record,
    'test_mse_loss_record': test_mse_loss_record,
}

dh.save_runtime_csv_results(keystring, result_dict)
dh.save_runtime_torch_model(keystring, gnn_model, 'gnn_model')


