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
dataset_name='drs3v6'
gravity_filter = 1.0
fire_filter = 100.0

gnn_class = gm.CustomGNN1

midrp_mlp_layers = [128, 1]
midrp_mlp_dropouts = 0.0
learning_rate = 0.001
batch_size = 128
epoch_num = 200

random_seed = 0
train_ratio = 0.8
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


############################### define the midr mlp model #################################
midrp_mlp_model = gm.MIDRPredictor1(layers=midrp_mlp_layers, dropouts=midrp_mlp_dropouts).to(device)
mlp_total_params = sum(p.numel() for p in midrp_mlp_model.parameters())
midrp_mlp_name = str(type(midrp_mlp_model))

############################## set the random seed #################################
timestamp_str = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
keystring = timestamp_str + ''.join(random.choices(string.ascii_letters, k=6))

torch.manual_seed(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)
############################### load the dataset #################################
train_dataset_list, test_dataset_list = dh.load_gf_train_test_normalized_labeled_dataset(dataset_name,gravity_filter, fire_filter, device)
print(f"Train dataset size: {len(train_dataset_list)}, Test dataset size: {len(test_dataset_list)}")

train_loaders = dh.create_loaders(train_dataset_list, batch_size, shuffle=True)
test_loaders = dh.create_loaders(test_dataset_list, batch_size, shuffle=False)
train_batch_num, test_batch_num = sum([len(loader) for loader in train_loaders.values()]), sum([len(loader) for loader in test_loaders.values()])
train_h_max_counts, test_h_max_counts= dh.count_from_loaders(train_loaders), dh.count_from_loaders(test_loaders)
print('Training:',json.dumps(train_h_max_counts, indent=4))
print('Testing:',json.dumps(test_h_max_counts, indent=4))

############################## define the midr gnn model #################################
node_attr_dim = train_dataset_list[0].x.shape[1]
edge_attr_dim = train_dataset_list[0].edge_attr.shape[1]
gnn_params.update(dict(
    node_attr_dim=node_attr_dim,
    edge_attr_dim=edge_attr_dim,
))
midrp_gnn_model = gnn_class(**gnn_params).to(device)
gnn_model_name = str(type(midrp_gnn_model))

gnn_total_params = sum(p.numel() for p in midrp_gnn_model.parameters())
print(f"mlp_total_params: {mlp_total_params}, gnn_total_params: {gnn_total_params}")
############################## define the optimizer #################################
mlp_optimizer = torch.optim.Adam(midrp_mlp_model.parameters(), lr=learning_rate)
gnn_optimizer = torch.optim.Adam(midrp_gnn_model.parameters(), lr=learning_rate)
torch_mse_loss = torch.nn.MSELoss()
torch_mae_loss = torch.nn.L1Loss()

####################### define the whole midr predictor model ###########################
def midr_predictor_model(midrp_gnn_model, midrp_mlp_model, data, N_forward_layers=1, gnn_grad_flag=False):
    if gnn_grad_flag:
        graph_embedding = midrp_gnn_model(data.x, data.edge_index, data.edge_attr, N_forward_layers=N_forward_layers, graph_embedding=True, batch=data.batch, embedding_position=0)
        return midrp_mlp_model(graph_embedding)

    with torch.no_grad():
        graph_embedding = midrp_gnn_model(data.x, data.edge_index, data.edge_attr, N_forward_layers=N_forward_layers, graph_embedding=True, batch=data.batch, embedding_position=0)
    return midrp_mlp_model(graph_embedding)

############################## train the model #################################
train_mse_loss_record, test_mse_loss_record = [], []
train_mae_loss_record, test_mae_loss_record = [], []

for epoch in range(epoch_num):
    midrp_mlp_model.train()
    train_mse_loss, train_mae_loss = 0, 0
    remaining_counts = train_h_max_counts.copy()
    while sum(remaining_counts.values()) > 0:
        probabilities = [count / sum(remaining_counts.values()) for count in remaining_counts.values()]
        selected_h_max = random.choices(list(train_loaders.keys()), probabilities, k=1)[0]
        loader = train_loaders[selected_h_max]

        ########### train the model with the selected h_max ###########
        for data in loader:
            remaining_counts[selected_h_max] -= batch_size

            pred = midr_predictor_model(midrp_gnn_model, midrp_mlp_model, data, N_forward_layers=selected_h_max, gnn_grad_flag=True).squeeze()

            mlp_optimizer.zero_grad()
            gnn_optimizer.zero_grad()
            loss = torch_mse_loss(pred, data.y_max)
            if epoch != 0:
                loss.backward()
                mlp_optimizer.step()
                gnn_optimizer.step()

            ########### record the loss ###########
            train_mse_loss += loss.item()

            with torch.no_grad():
                train_mae_loss += torch_mae_loss(pred, data.y_max).item()
            break # only train one batch for each h_max

    train_mse_loss_record.append(train_mse_loss / train_batch_num)
    train_mae_loss_record.append(train_mae_loss / train_batch_num)

    midrp_mlp_model.eval()
    test_mse_loss, test_mae_loss = 0, 0
    with torch.no_grad():
        for h_max, loader in test_loaders.items():
            for data in loader:
                pred = midr_predictor_model(midrp_gnn_model, midrp_mlp_model, data, N_forward_layers=h_max, gnn_grad_flag=False).squeeze()
                test_mse_loss += torch_mse_loss(pred, data.y_max).item()
                test_mae_loss += torch_mae_loss(pred, data.y_max).item()
                
    test_mse_loss_record.append(test_mse_loss / test_batch_num)
    test_mae_loss_record.append(test_mae_loss / test_batch_num)

    print(f"Epoch {epoch+1}/{epoch_num}, Train MSE loss: {train_mse_loss:.4f}, Test MSE loss: {test_mse_loss:.4f}")
    print(f"Epoch {epoch+1}/{epoch_num}, Train MAE loss: {train_mae_loss:.4f}, Test MAE loss: {test_mae_loss:.4f}")

################################## record and save the last epoch performance ##################################
final_train_mse_loss, final_test_mse_loss = train_mse_loss_record[-1], test_mse_loss_record[-1]
final_train_mae_loss, final_test_mae_loss = train_mae_loss_record[-1], test_mae_loss_record[-1]
print(f"Final train MSE loss: {final_train_mse_loss:.4f}, Final test MSE loss: {final_test_mse_loss:.4f}")
print(f"Final train MAE loss: {final_train_mae_loss:.4f}, Final test MAE loss: {final_test_mae_loss:.4f}")


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

fig, ax = plt.subplots()
ax.semilogy(train_mae_loss_record, label='Train')
ax.semilogy(test_mae_loss_record, label='Test')
ax.set_xlabel('Epoch')
ax.set_ylabel('MAE Loss')
ax.set_title('MAE Loss')
ax.legend()
ax.grid()
dh.save_runtime_fig(keystring, fig, 'mae_loss')

result_dict = {
    'train_mse_loss_record': train_mse_loss_record,
    'test_mse_loss_record': test_mse_loss_record,
    'train_mae_loss_record': train_mae_loss_record,
    'test_mae_loss_record': test_mae_loss_record,
}

dh.save_runtime_csv_results(keystring, result_dict)
dh.save_runtime_torch_model(keystring, midrp_mlp_model, 'midrp_mlp_model')
dh.save_runtime_torch_model(keystring, midrp_gnn_model, 'gnn_model')