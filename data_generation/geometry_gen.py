import pandas as pd
import os
import sys
import random
import numpy as np
import copy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# import torch
# from torch_geometric.data import Data
sys.path.append(os.path.dirname(sys.path[0]))
from utils import data_helper as dh

random_seed = 1
dataset_name = 'drs1'
# check if the folder exists, if not, create it, otherwise, raise error
data_folder = os.path.join(dh.get_datasets_dir(), dataset_name)
if not os.path.exists(data_folder):
    os.makedirs(data_folder)
    os.makedirs(os.path.join(data_folder, 'xlsx-data'))
    os.makedirs(os.path.join(data_folder, 'pt-data'))
    os.makedirs(os.path.join(data_folder, 'figs'))
else:
    raise FileExistsError(f"Folder {data_folder} already exists")

data_fig_folder = os.path.join(data_folder, 'figs')
data_xlsx_folder = os.path.join(data_folder, 'xlsx-data')
data_pt_folder = os.path.join(data_folder, 'pt-data')

save_fig = True
N_total_models = 3000
N_drop_times_per_model = 2

H_random_range = [2, 7]
W_random_range = [2, 7]
D_random_range = [2, 7]
height_range = [3000, 5000]
width_range = [3000, 5000]
depth_range = [3000, 5000]
element_drop_ratio = 0.08

model_idx = 0
while model_idx < N_total_models:
    num_H = random.randint(H_random_range[0], H_random_range[1]) # number of levels (height), z-axis
    num_W = random.randint(W_random_range[0], W_random_range[1]) # number of rows (width), x-axis
    num_D = random.randint(D_random_range[0], D_random_range[1]) # number of columns (depth), y-axis
    # (num_H * num_W * num_D) is the number of rooms

    height_mm = random.randint(height_range[0], height_range[1]) # height per level, z-axis
    width_mm = random.randint(width_range[0], width_range[1]) # width per row, x-axis
    depth_mm = random.randint(depth_range[0], depth_range[1]) # depth per column, y-axis

    num_nodes = (num_H + 1) * (num_W + 1) * (num_D + 1)
    num_beam_elements = num_H * ( (num_W + 1) * num_D +  num_W * (num_D + 1) )
    num_column_elements = num_H * (num_W + 1) * (num_D + 1)

    w_vals, d_vals, h_vals = np.meshgrid(np.arange(num_W + 1), np.arange(num_D + 1), np.arange(num_H + 1), indexing='ij')
    node_coords = np.stack([w_vals * width_mm, d_vals * depth_mm, h_vals * height_mm], axis=-1)
    node_numbers = np.arange(num_nodes).reshape(num_W + 1, num_D + 1, num_H + 1) + 1

    df_nodes = pd.DataFrame(dict(
        node_number = node_numbers.flatten(),
        x = node_coords[..., 0].flatten(),
        y = node_coords[..., 1].flatten(),
        z = node_coords[..., 2].flatten(),
        w = w_vals.flatten(),
        d = d_vals.flatten(),
        h = h_vals.flatten(),
    ))

    # Generate beam elements along W direction (connecting along width)
    start_nodes_w = node_numbers[:-1, :, 1:].flatten()
    end_nodes_w = node_numbers[1:, :, 1:].flatten()

    # Generate beam elements along D direction (connecting along depth)
    start_nodes_d = node_numbers[:, :-1, 1:].flatten()
    end_nodes_d = node_numbers[:, 1:, 1:].flatten()

    beam_elements = np.vstack([
        np.stack([start_nodes_w, end_nodes_w, ['x'] * len(start_nodes_w)], axis=-1),
        np.stack([start_nodes_d, end_nodes_d, ['y'] * len(start_nodes_d)], axis=-1)
    ])
    df_beam_elements = pd.DataFrame(beam_elements, columns=['start_node', 'end_node', 'direction'])
    df_beam_elements['type'] = 'beam'

    # Generate column elements along H direction (connecting along height)
    start_nodes_h = node_numbers[:, :, :-1].flatten()
    end_nodes_h = node_numbers[:, :, 1:].flatten()
    column_elements = np.stack([start_nodes_h, end_nodes_h, ['z'] * len(start_nodes_h)], axis=-1)
    df_column_elements = pd.DataFrame(column_elements, columns=['start_node', 'end_node', 'direction'])
    df_column_elements['type'] = 'column'

    df_elements = pd.concat([df_beam_elements, df_column_elements], axis=0)
    df_elements.reset_index(drop=True, inplace=True)
    df_elements['element_number'] = np.arange(len(df_elements)) + 1
    df_elements['start_node'] = df_elements['start_node'].astype(int)
    df_elements['end_node'] = df_elements['end_node'].astype(int)
    

    for drop_idx in range(N_drop_times_per_model):
        model_idx += 1
        random_seed += 1
        while True:
            # first, randomly remove some elements
            df_elements_new = copy.deepcopy(df_elements.sample(frac=(1-element_drop_ratio), random_state=random_seed).reset_index(drop=True))
            df_elements_new.sort_values('element_number', inplace=True)
            df_elements_new.reset_index(drop=True, inplace=True)

            # then, remove the nodes that are not connected by any elements
            nodes_in_use = np.unique(df_elements_new[['start_node', 'end_node']].values.flatten())
            df_nodes_new = copy.deepcopy(df_nodes[df_nodes['node_number'].isin(nodes_in_use)].reset_index(drop=True))

            # also, remove the nodes that are only connected by one element, except the bottom level nodes
            node_connections = df_elements_new[['start_node', 'end_node']].values.flatten()
            connection_counts = pd.Series(node_connections).value_counts()
            nodes_to_remove = connection_counts[connection_counts == 1].index
            bottom_level_nodes = df_nodes['node_number'][df_nodes['h'] == 0].values
            nodes_to_remove = [node for node in nodes_to_remove if node not in bottom_level_nodes]
            df_nodes_new = df_nodes_new[~df_nodes_new['node_number'].isin(nodes_to_remove)].reset_index(drop=True)
            
            # finally, remove the elements that are connected to the removed nodes
            edges_to_remove = df_elements_new[(df_elements_new['start_node'].isin(nodes_to_remove)) | (df_elements_new['end_node'].isin(nodes_to_remove))]
            df_elements_new = df_elements_new[~df_elements_new['element_number'].isin(edges_to_remove['element_number'])].reset_index(drop=True)

            ## there should be at least 2 nodes in the bottom level
            if df_nodes[df_nodes['h'] == 0].shape[0] > 1:
                print('generated a model with {} nodes and {} elements'.format(df_nodes_new.shape[0], df_elements_new.shape[0]))
                break
            print('generate failed, retrying...')
            random_seed += 1
        if save_fig:
            fig, ax = dh.visualize_grid(df_nodes_new, df_elements_new)
            ax.set_title('{}*{} width, {}*{} depth, {}*{} height'.format(num_W, width_mm, num_D, depth_mm, num_H, height_mm))
            # fig_filename should be like m0001.png
            fig_filename = f'm{model_idx:04d}.png'
            fig_filepath = os.path.join(data_fig_folder, fig_filename)
            fig.savefig(fig_filepath)
            plt.close()
        
        data_filename = f'm{model_idx:04d}.xlsx'
        data_xlsx_filepath = os.path.join(data_xlsx_folder, data_filename)
        with pd.ExcelWriter(data_xlsx_filepath) as writer:
            df_nodes_new.to_excel(writer, sheet_name='Node_Coordinate', index=False)
            df_elements_new.to_excel(writer, sheet_name='Element', index=False)
        print(f"Model {model_idx} saved to {data_xlsx_filepath}")

