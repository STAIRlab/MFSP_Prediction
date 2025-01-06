import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from collections import defaultdict
from torch_geometric.loader import DataLoader

def get_base_dir():
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'config.json'))
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")
    with open(config_path, 'r') as config_file:
        config = json.load(config_file)
    return config.get('basedir')

def get_datasets_dir():
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'config.json'))
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")
    with open(config_path, 'r') as config_file:
        config = json.load(config_file)
    
    basedir = config.get('basedir')
    datadir = config.get('dataset_relative_basedir')
    return os.path.join(basedir, datadir)

def get_runtime_data_dir():
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'config.json'))
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")
    with open(config_path, 'r') as config_file:
        config = json.load(config_file)

    basedir = config.get('basedir')
    datadir = config.get('data_runtime_relative_base_dir')
    return os.path.join(basedir, datadir)


def visualize_grid(df_nodes, df_elements):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制节点
    ax.scatter(df_nodes['x'], df_nodes['y'], df_nodes['z'], c='r', marker='o', s=50, label='Nodes')

    # 绘制梁和柱
    for _, element in df_elements.iterrows():
        start_node = df_nodes[df_nodes['node_number'] == element['start_node']].iloc[0]
        end_node = df_nodes[df_nodes['node_number'] == element['end_node']].iloc[0]
        
        # 根据起始节点和终止节点的坐标绘制线段
        ax.plot(
            [start_node['x'], end_node['x']],
            [start_node['y'], end_node['y']],
            [start_node['z'], end_node['z']],
            label=f"{element['type']} {element['direction']}",
            color='b' if element['type'] == 'beam' else 'g'
        )
    
    # 设置轴标签
    ax.set_xlabel('X (Width)')
    ax.set_ylabel('Y (Depth)')
    ax.set_zlabel('Z (Height)')
    return fig, ax

def count_from_loaders(loaders):
    h_max_counts = {}
    for h_max, loader in loaders.items():
        h_max_counts[h_max] = len(loader.dataset)
    return h_max_counts

def create_loaders(dataset, batch_size, shuffle):
    h_max_groups = defaultdict(list)
    for data in dataset:
        h_max_groups[data.h_max.item()].append(data)
    loaders = {h_max: DataLoader(group, batch_size=batch_size, shuffle=shuffle) 
               for h_max, group in h_max_groups.items()}
    return loaders
    
def create_loaders(dataset, batch_size, shuffle):
    h_max_groups = defaultdict(list)
    for data in dataset:
        h_max_groups[data.h_max.item()].append(data)
    loaders = {h_max: DataLoader(group, batch_size=batch_size, shuffle=shuffle) 
               for h_max, group in h_max_groups.items()}
    return loaders

def load_gf_train_test_normalized_labeled_dataset(dataset_name='drs3v6', gravity_filter_threshold=1.00, fire_filter_threshold=100, device='cpu', random_split_dataset=False, test_only=False):
    "gf stands for gravity and fire"
    data_dir = os.path.join(get_datasets_dir(), 'defective_regular_structure3_var6', 'pt-data').replace("\\", "/")
    train_data_filename = f"{dataset_name}_filtered_normalized_train_g{gravity_filter_threshold:.2f}_gf{fire_filter_threshold:.2f}.pt"
    test_data_filename = f"{dataset_name}_filtered_normalized_test_g{gravity_filter_threshold:.2f}_gf{fire_filter_threshold:.2f}.pt"
    if test_only:
        return torch.load(os.path.join(data_dir, test_data_filename), map_location=device)
    return torch.load(os.path.join(data_dir, train_data_filename), map_location=device), torch.load(os.path.join(data_dir, test_data_filename), map_location=device)
    

def save_runtime_fig(keystring, fig, fig_name, fig_format='png'):
    runtime_data_dir = os.path.join(get_runtime_data_dir(), keystring)
    if os.path.exists(runtime_data_dir) == False:
        os.mkdir(runtime_data_dir)
    fig.savefig(os.path.join(runtime_data_dir, fig_name + '.' + fig_format))
    print(f"Figure saved as {os.path.join(runtime_data_dir, fig_name + '.' + fig_format)}")

def save_runtime_csv_results(keystring, name_result_dict):
    runtime_data_dir = os.path.join(get_runtime_data_dir(), keystring)
    if os.path.exists(runtime_data_dir) == False:
        os.mkdir(runtime_data_dir)
    for name, result in name_result_dict.items():
        # if is list, make it into ndarray, then save,
        # if is ndarray or dataframe, save directly 
        if isinstance(result, list):
            result = np.array(result)

        if isinstance(result, np.ndarray):
            np.savetxt(os.path.join(runtime_data_dir, name + '.csv'), result, delimiter=',')
        elif isinstance(result, pd.DataFrame):
            result.to_csv(os.path.join(runtime_data_dir, name + '.csv'))
        else:
            print(f"Result {name} is not a valid type for saving.")            
    print(f"Results saved to {runtime_data_dir}")

def save_runtime_torch_model(keystring, torch_model, model_name):
    runtime_data_dir = os.path.join(get_runtime_data_dir(), keystring)
    if os.path.exists(runtime_data_dir) == False:
        os.mkdir(runtime_data_dir)
    torch.save(torch_model.state_dict(), os.path.join(runtime_data_dir, model_name + '.pt'))
    print(f"Model saved to {runtime_data_dir}")


def backup_script(keystring, script_filepath):
    backup_folder = os.path.join(get_runtime_data_dir(), keystring)
    if os.path.exists(backup_folder) == False:
        os.mkdir(backup_folder)

    current_script_path = Path(script_filepath).resolve()
    script_name = current_script_path.stem  # 不带扩展名的文件名
    script_extension = current_script_path.suffix  # 文件扩展名
    
    backup_file_name = f"{script_name}_{keystring}{script_extension}"
    backup_file_path = Path(backup_folder) / backup_file_name
    
    shutil.copy(current_script_path, backup_file_path)
    print(f"Backup created at: {backup_file_path}")


def check_and_load_midrp_processed_train_test_dataset(mdrp_checkpoint_keystring, dataset_name='drs4v2', gravity_filter_threshold=1.00, device='cpu'):
    data_dir = os.path.join(get_datasets_dir(), dataset_name, 'pt-data').replace("\\", "/")
    train_data_filename = f"{dataset_name}_g{gravity_filter_threshold:.2f}_train_{mdrp_checkpoint_keystring}.pt"
    test_data_filename = f"{dataset_name}_g{gravity_filter_threshold:.2f}_test_{mdrp_checkpoint_keystring}.pt"
    if os.path.exists(os.path.join(data_dir, train_data_filename)) and os.path.exists(os.path.join(data_dir, test_data_filename)):
        return torch.load(os.path.join(data_dir, train_data_filename), map_location=device), torch.load(os.path.join(data_dir, test_data_filename), map_location=device)
    else:
        return None, None

def load_unlabeled_train_test_normalized_dataset(dataset_name='drs4v2', gravity_filter_threshold=1.00, device='cpu'):
    data_dir = os.path.join(get_datasets_dir(), dataset_name, 'pt-data').replace("\\", "/")
    train_data_filename = f"{dataset_name}_g{gravity_filter_threshold:.2f}_train.pt"
    test_data_filename = f"{dataset_name}_g{gravity_filter_threshold:.2f}_test.pt"
    return torch.load(os.path.join(data_dir, train_data_filename), map_location=device), torch.load(os.path.join(data_dir, test_data_filename), map_location=device)
    

def save_midrp_processed_train_test_dataset(train_dataset_list, test_dataset_list, mdrp_checkpoint_keystring, dataset_name='drs4v2', gravity_filter_threshold=1.00):
    data_dir = os.path.join(get_datasets_dir(), dataset_name, 'pt-data').replace("\\", "/")
    train_data_filename = f"{dataset_name}_g{gravity_filter_threshold:.2f}_train_{mdrp_checkpoint_keystring}.pt"
    test_data_filename = f"{dataset_name}_g{gravity_filter_threshold:.2f}_test_{mdrp_checkpoint_keystring}.pt"
    torch.save(train_dataset_list, os.path.join(data_dir, train_data_filename))
    torch.save(test_dataset_list, os.path.join(data_dir, test_data_filename))
    print(f"Train and test data saved to {os.path.join(data_dir, train_data_filename)} and {os.path.join(data_dir, test_data_filename)}")


def check_and_load_midrp_processed_full_info_train_test_dataset(mdrp_checkpoint_keystring, dataset_name='drs4v2', gravity_filter_threshold=1.00, device='cpu'):
    data_dir = os.path.join(get_datasets_dir(), dataset_name, 'pt-data').replace("\\", "/")
    train_data_filename = f"{dataset_name}_g{gravity_filter_threshold:.2f}_train_full_info_{mdrp_checkpoint_keystring}.pt"
    test_data_filename = f"{dataset_name}_g{gravity_filter_threshold:.2f}_test_full_info_{mdrp_checkpoint_keystring}.pt"
    if os.path.exists(os.path.join(data_dir, train_data_filename)) and os.path.exists(os.path.join(data_dir, test_data_filename)):
        return torch.load(os.path.join(data_dir, train_data_filename), map_location=device), torch.load(os.path.join(data_dir, test_data_filename), map_location=device)
    else:
        return None, None



def save_midrp_processed_full_info_train_test_dataset(train_dataset_list, test_dataset_list, mdrp_checkpoint_keystring, dataset_name='drs4v2', gravity_filter_threshold=1.00):
    data_dir = os.path.join(get_datasets_dir(), dataset_name, 'pt-data').replace("\\", "/")
    train_data_filename = f"{dataset_name}_g{gravity_filter_threshold:.2f}_train_full_info_{mdrp_checkpoint_keystring}.pt"
    test_data_filename = f"{dataset_name}_g{gravity_filter_threshold:.2f}_test_full_info_{mdrp_checkpoint_keystring}.pt"
    torch.save(train_dataset_list, os.path.join(data_dir, train_data_filename))
    torch.save(test_dataset_list, os.path.join(data_dir, test_data_filename))
    print(f"Train and test data saved to {os.path.join(data_dir, train_data_filename)} and {os.path.join(data_dir, test_data_filename)}")

