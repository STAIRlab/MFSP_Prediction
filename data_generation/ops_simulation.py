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

parser = argparse.ArgumentParser()
parser.add_argument('--start_model_number', type=int, default=1)
parser.add_argument('-s', '--start', type=int, default=1, help="Start model number")
parser.add_argument('-e', '--end', type=int, default=1, help="End model number")
args = parser.parse_args()

start_model_number = args.start
end_model_number = args.end
print(f"Start model number: {start_model_number}, End model number: {end_model_number}")

fire_duration = 60 # minutes

random_seed = 1207
origin_dataset_name = 'drs1'
variant_number = 1

N_mat = 5
gravity_param_range_column = [-1, -0.5]
gravity_param_range_edge_beam = [-4.5, -1.5]
gravity_param_range_inner_beam = [-7.5, -3]
mat_dict = {tag: dict(
    Es = int(210000 * np.random.uniform(0.8, 1.2)),
    Fy = int(250 * np.random.uniform(0.8, 1.2)),
    b = 0.001 * np.random.uniform(0.8, 1.2),
    ) for tag in range(1, N_mat+1)}


init_temp_incr_rate_up = 5
init_temp_incr_rate_down = 2

t_up = 16
t_h = 18
t_down = 30
r_up = 0.95
r_down = 0.97

alpha_up_in_mm = 18e3
alpha_h_in_mm = 10e3
alpha_down_in_mm = 5e3

N_temp_samples = 20

N_fire_steps = 100
fire_stepsize = 0.01

T_0 = 0
N_fires_per_model = 30


new_dataset_name = f"{origin_dataset_name}_var{variant_number}"

origin_data_folder = os.path.join(dh.get_datasets_dir(), origin_dataset_name)
origin_data_xlsx_folder = os.path.join(origin_data_folder, 'xlsx-data').replace("\\", "/")
opensees_scripts_folder = os.path.join(dh.get_base_dir(), 'opensees_scripts', new_dataset_name).replace("\\", "/")
new_dataset_xlsx_folder = os.path.join(dh.get_datasets_dir(), new_dataset_name, 'xlsx-data').replace("\\", "/")
new_dataset_sim_result_folder = os.path.join(dh.get_datasets_dir(), new_dataset_name, 'sim-result').replace("\\", "/")
new_dataset_pt_folder = os.path.join(dh.get_datasets_dir(), new_dataset_name, 'pt-data').replace("\\", "/")

if not os.path.exists(origin_data_folder):
    raise FileNotFoundError(f"Folder {origin_data_folder} does not exist")

if os.path.exists(os.path.join(dh.get_datasets_dir(), new_dataset_name)):
    print(f"Warning: Folder {new_dataset_name} already exists. It will be overwritten.")

# if not exist, create the folder
if not os.path.exists(new_dataset_xlsx_folder):
    os.makedirs(new_dataset_xlsx_folder)
if not os.path.exists(new_dataset_sim_result_folder):
    os.makedirs(new_dataset_sim_result_folder)
if not os.path.exists(opensees_scripts_folder):
    os.makedirs(opensees_scripts_folder)
if not os.path.exists(new_dataset_pt_folder):
    os.makedirs(new_dataset_pt_folder)

tcl_content_prefix = f"""
wipe all;
model basic -ndm 3 -ndf 6;
set script_path [file dirname [info script]]
cd $script_path
"""

beam_transf_tag = 1
column_transf_tag = 2
tcl_transf_def = f"""
set beamTransfTag {beam_transf_tag}; 
geomTransf Linear $beamTransfTag 0 0 1;
set columnTransfTag {column_transf_tag};
geomTransf Linear $columnTransfTag -1 0 0;
"""

tcl_gravity_analysis_def = """
set Tol 1.0e-8;
set gravity_Nstep 10;

constraints Plain;
numberer RCM;
system UmfPack;
test NormDispIncr $Tol $gravity_Nstep 500;
algorithm ExpressNewton 20;
# algorithm Newton;
integrator LoadControl 0.1;
analysis Static;
analyze $gravity_Nstep;
puts "Gravity analysis is done!";

loadConst -time 0.0;
"""

tcl_fire_analysis_def = f"""
puts "Fire Analysis start"

set fire_Nstep {N_fire_steps};
set Factor {fire_stepsize};
# set Factor [expr 1.0/$fire_Nstep]; 	# first load increment;

constraints Plain;					# how it handles boundary conditions
numberer RCM;						# renumber dof's to minimize band-width (optimization)
system UmfPack;
test NormDispIncr 1e-8 1000;
algorithm ExpressNewton 20;
integrator LoadControl $Factor;	# determine the next time step for an analysis
analysis Static;			# define type of analysis static or transient
analyze $fire_Nstep;		# apply fire load

puts "Fire Analysis Done"
"""

if os.name == 'nt':
    # change the system from UmfPack to BandGeneral
    tcl_fire_analysis_def = tcl_gravity_analysis_def.replace('UmfPack', 'BandGeneral')
    print(tcl_fire_analysis_def)
elif os.name == 'posix':
    # leave it the UmfPack
    pass


def compute_drift_ratio(row, delta_h):
    if row['h'] == 0: 
        return 0
    if pd.notnull(row['final_disp_x_prev']) and pd.notnull(row['final_disp_y_prev']):
        delta_disp_x = row['final_disp_x'] - row['final_disp_x_prev']
        delta_disp_y = row['final_disp_y'] - row['final_disp_y_prev']
        interlayer_displacement = np.sqrt(delta_disp_x**2 + delta_disp_y**2)
        return interlayer_displacement / delta_h  * 100
    else:
        return row['h'] / (row['h'] * delta_h)  * 100

def compute_gravity_drift_ratio(row, delta_h):
    if row['h'] == 0: 
        return 0
    if pd.notnull(row['gravity_disp_x_prev']) and pd.notnull(row['gravity_disp_y_prev']):
        delta_disp_x = row['gravity_disp_x'] - row['gravity_disp_x_prev']
        delta_disp_y = row['gravity_disp_y'] - row['gravity_disp_y_prev']
        interlayer_displacement = np.sqrt(delta_disp_x**2 + delta_disp_y**2)
        return interlayer_displacement / delta_h  * 100
    else:
        return row['h'] / (row['h'] * delta_h)  * 100

def compress_by_model_number(folder, model_number):
    prefix = f"m{model_number:04d}"
    target_7z_file = os.path.join(folder, f"{prefix}.7z")
    ## if the target_7z_file exists, delete it
    if os.path.exists(target_7z_file):
        os.remove(target_7z_file)
    # 获取所有以 prefix 开头的文件和文件夹
    all_items = os.listdir(folder)
    items_to_compress = [
        os.path.join(folder, item)
        for item in all_items if item.startswith(prefix)
    ]
    
    if not items_to_compress:
        print(f"No files or folders found with prefix {prefix} in {folder}.")
        return
    
    # 构建 7zz 压缩命令
    compress_command = ["7zz", "a", target_7z_file] + items_to_compress
    
    try:
        # 执行压缩命令
        subprocess.run(compress_command, check=True)
        print(f"Successfully compressed items into {target_7z_file}.")
        
        # 删除源文件和文件夹
        for item in items_to_compress:
            if os.path.isfile(item):
                os.remove(item)
            elif os.path.isdir(item):
                shutil.rmtree(item) 
                # os.rmdir(item)  # 仅能删除空目录
        print(f"Deleted original files and folders with prefix {prefix}.")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred during compression: {e}")
    except Exception as e:
        print(f"Error during file removal: {e}")

########### generate the tcl_file and run the gravity analysis #####################
origin_xlsx_file_list = os.listdir(origin_data_xlsx_folder)
origin_xlsx_file_list.sort()
for filename in origin_xlsx_file_list:
    #### timing #####
    model_start_time = time.time()

    model_number = int(filename[1:5])
    if (model_number < start_model_number):
        print(f"Skip model {filename}")
        continue
    elif (model_number > end_model_number):
        print(f"End of the program.")
        break

    # read the origin data from xlsx
    if filename.endswith('.xlsx'):
        df_nodes = pd.read_excel(os.path.join(origin_data_xlsx_folder, filename), sheet_name='Node_Coordinate')
        df_elements = pd.read_excel(os.path.join(origin_data_xlsx_folder, filename), sheet_name='Element')
    else:
        continue

    df_elements['mat_tag'] = np.random.choice(list(mat_dict.keys()), size=len(df_elements))
    df_elements['Es'] = df_elements['mat_tag'].apply(lambda x: mat_dict[x]['Es'])
    df_elements['Fy'] = df_elements['mat_tag'].apply(lambda x: mat_dict[x]['Fy'])
    df_elements['b'] = df_elements['mat_tag'].apply(lambda x: mat_dict[x]['b'])

    node_dict = df_nodes.set_index('node_number')[['x', 'y', 'z', 'h']].T.to_dict()
    def calculate_length(row):
        start_coords = node_dict[row['start_node']]
        end_coords = node_dict[row['end_node']]
        length = np.sqrt((end_coords['x'] - start_coords['x'])**2 +
                    (end_coords['y'] - start_coords['y'])**2 +
                    (end_coords['z'] - start_coords['z'])**2)
        h = min(start_coords['h'], end_coords['h'])
        return pd.Series([length, h])
    df_elements[['length', 'h']] = df_elements.apply(calculate_length, axis=1)

    boundary_nodes = df_nodes[
        (df_nodes['w'].isin([df_nodes['w'].min(), df_nodes['w'].max()])) |
        (df_nodes['d'].isin([df_nodes['d'].min(), df_nodes['d'].max()]))
    ]['node_number'].tolist()

    def determine_beam_type(row):
        if row['type'] == 'column':
            return 'column'
        elif row['type'] == 'beam':
            # 检查 start_node 和 end_node 是否都在边界节点中
            if row['start_node'] in boundary_nodes and row['end_node'] in boundary_nodes:
                return 'edge_beam'
            else:
                return 'inner_beam'
        return None

    df_elements['beam_type'] = df_elements.apply(determine_beam_type, axis=1)

    ###################  Build the opensees model ############################
    ##### define the nodes and there boundary conditions #####################
    tcl_node_def = ""
    tcl_boundary_cond = ""
    for idx, row in df_nodes.iterrows():
        tcl_node_def += f"node {row['node_number']} {row['x']} {row['y']} {row['z']};\n"
        if row['h'] == 0:
            tcl_boundary_cond += f"fix {row['node_number']} 1 1 1 1 1 1;\n"

    ######################### define the material properties ####################
    tcl_mat_def = ""
    for mat_tag, mat_prop in mat_dict.items():
        tcl_mat_def += f"set matTag{mat_tag} {mat_tag};\nuniaxialMaterial Steel01Thermal $matTag{mat_tag} {mat_prop['Fy']} {mat_prop['Es']} {mat_prop['b']:.5f};\n"

    ######################### define the sections ###############################
    tcl_sec_def = ""
    for idx, row in mat_dict.items():
        tcl_sec_def += f"set secTag{idx} {idx};\nsection FiberThermal $secTag{idx} -GJ {row['Es']} {{\n    patch quad $matTag{idx} 2 2 -50 -50 50 -50 50 50 -50 50\n}}\n"

    ######################### define the elements ###############################
    tcl_element_def = ""
    for idx, row in df_elements.iterrows():
        if row['type'] == 'beam':
            tcl_element_def += f"element dispBeamColumnThermal {row['element_number']} {row['start_node']} {row['end_node']} 3 $secTag{row['mat_tag']} $beamTransfTag;\n"
        elif row['type'] == 'column':
            tcl_element_def += f"element dispBeamColumnThermal {row['element_number']} {row['start_node']} {row['end_node']} 3 $secTag{row['mat_tag']} $columnTransfTag;\n"
    
    
    ########################## define the recorder ##############################
    output_filename = f"{filename.split('.')[0]}_gravity.txt"
    output_filepath = os.path.join(new_dataset_sim_result_folder, output_filename).replace("\\", "/")
    # relative_output_filepath = os.path.relpath(output_filepath, opensees_scripts_folder).replace("\\", "/")
    tcl_recorder_def = "recorder Node -file " + output_filepath + " -time -node "
    for idx, row in df_nodes.iterrows():
        tcl_recorder_def += f"{row['node_number']} "
    tcl_recorder_def += "-dof 1 2 3 disp;\n"

    ########################## define the gravity load ###########################
    # according to the element type and beam type, assign the gravity parameter:
    df_elements['gravity_param'] = 0.0
    for idx, row in df_elements.iterrows():
        if row['type'] == 'column':
            df_elements.loc[idx, 'gravity_param'] = np.random.uniform(*gravity_param_range_column)
        elif row['type'] == 'beam':
            if row['beam_type'] == 'edge_beam':
                df_elements.loc[idx, 'gravity_param'] = np.random.uniform(*gravity_param_range_edge_beam)
            elif row['beam_type'] == 'inner_beam':
                df_elements.loc[idx, 'gravity_param'] = np.random.uniform(*gravity_param_range_inner_beam)

    tcl_column_gravity_load_def = "pattern Plain 1 Linear {\n"
    tcl_beam_gravity_load_def = "pattern Plain 2 Linear {\n"
    for idx, row in df_elements.iterrows():
        if row['type'] == 'column':
            tcl_column_gravity_load_def += f"    eleLoad -ele {row['element_number']} -type -beamUniform 0 0 {row['gravity_param']:.5f};\n"
        elif row['type'] == 'beam':
            tcl_beam_gravity_load_def += f"    eleLoad -ele {row['element_number']} -type -beamUniform 0 {row['gravity_param']:.5f} 0;\n"
    tcl_column_gravity_load_def += "}\n"
    tcl_beam_gravity_load_def += "}\n"

    tcl_gravity_content_all = ''.join([
        tcl_content_prefix,
        tcl_mat_def,
        tcl_sec_def,
        tcl_transf_def,
        tcl_node_def,
        tcl_boundary_cond,
        tcl_element_def,
        tcl_recorder_def,
        tcl_column_gravity_load_def,
        tcl_beam_gravity_load_def,
        tcl_gravity_analysis_def,
    ])

    ################### save evaluate the tcl content ###############################
    tcl_gravity_path = os.path.join(opensees_scripts_folder, f"{filename.split('.')[0]}_gravity.tcl")
    with open(tcl_gravity_path, 'w') as f:
        f.write(tcl_gravity_content_all)
        print(f"File {tcl_gravity_path} is written.")
    model = ops.Model()
    model.eval(tcl_gravity_content_all)
    gravity_displacements = {
        node: model.nodeDisp(node)[:3] for node in model.getNodeTags()
    }
    # for node in model.getNodeTags():
        # gravity_displacements.update({node: model.nodeDisp(node)[:3]})
    model.wipe()
    ######################### save the displacement to xlsx: df_nodes ############################
    df_nodes['gravity_disp_x'] = df_nodes['node_number'].apply(lambda x: gravity_displacements[x][0])
    df_nodes['gravity_disp_y'] = df_nodes['node_number'].apply(lambda x: gravity_displacements[x][1])
    df_nodes['gravity_disp_z'] = df_nodes['node_number'].apply(lambda x: gravity_displacements[x][2])


    ######################## Then about fires ############################################
    ##### some preparation for the fire analysis #########
    ##### edges do not change with fire points ##########
    def get_ele_midpoint(row):
        start_node = df_nodes.loc[df_nodes['node_number'] == row['start_node']]
        end_node = df_nodes.loc[df_nodes['node_number'] == row['end_node']]
        return pd.Series((start_node[['x', 'y', 'z']].values + end_node[['x', 'y', 'z']].values).squeeze() / 2)
    df_elements[['mid_x', 'mid_y', 'mid_z']] = df_elements.apply(get_ele_midpoint, axis=1)
    df_nodes['pyg_node_number'] = np.arange(0, len(df_nodes), dtype=int)
    node_map = {origin_node_number: pyg_node_number for origin_node_number, pyg_node_number in zip(df_nodes['node_number'], df_nodes['pyg_node_number'])}
    df_elements['start_node_mapped'] = df_elements['start_node'].map(node_map)
    df_elements['end_node_mapped'] = df_elements['end_node'].map(node_map)
    one_hot_direction = pd.get_dummies(df_elements['direction'], prefix='direction').astype(int)
    df_elements = pd.concat([df_elements, one_hot_direction], axis=1)

    ## double beams is used to make the beams undirected, only for .pt file
    edges_xy = df_elements[df_elements['direction'].isin(['x', 'y'])].copy()
    edges_z = df_elements[df_elements['direction'] == 'z'].copy()
    edges_xy_reversed = edges_xy.rename(columns={'start_node_mapped': 'end_node_mapped', 'end_node_mapped': 'start_node_mapped'})
    edges_xy = pd.concat([edges_xy, edges_xy_reversed], ignore_index=True)
    df_elements_double_beams = pd.concat([edges_xy, edges_z], ignore_index=True)

    edge_attr_names = ['Es', 'Fy', 'b', 'length', 'h', 'gravity_param', 'direction_x', 'direction_y', 'direction_z']
    edge_attrs_double_beams = torch.tensor(df_elements_double_beams[edge_attr_names].values, dtype=torch.float)
    edge_index = torch.tensor(df_elements[['start_node_mapped', 'end_node_mapped']].values.T, dtype=torch.long)
    edge_index_double_beams = torch.tensor(df_elements_double_beams[['start_node_mapped', 'end_node_mapped']].values.T, dtype=torch.long)

    ####### first define the room index of fires ###############
    W, D, H = df_nodes['w'].max(), df_nodes['d'].max(), df_nodes['h'].max()
    unit_width, unit_depth, unit_height = df_nodes['x'].max() / W, df_nodes['y'].max() / D, df_nodes['z'].max() / H
    N_rooms = H * W * D

    all_room_indices = [(w,d,h) for w in range(W) for d in range(D) for h in range(H)]
    if N_rooms >= N_fires_per_model:
        fire_room_indices = random.sample(all_room_indices, N_fires_per_model) # sample without replacement
    else:
        rest_room_indices = random.choices(all_room_indices, k=N_fires_per_model - N_rooms) # sample with replacement
        fire_room_indices = all_room_indices + rest_room_indices
    fire_coord_list = [
        (random.uniform(w*unit_width, (w+1)*unit_width), random.uniform(d*unit_depth, (d+1)*unit_depth), random.uniform(h*unit_height, (h+1)*unit_height)) for w,d,h in fire_room_indices
    ]

    valid_fire_number_list = []
    max_drift_ratio_list = []
    pt_graph_data_list = []

    for fire_idx in range(N_fires_per_model):
        fire_start_time = time.time()
        fire_number = fire_idx + 1
        ######### deal with the excel file ################
        modelfire_xlsx_filename = f"{filename.split('.')[0]}_fire{fire_number:04d}.xlsx"
        fire_coord = fire_coord_list[fire_idx]
        fire_room_idx_tuple = fire_room_indices[fire_idx]
        x0, y0, z0, w0, d0, h0 = fire_coord[0], fire_coord[1], fire_coord[2], fire_room_idx_tuple[0], fire_room_idx_tuple[1], fire_room_idx_tuple[2]
        df_fire = pd.DataFrame(dict(
            x = x0,
            y = y0,
            z = z0,
            w = w0,
            d = d0,
            h = h0,
        ),index=[0])


        def calculate_t1(row, x0,y0,z0,h0):
            x, y, z, h = row['mid_x'], row['mid_y'], row['mid_z'], int(row['h'])
            if h == h0:
                l = np.sqrt((x - x0)**2 + (y - y0)**2)
                return t_h * (1 - np.exp( -l / alpha_h_in_mm))
            elif h > h0: # spread upward
                l = np.sqrt((x - x0)**2 + (y - y0)**2 + (z - z0)**2)
                return t_up * ( 1 - np.power(r_up, h - h0)) / (1 - r_up) * (1 - np.exp( -l / alpha_up_in_mm))
            else: # spread downward
                l = np.sqrt((x - x0)**2 + (y - y0)**2 + (z - z0)**2)
                return t_down * ( 1 - np.power(r_down, h0 - h)) / (1 - r_down) * (1 - np.exp( -l / alpha_down_in_mm))
        df_elements['t1'] = df_elements.apply(calculate_t1, args=(x0,y0,z0,h0), axis=1)
        
        ele_temp_file_folder = os.path.join(opensees_scripts_folder, modelfire_xlsx_filename[:-5]).replace("\\", "/")
        if not os.path.exists(ele_temp_file_folder):
            os.mkdir(ele_temp_file_folder)
        
        for ele_idx, row in df_elements.iterrows():
            element_number = row['element_number']
            t1 = row['t1'] 
            
            ele_temp_filepath = os.path.join(ele_temp_file_folder, f"ele{element_number}_temp.txt")
            
            if row['h'] >= h0:
                init_temp_incr_rate = init_temp_incr_rate_up / (row['h'] - h0 + 1)
            else:
                init_temp_incr_rate = init_temp_incr_rate_down / (h0 - row['h'])

            if t1 < fire_duration:
                t_values = [0, t1]
                Temp_values = [T_0, T_0 + init_temp_incr_rate * t1]
                Temp_end = dh.iso_834_curve(fire_duration - t1, Temp_values[1])
                
                num_stage2_samples = N_temp_samples - 1
                Temp_stage2_values = np.linspace(Temp_values[1], Temp_end, num_stage2_samples)

                # for T in Temp_stage2_values[1:-1]:
                for T in Temp_stage2_values[1:]:
                    t = dh.iso_834_curve_inverse(T, Temp_values[1]) + t1
                    t_values.append(t)
                    Temp_values.append(T)
                # t_values.append(fire_duration)
                # Temp_values.append(Temp_end)

                # # for additional lines to garantee the output file is complete
                # t_values.append(fire_duration * N_fire_steps * fire_stepsize)
                # Temp_values.append(Temp_end)

                t_values = np.array(t_values) / fire_duration
                Temp_values = np.array(Temp_values)
            else: # t1 >= fire_duration
                t_values = np.linspace(0, fire_duration, N_temp_samples) / fire_duration
                Temp_values = np.linspace(T_0, init_temp_incr_rate * fire_duration, N_temp_samples)      
                # t_values = np.append(t_values, N_fire_steps * fire_stepsize)
                # Temp_values = np.append(Temp_values, Temp_values[-1])

            with open(ele_temp_filepath, 'w') as f:
                for i in range(N_temp_samples):
                    f.write(f"{t_values[i]:.6f} {Temp_values[i]:.6f} {Temp_values[i]:.6f}\n")
                    
        tcl_thermal_load_def = "pattern Plain 11 Linear {\n"
        for index, row in df_elements.iterrows():
            element_number = row['element_number']
            ele_temp_filepath = os.path.join(ele_temp_file_folder, f"ele{element_number}_temp.txt").replace("\\", "/")
            # ele_temp_relative_filepath = os.path.relpath(ele_temp_filepath, opensees_scripts_folder).replace("\\", "/")
            tcl_thermal_load_def += f"    eleLoad -ele {element_number} -type -beamThermal -source {ele_temp_filepath} -50 50;\n"
        tcl_thermal_load_def += "}\n"

        recorder_path = os.path.join(new_dataset_sim_result_folder, modelfire_xlsx_filename.replace("xlsx", 'txt'))
        tcl_content_all = re.sub(r'^(recorder Node -file\s)([^\s]+)', r'\1' + recorder_path, tcl_gravity_content_all, flags=re.MULTILINE)
        tcl_content_all = tcl_content_all + tcl_thermal_load_def + tcl_fire_analysis_def

        tcl_filepath = os.path.join(opensees_scripts_folder, modelfire_xlsx_filename[:-5] + '.tcl').replace("\\", "/")
        with open(tcl_filepath, 'w') as f:
            f.write(tcl_content_all)

        print(f"Python: Fire script saved to {tcl_filepath}")
        print(f"Python: Evaluating the fire script {modelfire_xlsx_filename[:-5]}...")
        model = ops.Model()
        model.eval(tcl_content_all)
        print("---"*16)
        final_displacements = {
            node: model.nodeDisp(node)[:3] for node in model.getNodeTags()
        }

        model.wipe()
        

        ################################## update the df_nodes ########################################
        df_nodes.loc[:, "final_disp_x"] = df_nodes["node_number"].apply(lambda x: final_displacements[x][0])
        df_nodes.loc[:, "final_disp_y"] = df_nodes["node_number"].apply(lambda x: final_displacements[x][1])
        df_nodes.loc[:, "final_disp_z"] = df_nodes["node_number"].apply(lambda x: final_displacements[x][2])
        df_nodes.loc[:, "final_disp_mag"] = df_nodes.apply(lambda row: np.sqrt(row["final_disp_x"]**2 + row["final_disp_y"]**2 + row["final_disp_z"]**2), axis=1)

        ############################# then also calculate the drift ratio ##############################
        unit_lengths = df_nodes[['x', 'y', 'z']].max().values / df_nodes[['w', 'd', 'h']].max().values
        df_nodes_prev = df_nodes.copy()
        df_nodes_prev['h'] += 1
        df_merged = df_nodes.merge(
            df_nodes_prev,
            on = ['w', 'd', 'h'],
            how = 'left',
            suffixes = ('', '_prev'),
        )
        
        delta_h = unit_lengths[2]
        df_merged['drift_ratio'] = df_merged.apply(lambda row: compute_drift_ratio(row, delta_h), axis=1)
        df_merged['gravity_drift_ratio'] = df_merged.apply(lambda row: compute_gravity_drift_ratio(row, delta_h), axis=1)
        df_nodes = df_merged[['node_number', 'x', 'y', 'z', 'w', 'd', 'h', 'gravity_disp_x', 'gravity_disp_y', 'gravity_disp_z', 'final_disp_x', 'final_disp_y', 'final_disp_z', 'final_disp_mag', 'pyg_node_number', 'drift_ratio', 'gravity_drift_ratio']]
        node_attr_names = ['x', 'y', 'z', 'h']
        node_attrs = torch.tensor(df_nodes[node_attr_names].values, dtype=torch.float)
        gravity_drift_ratio = torch.tensor(df_nodes['gravity_drift_ratio'].values, dtype=torch.float).view(-1,1)

        ########################### (almost) finally save the results #############################
        ############# 1/2. xlsx file ################
        with pd.ExcelWriter(os.path.join(new_dataset_xlsx_folder, modelfire_xlsx_filename).replace("\\", "/")) as writer:
            df_nodes.to_excel(writer, sheet_name='Node_Coordinate', index=False)
            df_elements.to_excel(writer, sheet_name='Element', index=False)
            df_fire.to_excel(writer, sheet_name='Fire_Coordinate', index=False)

        ############# 2/2. pt file ################
        ############### make the beams undirected before to pt#############################
        df_elements_reverse = df_elements.copy()
        df_elements_reverse = df_elements_reverse.rename(columns={'start_node_mapped': 'end_node_mapped', 'end_node_mapped': 'start_node_mapped'})
        df_elements_doubled = pd.concat([df_elements, df_elements_reverse], axis=0)
        edge_index_doubled = torch.tensor(df_elements_doubled[['start_node_mapped', 'end_node_mapped']].values.T, dtype=torch.long)
        edge_attr_names = ['Es', 'Fy', 'b', 'length', 'h', 'gravity_param', 'direction_x', 'direction_y', 'direction_z']
        edge_attr_doubled = torch.tensor(df_elements_doubled[edge_attr_names].values, dtype=torch.float)
        unit_lengths = df_nodes[['x', 'y', 'z']].max().values / df_nodes[['w', 'd', 'h']].max().values
        node_indices = torch.tensor(df_nodes[['w','d','h']].values, dtype=torch.long)
        max_level = torch.tensor(df_nodes['h'].max(), dtype=torch.long)
        delta_h = unit_lengths[2]

        node_dr = torch.tensor(df_nodes['drift_ratio'].values, dtype=torch.float).view(-1,1)
        fire_point = torch.tensor(df_fire[['x', 'y', 'z', 'h']].values, dtype=torch.float).squeeze()
        fire_room_index = torch.tensor(df_fire[['w', 'd', 'h']].values, dtype=torch.long).squeeze()
        drift_ratio_node_ranking = torch.argsort(node_dr.squeeze(), descending=True).reshape(-1, 1)

        pt_graph_data = Data(
            x = node_attrs, 
            y = node_dr,
            edge_index = edge_index_doubled,
            edge_attr = edge_attr_doubled,
            model_number = model_number,
            fire_number = fire_number,
            fire_point = fire_point,
            fire_room_index=fire_room_index,
            unit_lengths=unit_lengths,
            drift_ratio_node_ranking=drift_ratio_node_ranking,
            max_drift_ratio=torch.tensor(df_nodes['drift_ratio'].max(), dtype=torch.float),
            max_level = max_level,
            gravity_drift_ratio = gravity_drift_ratio,
            gravity_max_drift_ratio = gravity_drift_ratio.max(),
        )

        valid_fire_number_list.append(fire_number)
        max_drift_ratio_list.append(df_nodes['drift_ratio'].max())
        pt_graph_data_list.append(pt_graph_data)
        # however, save pt should be done after all fires are done
        fire_end_time = time.time()
        print(f"M{model_number:04d} Fire {fire_number:04d} processed in {fire_end_time - fire_start_time:.2f} seconds.")

    ################################# save pt file ########################################
    print("------"*16)
    combined = list(zip(valid_fire_number_list, max_drift_ratio_list, pt_graph_data_list))
    sorted_combined = sorted(combined, key=lambda x: x[1], reverse=True)
    for rank, (fire_number, max_drift_ratio, pt_graph_data) in enumerate(sorted_combined, 1):
        pt_graph_data.drift_ratio_ranking = torch.tensor(rank, dtype=torch.long)
        pt_filename = f"m{model_number:04d}_f{fire_number:04d}.pt"
        pt_filepath = os.path.join(new_dataset_pt_folder, pt_filename).replace("\\", "/")
        torch.save(pt_graph_data, pt_filepath)
        print(f"Rank {rank}: {pt_filename} saved")
    print(f"Model {model_number} processed")
    print("---" * 16)
    
    ##################### finally zip and then delete the original opensees folder ##############
    ## in python, 7zz to compress all the fires and folders starts with m{model_number:04d}
    ## then, after compress them, delete them.
    print("Compressing and deleting the original opensees files...")
    compress_by_model_number(opensees_scripts_folder, model_number)
    print("Compressing and deleting the original xlsx files...")
    compress_by_model_number(new_dataset_xlsx_folder, model_number)
    print("Compressing and deleting the original sim result files...")    
    compress_by_model_number(new_dataset_sim_result_folder, model_number)

print(f"Model {start_model_number} to {end_model_number} are processed.")





