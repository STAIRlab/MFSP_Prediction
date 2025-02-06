# Prediction of the Most Fire-Sensitive Point in Building Structures with Differentiable Agents for Thermal Simulators

[arXiv preprint](https://arxiv.org/abs/2502.03424)

## dataset
- [Google Drive](https://drive.google.com/drive/folders/1YU_oDelIoU-TRzj0TNDdZUvULSFNzlS7 )
    - `drs3v6` is the unfiltered labeled dataset. It includes 3,000 structures and their gravity simulation results as well as 30 fire case simulation for each structure.
    - `drs4v2` is the unfiltered unlabeled dataset. It includes 30,000 structures and their gravity simulation results.

## requirements
- to generate dataset, you may need:
    - opensees: https://pypi.org/project/opensees/
    - pandas
- to run or train the neural networks, you may need:
    - pytorch
    - pytorch_geometric
- before running, rename the `config-demo.json` as `config.json` and edit the `basedir` as the absolute path of your working directory.


## Dataset Generation
- Generate geometry by `geometry_gen.py`
- OpenSees Simulation by `ops_simulation.py`
    - for the saved `.pt` file, it is supposed to contain the following information

    | attribute_name | meannning | comment |
    | --- | --- | --- |
    | x | node features: the coordinates and indices: $[x_i, y_i, z_i, h_i]$ | dim: (N_node, 4) |
    | y | drift ratio for each node | dim: (N_node, 1) |
    | edge_index | the starting and ending nodes' number of each edge | dim: (2, N_edge) |
    | edge_attr | edge features: ['Es', 'Fy', 'b', 'length', 'h', 'gravity_param', 'direction_x', 'direction_y', 'direction_z']| dim: (N_edge, 9) |
    | model_number | model number | scaler, normal int, not torch tensor |
    | fire_number | fire number (corresponding to file) | scaler, normal int, not torch tensor | 
    | fire_point | coordinates of the fire: $[x_0, y_0, z_0, h_0]$ | dim: (4)  |
    | fire_room_index | fire room indices: $[w_i, d_i, h_i]$ | dim: (3) |
    | unit_lengths | unit width, unit depth, unit height, in mm | dim: (3) |
    | node_indices | node indices: $[w_i, d_i, h_i]$ | dim: (N_node, 3) |
    | drift_ratio_node_ranking | ranking of each node by node's drift ratio, descending | dim: (N_node, 1) |
    | drift_ratio_ranking | ranking of the maximum drift ratio of each graph | scaler, torch.long |
    | max_drift_ratio | max of y | scaler tensor |
    | max_level | max of x[:,3] | scaler tensor |

- Then in `filter.py`, the dataset was filtered by the maximum interstory drift ratio under gravity (also filter ou those obviously unreliable data).

- The scripts in data_generation folder now only present the generation of labeled dataset. Functions to generate unlabeled is a subset of that to generate labeled dataset. 


## Maximum Interstory Drift Ratio (MIDR) Predictor
- As the filenames indicate, there are `strawman1`, `strawman2` and `proposed` scripts, they are for trainning the NN models
- Then the `evaluation` scripts are to evaluate these models together.

## Most Fire-Sensitive Point (MFSP) Predictor
- mfsp_train.py to train a MFSP predictor and mfsp_eval.py to evaluate it.
