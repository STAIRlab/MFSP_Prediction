import torch
import torch.nn.functional as F
from torch import nn
import torch_geometric.nn as gnn
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import MLP
from torch_geometric.data import Data
from torch_geometric.nn import global_mean_pool, global_max_pool
from torch_geometric.utils import add_self_loops

class CustomEdgeUpdateLayer1(MessagePassing):
    def __init__(self, node_in_dim, edge_in_dim, mlp_hidden_layers=[], dropout_rates=0.0, **kwargs):
        super(CustomEdgeUpdateLayer1, self).__init__(aggr=None)
        self.node_in_dim = node_in_dim
        self.edge_in_dim = edge_in_dim
        self.mlp_layers = [2 * node_in_dim + edge_in_dim] + mlp_hidden_layers + [edge_in_dim]
        self.mlp = self._build_mlp(self.mlp_layers, dropout_rates)

    def _build_mlp(self, layers, dropout_rates):
        mlp = []
        if isinstance(dropout_rates, (int, float)):
            dropout_rates = [dropout_rates] * (len(layers) - 2)
        elif len(dropout_rates) < len(layers) - 2:
            dropout_rates += [dropout_rates[-1]] * (len(layers) - 2)
        for i in range(len(layers) - 2):
            mlp.append(nn.Linear(layers[i], layers[i + 1]))
            mlp.append(nn.ReLU())
            mlp.append(nn.Dropout(dropout_rates[i]))

        mlp.append(nn.Linear(layers[-2], layers[-1]))
        
        return nn.Sequential(*mlp)

    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):
        edge_input = torch.cat([x_i, x_j, edge_attr], dim=-1)
        return self.mlp(edge_input)

    def aggregate(self, inputs, index):
        return inputs

    def update(self, aggr_out):
        return aggr_out

class CustomGNNLayer1(MessagePassing):
    def __init__(self, node_in_dim, edge_in_dim, 
                node_out_dim=None,
                aggr='max',
                update_bypass_flag=False,
                update_with_target_flag=False, 
                extended_message_in_flag=True, 
                message_out_dim=None,
                message_mlp_hidden_layers=[], 
                message_dropout_rates=0.3, 
                update_mlp_hidden_layers=[],
                update_dropout_rates=0.3,
                **kwargs):
        super(CustomGNNLayer1, self).__init__(aggr=aggr)  
        self.node_in_dim = node_in_dim 
        self.node_out_dim = node_out_dim if node_out_dim else node_in_dim # applied in update function
        self.aggr = aggr
        
        self.extended_message_in_flag = extended_message_in_flag
        self.message_out_dim = message_out_dim if message_out_dim else node_in_dim

        self.message_dropout_rates = message_dropout_rates
        self.message_in_dim = 4 * node_in_dim + edge_in_dim if extended_message_in_flag else node_in_dim + edge_in_dim
        self.message_mlp_layers = [self.message_in_dim] + message_mlp_hidden_layers + [self.message_out_dim]
        self.message_mlp = self._build_mlp(self.message_mlp_layers, self.message_dropout_rates)
        
        self.update_bypass_flag = update_bypass_flag
        self.update_with_target_flag = update_with_target_flag
        if self.update_with_target_flag and (not self.update_bypass_flag):
            self.update_in_dim = 2 * node_in_dim
        else:
            self.update_in_dim = node_in_dim 
        self.update_out_dim = node_out_dim
        self.update_dropout_rates = update_dropout_rates
        self.update_mlp_layers = [self.update_in_dim] + update_mlp_hidden_layers +[self.update_out_dim]
        self.update_mlp = self._build_mlp(self.update_mlp_layers, self.update_dropout_rates)
        self.update_mlp.append(nn.ReLU())

    def _build_mlp(self, layers, dropout_rates):
        mlp = []
        if isinstance(dropout_rates, (int, float)):
            dropout_rates = [dropout_rates] * (len(layers) - 2)
        elif len(dropout_rates) < len(layers) - 2:
            dropout_rates += [dropout_rates[-1]] * (len(layers) - 2)

        for i in range(len(layers) - 2):
            mlp.append(nn.Linear(layers[i], layers[i + 1]))
            mlp.append(nn.ReLU())
            mlp.append(nn.Dropout(dropout_rates[i]))

        mlp.append(nn.Linear(layers[-2], layers[-1]))
        
        return nn.Sequential(*mlp)

    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):
        if self.extended_message_in_flag:
            diff = torch.abs(x_i - x_j)
            avg = (x_i + x_j) / 2
            message_input = torch.cat([x_i, x_j, diff, avg, edge_attr], dim=-1)
        else:
            message_input = torch.cat([x_j, edge_attr], dim=-1)
        return self.message_mlp(message_input)
        

    def update(self, aggr_out, x):
        if self.update_with_target_flag:
            if self.update_bypass_flag:
                return x + self.update_mlp(aggr_out)
            else:
                return self.update_mlp(torch.cat([x, aggr_out], dim=-1))
        else:
            return self.update_mlp(aggr_out)


class CustomGNN1(torch.nn.Module):
    def __init__(self, node_attr_dim, edge_attr_dim, node_emb_dim=32, 
                final_out_dim=1, num_gnn_layers=7, shared_hidden=False, 
                aggr='max',
                edge_emb_flag=False,
                edge_emb_dim=32,
                inter_layer_pypass_flag=True, 
                update_edge_flag=False, 
                extended_message_in_flag=False,
                message_mlp_hidden_layers=[], 
                message_dropout_rates=0.3, 
                intra_layer_pypass_flag=True,
                update_with_target_flag=False,
                update_mlp_hidden_layers=[],
                update_dropout_rates=0.3,
                node_init_mlp_hidden_layers=[],
                node_init_mlp_dropout_rates=0,
                edge_init_mlp_hidden_layers=[], 
                edge_init_mlp_dropout_rates=0, 
                post_mlp_hidden_layers=[], 
                post_mlp_dropout_rates=0.3,
                edge_update_mlp_hidden_layers=[],
                edge_update_mlp_dropout_rates=0.2,
                **kwargs):
        super(CustomGNN1, self).__init__()

        self.num_gnn_layers = num_gnn_layers   
        self.shared_hidden = shared_hidden
        self.inter_layer_pypass_flag = inter_layer_pypass_flag           

        self.node_attr_dim = node_attr_dim
        self.node_emb_dim = node_emb_dim
        self.node_init_mlp_layers = [node_attr_dim] + node_init_mlp_hidden_layers + [node_emb_dim]
        self.node_init_mlp = self._build_mlp(self.node_init_mlp_layers, node_init_mlp_dropout_rates)

        self.edge_emb_flag = edge_emb_flag
        self.edge_attr_dim = edge_attr_dim
        if edge_emb_flag:
            self.edge_emb_dim = edge_emb_dim
            self.edge_init_mlp_dropout_rates = edge_init_mlp_dropout_rates
            self.edge_init_mlp_layers = [edge_attr_dim] + edge_init_mlp_hidden_layers + [edge_emb_dim]
            self.edge_init_mlp = self._build_mlp(self.edge_init_mlp_layers, edge_init_mlp_dropout_rates)
        else:
            edge_emb_dim = edge_attr_dim
            self.edge_emb_dim = self.edge_attr_dim


        self.final_out_dim = final_out_dim
        self.node_init_mlp_dropout_rates = node_init_mlp_dropout_rates
        self.post_mlp_dropout_rates = post_mlp_dropout_rates           
        self.post_mlp_hidden_layers = [node_emb_dim] + post_mlp_hidden_layers + [final_out_dim]
        self.post_mlp = self._build_mlp(self.post_mlp_hidden_layers, post_mlp_dropout_rates)

        self.update_edge_flag = update_edge_flag   
        if update_edge_flag:
            self.edge_update_mlp = CustomEdgeUpdateLayer1(node_in_dim=node_emb_dim, edge_in_dim=edge_emb_dim, mlp_hidden_layers=edge_update_mlp_hidden_layers, dropout_rates=edge_update_mlp_dropout_rates)


        ## to be passed to specific layers
        self.intra_layer_pypass_flag    = intra_layer_pypass_flag       
        self.update_mlp_hidden_layers   = update_mlp_hidden_layers    
        self.update_with_target_flag    = update_with_target_flag
        self.update_dropout_rates       = update_dropout_rates
        self.message_mlp_hidden_layers  = message_mlp_hidden_layers     
        self.message_dropout_rates      = message_dropout_rates 
        self.extended_message_in_flag   = extended_message_in_flag    
        self.aggr = aggr


        if self.shared_hidden:
            self.gnn_layers = torch.nn.ModuleList([
                CustomGNNLayer1(
                    node_in_dim=node_emb_dim,
                    edge_in_dim=edge_emb_dim,
                    node_out_dim=node_emb_dim,
                    update_bypass_flag=intra_layer_pypass_flag,
                    update_with_target_flag=update_with_target_flag,
                    extended_message_in_flag=extended_message_in_flag,
                    message_out_dim=node_emb_dim,
                    message_mlp_hidden_layers=message_mlp_hidden_layers,
                    message_dropout_rates=message_dropout_rates,
                    update_mlp_hidden_layers=update_mlp_hidden_layers,
                    update_dropout_rates=update_dropout_rates,
                    aggr=aggr,
            )])
            
        else:
            self.gnn_layers = torch.nn.ModuleList([
                CustomGNNLayer1(
                    node_in_dim=node_emb_dim,
                    edge_in_dim=edge_emb_dim,
                    node_out_dim=node_emb_dim,
                    update_bypass_flag=intra_layer_pypass_flag,
                    update_with_target_flag=update_with_target_flag,
                    extended_message_in_flag=extended_message_in_flag,
                    message_out_dim=node_emb_dim,
                    message_mlp_hidden_layers=message_mlp_hidden_layers,
                    message_dropout_rates=message_dropout_rates,
                    update_mlp_hidden_layers=update_mlp_hidden_layers,
                    update_dropout_rates=update_dropout_rates,
                    aggr=aggr,
            ) for _ in range(num_gnn_layers)])

        post_mlp_layers = [node_emb_dim] + post_mlp_hidden_layers + [final_out_dim]
        self.post_mlp = self._build_mlp(post_mlp_layers, post_mlp_dropout_rates)
        # self.post_mlp.append(nn.Linear(node_emb_dim,final_out_dim))

    def _build_mlp(self, layers, dropout_rates):
        mlp = []
        if isinstance(dropout_rates, (int, float)):
            dropout_rates = [dropout_rates] * (len(layers) - 2)
        elif len(dropout_rates) < len(layers) - 2:
            dropout_rates += [dropout_rates[-1]] * (len(layers) - 2)

        for i in range(len(layers) - 2):
            mlp.append(nn.Linear(layers[i], layers[i + 1]))
            mlp.append(nn.ReLU())
            mlp.append(nn.Dropout(dropout_rates[i]))

        mlp.append(nn.Linear(layers[-2], layers[-1]))
        
        return nn.Sequential(*mlp)

    def _avg_max_pool_concat(self, x, batch):
        avg_pool = global_mean_pool(x, batch)
        max_pool = global_max_pool(x, batch)
        return torch.cat([avg_pool, max_pool], dim=1)
    
    def forward(self, x, edge_index, edge_attr, graph_embedding=False, batch=None, N_forward_layers=None):
        if N_forward_layers is None:
            N_forward_layers = self.num_gnn_layers
        else:
            assert N_forward_layers <= self.num_gnn_layers
            assert N_forward_layers >= 2


        x = self.node_init_mlp(x)
        if self.edge_emb_flag:
            edge_attr = self.edge_init_mlp(edge_attr)

        if not graph_embedding:
            for idx in range(N_forward_layers):
                if self.inter_layer_pypass_flag:
                    node_identity = x
                    if self.update_edge_flag:
                        edge_identity = edge_attr
                
                ## update node
                if self.shared_hidden:
                    x = self.gnn_layers[0](x, edge_index, edge_attr)
                else:
                    x = self.gnn_layers[idx](x, edge_index, edge_attr)

                ## update edge
                if self.update_edge_flag:
                    edge_attr = self.edge_update_mlp(x, edge_index, edge_attr)

                if self.inter_layer_pypass_flag:
                    x = x + node_identity
                    if self.update_edge_flag:
                        edge_attr = edge_attr + edge_identity
            return self.post_mlp(x)
        

        for idx in range(N_forward_layers):
            if self.inter_layer_pypass_flag:
                node_identity = x
                if self.update_edge_flag:
                    edge_identity = edge_attr
            
            ## update node
            if self.shared_hidden:
                x = self.gnn_layers[0](x, edge_index, edge_attr)
            else:
                x = self.gnn_layers[idx](x, edge_index, edge_attr)

            ## update edge
            if self.update_edge_flag:
                edge_attr = self.edge_update_mlp(x, edge_index, edge_attr)

            if self.inter_layer_pypass_flag:
                x = x + node_identity
                if self.update_edge_flag:
                    edge_attr = edge_attr + edge_identity
        return self._avg_max_pool_concat(x, batch)


class MIDRPredictor1(torch.nn.Module):
    def __init__(self, layers=[64, 1], dropouts=0.1, **kwargs):
        super(MDRPredictor1, self).__init__()
        self.layers = self._build_mlp(layers, dropouts)
            
    def _build_mlp(self, layers, dropout_rates):
        mlp = []
        if isinstance(dropout_rates, (int, float)):
            dropout_rates = [dropout_rates] * (len(layers) - 2)
        elif len(dropout_rates) < len(layers) - 2:
            dropout_rates += [dropout_rates[-1]] * (len(layers) - 2)

        for i in range(len(layers) - 2):
            mlp.append(nn.Linear(layers[i], layers[i + 1]))
            mlp.append(nn.ReLU())
            mlp.append(nn.Dropout(dropout_rates[i]))

        mlp.append(nn.Linear(layers[-2], layers[-1]))
        
        return nn.Sequential(*mlp)

    def forward(self, x):
        return self.layers(x)


class MFSPPredictor1(nn.Module):
    def __init__(self, layers=[128, 32, 3], dropouts=0.1, **kwargs):
        super(MFSPPredictor1, self).__init__()
        self.layers = self._build_bypass_mlp(layers, dropouts)
        self.output_activation = nn.Sigmoid()

    def _build_bypass_mlp(self, layers, dropout_rates):
        blocks = []
        if isinstance(dropout_rates, (int, float)):
            dropout_rates = [dropout_rates] * (len(layers) - 2)
        elif len(dropout_rates) < len(layers) - 2:
            dropout_rates += [dropout_rates[-1]] * (len(layers) - 2)

        for i in range(len(layers) - 1):
            in_dim, out_dim = layers[i], layers[i + 1]
            block = []
            block.append(nn.Linear(in_dim, out_dim))
            if i < len(layers) - 2: 
                block.append(nn.ReLU())
                block.append(nn.Dropout(dropout_rates[i]))
            blocks.append(nn.Sequential(*block))
        
        return nn.ModuleList(blocks)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            out = layer(x)  
            x = out  

        return self.output_activation(out)
