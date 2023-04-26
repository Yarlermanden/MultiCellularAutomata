from GNCAmodel import GNCA
from torch_geometric_temporal.nn.recurrent import gconv_gru
from torch_geometric.nn import GCN
import torch
import torch.nn as nn
from torch_geometric.nn.norm import pair_norm

from custom_conv import *
from enums import *
from graphUtils import *

class Conv(GNCA):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.velNorm = 1.0*self.settings.scale/self.velocity_scale
        self.attrNorm = 1.0*self.settings.scale/self.settings.radius_food

        self.hidden_after_size = self.hidden_size + 2 + 2
        if self.model_type == ModelType.WithGlobalNode: self.hidden_after_size += self.hidden_size

        self.mlp_after = nn.Sequential(
            nn.Linear(self.hidden_after_size, self.hidden_size+4),
            nn.Tanh(),
            #nn.Linear(self.hidden_after_size, self.hidden_after_size),
            #nn.Tanh(),
            nn.Linear(self.hidden_size+4, self.output_channels),
            nn.Tanh(),
        )

        self.conv_layer_cell = CustomConv(self.hidden_size, dim=self.edge_dim-1, aggr='mean')
        self.mean_conv = MeanEdgeConv(2, dim=self.edge_dim-1)
        if self.model_type == ModelType.WithGlobalNode:
            #self.conv_layer_global = CustomConvSimple(self.hidden_size, dim=self.edge_dim-1, aggr='mean')
            self.conv_layer_global = GCN(self.hidden_size, self.hidden_size, 1, self.hidden_size)
        #self.gConvGRU = gconv_gru.GConvGRU(in_channels=2, out_channels=2, K=1).to(self.device)

        self.H = None
        self.pair_norm = pair_norm.PairNorm()
        for param in self.parameters():
            param.grad = None

    def gru(self, edges, x):
        if self.H is None:
            self.H = torch.zeros_like(x, device=self.device)
        if self.node_indices_to_keep is not None:
            self.H = self.H[self.node_indices_to_keep].view(self.node_indices_to_keep.shape[0], self.H.shape[1])
        self.H = torch.tanh(self.gConvGRU(x, edges, H=self.H))
        return self.H

    def message_pass(self, graph):
        food_edges = graph.edge_index[:, torch.nonzero(graph.edge_attr[:, 3] == 0).flatten()]
        food_attr = graph.edge_attr[torch.nonzero(graph.edge_attr[:, 3] == 0).flatten()][:, :3]
        cell_edges = graph.edge_index[:, torch.nonzero(graph.edge_attr[:, 3] == 1).flatten()]
        cell_attr = graph.edge_attr[torch.nonzero(graph.edge_attr[:, 3] == 1).flatten()][:, :3]
        if self.model_type == ModelType.WithGlobalNode:
            global_edges = graph.edge_index[:, torch.nonzero(graph.edge_attr[:, 3] == 2).flatten()]
        wall_edges = graph.edge_index[:, torch.nonzero(graph.edge_attr[:, 3] == 4).flatten()]
        wall_attr = graph.edge_attr[torch.nonzero(graph.edge_attr[:, 3] == 4).flatten()][:, :3]
        c_mask = cell_mask(graph.x)

        x_origin = torch.concat((graph.x[:, 2:4], graph.x[:, 5:6], graph.x[:, 6:]), dim=1)  #vel, energy, hidden
        x_origin[c_mask, :2] * self.velNorm
        x_origin[c_mask, 2] * 0.01 #energy norm
        food_attr *= self.attrNorm
        cell_attr *= self.attrNorm
        wall_attr *= self.attrNorm
        
        x = x_origin
        x_food = self.mean_conv(x=x, edge_index=food_edges, edge_attr=food_attr)
        x_wall = self.mean_conv(x=x, edge_index=wall_edges, edge_attr=wall_attr)
        x_cell = torch.tanh(self.conv_layer_cell(x=x, edge_index=cell_edges, edge_attr=cell_attr))
        #having no edges in a specific type now results in these being 0 all across the board
        #x = x_food + x_cell #could consider catting this instead?
        if self.model_type == ModelType.WithGlobalNode:
            #c_mask = torch.bitwise_or(c_mask, graph.x[:,4] == NodeType.GlobalCell)
            #x_global = self.conv_layer_global(x=x, edge_index=global_edges, edge_attr=global_attr)
            x_global = torch.tanh(self.conv_layer_global(x=x, edge_index=global_edges))
            g_mask = graph.x[:, 4] == NodeType.GlobalCell
            graph.x[g_mask, 2:4] = x_global[g_mask, :2] #update global
            graph.x[g_mask, 5:] = x_global[g_mask, 2:] #update global
            x = torch.concat((x_food, x_cell, x_wall, x_global), dim=1)
        else: x = torch.concat((x_food, x_cell, x_wall), dim=1)

        output = torch.zeros((x.shape[0], self.output_channels), device=self.device)
        output[c_mask] = self.mlp_after(x[c_mask])
        x = output
        #x[:, :2] += self.gru(cell_edges, x[:, :2])

        #... and normalize hidden features H
        x[c_mask, 2:] = torch.tanh(x[c_mask, 2:]/10 + x_origin[c_mask, 3:]*0.75)
        #x[:, 2:] = torch.tanh(self.pair_norm(x[:, 2:] + x_origin[:, 3:]))
        #x[:, 2:] = self.pair_norm(x[:, 2:] + x_origin[:, 3:])
        #x[c_mask, 2:] = self.pair_norm.forward(x[c_mask, 2:] + x_origin[c_mask, 3:]) #TODO compute this correctly

        return x 

    def forward(self, *args):
        self.H = None
        self.node_indices_to_keep = None
        self.conv_layer_cell = self.conv_layer_cell.to(self.device)
        if self.model_type == ModelType.WithGlobalNode:
            self.conv_layer_global = self.conv_layer_global.to(self.device)
        return super().forward(*args)