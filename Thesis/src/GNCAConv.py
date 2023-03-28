from GNCAmodel import GNCA
from torch_geometric_temporal.nn.recurrent import gconv_gru
from torch_geometric.nn import GCN
import torch
import torch.nn as nn

from custom_conv import CustomConv, CustomConvSimple

class Conv(GNCA):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = self.input_channels*1

        self.velNorm = 1.0*self.scale/self.max_velocity
        self.attrNorm = 1.0*self.scale/self.radius_food
        
        self.conv_layer_food = CustomConvSimple(self.hidden_size, dim=self.edge_dim-1, aggr='add')
        self.conv_layer_cell = CustomConvSimple(self.hidden_size, dim=self.edge_dim-1, aggr='add')
        if self.with_global_node:
            self.conv_layer_global = CustomConvSimple(self.hidden_size, dim=self.edge_dim-1, aggr='mean')
            self.gConvGRU = gconv_gru.GConvGRU(in_channels=self.hidden_size*3, out_channels=self.hidden_size*3, K=1).to(self.device)
        else:
            self.gConvGRU = gconv_gru.GConvGRU(in_channels=self.hidden_size*2, out_channels=self.hidden_size*2, K=1).to(self.device)

        self.mlp_before = nn.Sequential(
            nn.Linear(self.input_channels, self.hidden_size),
            nn.Tanh(),
        )
        self.H = None

        for param in self.parameters():
            param.grad = None

    def message_pass(self, graph):
        food_edges = graph.edge_index[:, torch.nonzero(graph.edge_attr[:, 3] == 0).flatten()]
        food_attr = graph.edge_attr[torch.nonzero(graph.edge_attr[:, 3] == 0).flatten()][:, :3]
        cell_edges = graph.edge_index[:, torch.nonzero(graph.edge_attr[:, 3] == 1).flatten()]
        cell_attr = graph.edge_attr[torch.nonzero(graph.edge_attr[:, 3] == 1).flatten()][:, :3]
        if self.with_global_node:
            global_edges = graph.edge_index[:, torch.nonzero(graph.edge_attr[:, 3] == 2).flatten()]
            global_attr = graph.edge_attr[torch.nonzero(graph.edge_attr[:, 3] == 2).flatten()][:, :3]

        x_origin = torch.concat((graph.x[:, 2:4] * self.velNorm, graph.x[:, 5:]), dim=1) 
        food_attr *= self.attrNorm
        cell_attr *= self.attrNorm
        
        x = self.mlp_before(x_origin)
        x_food = self.conv_layer_food(x=x, edge_index=food_edges, edge_attr=food_attr)
        x_cell = self.conv_layer_cell(x=x, edge_index=cell_edges, edge_attr=cell_attr)
        #x = x_food + x_cell #could consider catting this instead?
        if self.with_global_node:
            x_global = self.conv_layer_global(x=x, edge_index=global_edges, edge_attr=global_attr)
            x = torch.concat((x_food, x_cell, x_global), dim=1)
        else: x = torch.concat((x_food, x_cell), dim=1)

        if self.with_global_node:
            x = x[:, :self.hidden_size] + x[:, self.hidden_size:self.hidden_size*2] + x[:, self.hidden_size*2:]
        else: x = x[:, :self.hidden_size] + x[:, self.hidden_size:]

        x = x_origin + x
        return x

    def forward(self, *args):
        self.H = None
        self.node_indices_to_keep = None
        self.mlp_before = self.mlp_before.to(self.device)
        self.conv_layer_cells = self.conv_layer_cell.to(self.device)
        self.conv_layer_food = self.conv_layer_food.to(self.device)
        if self.with_global_node:
            self.conv_layer_global = self.conv_layer_global.to(self.device)
        return super().forward(*args)