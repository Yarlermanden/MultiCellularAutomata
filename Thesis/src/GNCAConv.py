from GNCAmodel import GNCA
from torch_geometric_temporal.nn.recurrent import gconv_gru
from torch_geometric.nn import GCN
import torch
import torch.nn as nn

from custom_conv import CustomConvSimple
from enums import *

class Conv(GNCA):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.velNorm = 1.0*self.settings.scale/self.max_velocity
        self.attrNorm = 1.0*self.settings.scale/self.settings.radius_food
        
        #self.mlp_before = nn.Sequential(
        #    nn.Linear(self.input_channels, self.hidden_size),
        #    nn.Tanh(),
        #)

        self.hidden_after_size = self.hidden_size + 2 + 2
        if self.model_type == ModelType.WithGlobalNode: self.hidden_after_size += self.hidden_size

        self.mlp_after = nn.Sequential(
            nn.Linear(self.hidden_after_size, self.hidden_after_size),
            nn.Tanh(),
            #nn.Linear(self.hidden_after_size, self.hidden_after_size),
            #nn.Tanh(),
            nn.Linear(self.hidden_after_size, self.output_channels),
            nn.Tanh(),
        )

        self.conv_layer_food = CustomConvSimple(2, dim=self.edge_dim-1, aggr='mean')
        self.conv_layer_cell = CustomConvSimple(self.hidden_size, dim=self.edge_dim-1, aggr='mean')
        self.conv_layer_wall = CustomConvSimple(2, dim=self.edge_dim-1, aggr='mean')
        if self.model_type == ModelType.WithGlobalNode:
            self.conv_layer_global = CustomConvSimple(self.hidden_size, dim=self.edge_dim-1, aggr='mean')
        #self.gConvGRU = gconv_gru.GConvGRU(in_channels=self.hidden_after_size, out_channels=self.hidden_after_size, K=1).to(self.device)

        self.H = None
        for param in self.parameters():
            param.grad = None

    def gru(self, graph, x):
        if self.H is None:
            self.H = torch.zeros_like(x, device=self.device)
        if self.node_indices_to_keep is not None:
            self.H = self.H[self.node_indices_to_keep].view(self.node_indices_to_keep.shape[0], self.H.shape[1])
        self.H = torch.tanh(self.gConvGRU(x, graph.edge_index, H=self.H))
        return self.H

    def message_pass(self, graph):
        food_edges = graph.edge_index[:, torch.nonzero(graph.edge_attr[:, 3] == 0).flatten()]
        food_attr = graph.edge_attr[torch.nonzero(graph.edge_attr[:, 3] == 0).flatten()][:, :3]
        cell_edges = graph.edge_index[:, torch.nonzero(graph.edge_attr[:, 3] == 1).flatten()]
        cell_attr = graph.edge_attr[torch.nonzero(graph.edge_attr[:, 3] == 1).flatten()][:, :3]
        if self.model_type == ModelType.WithGlobalNode:
            global_edges = graph.edge_index[:, torch.nonzero(graph.edge_attr[:, 3] == 2).flatten()]
            global_attr = graph.edge_attr[torch.nonzero(graph.edge_attr[:, 3] == 2).flatten()][:, :3]
        wall_edges = graph.edge_index[:, torch.nonzero(graph.edge_attr[:, 3] == 4).flatten()]
        wall_attr = graph.edge_attr[torch.nonzero(graph.edge_attr[:, 3] == 4).flatten()][:, :3]

        energy_norm = graph.x[:, 5:6] * 0.01
        x_origin = torch.concat((graph.x[:, 2:4] * self.velNorm, energy_norm, graph.x[:, 6:]), dim=1)  #vel, energy, hidden
        food_attr *= self.attrNorm
        cell_attr *= self.attrNorm
        wall_attr *= self.attrNorm
        
        #x = self.mlp_before(x_origin)
        x = x_origin
        x_food = self.conv_layer_food(x=x[:, :2], edge_index=food_edges, edge_attr=food_attr)
        x_cell = self.conv_layer_cell(x=x, edge_index=cell_edges, edge_attr=cell_attr)
        x_wall = self.conv_layer_wall(x=x[:, :2], edge_index=wall_edges, edge_attr=wall_attr)
        #having no edges in a specific type now results in these being 0 all across the board
        #x = x_food + x_cell #could consider catting this instead?
        if self.model_type == ModelType.WithGlobalNode:
            x_global = self.conv_layer_global(x=x, edge_index=global_edges, edge_attr=global_attr)
            x = torch.concat((x_food, x_cell, x_wall, x_global), dim=1)
        else: x = torch.concat((x_food, x_cell, x_wall), dim=1)
        x = torch.tanh(x)

        #x = self.gru(graph, x)
        x = self.mlp_after(x)

        #don't make residual connections as we don't want momentum to carry over or values to be able to go beyond/below 1/-1 
        #simply persist should result in staying still and not moving...
        #x[:, :2] += x_origin[:, :2] #only the velX and velY should be kept
        #hidden values shouldn't be added or subtracted - but instead completely override?

        #x += torch.concat((x_origin[:, :2], x_origin[:, 3:]), dim=1)
        #x = torch.tanh(x) #enforces values between -1 and 1 - furthermore forces values a bit lower than otherwise, meaning if nothing is added, it will decrease with time - hopefully less exploding values
        return x

    def forward(self, *args):
        self.H = None
        self.node_indices_to_keep = None
        #self.mlp_before = self.mlp_before.to(self.device)
        self.conv_layer_cell = self.conv_layer_cell.to(self.device)
        self.conv_layer_food = self.conv_layer_food.to(self.device)
        self.conv_layer_wall = self.conv_layer_wall.to(self.device)
        if self.model_type == ModelType.WithGlobalNode:
            self.conv_layer_global = self.conv_layer_global.to(self.device)
        return super().forward(*args)