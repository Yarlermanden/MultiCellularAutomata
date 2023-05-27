from GNCAmodel import GNCA
from torch_geometric_temporal.nn.recurrent import gconv_gru
from torch_geometric.nn import GCN
import torch
import torch.nn as nn
from torch_geometric.nn.norm import pair_norm

from custom_conv import *
from enums import *
from graphUtils import *
from gat_edge_conv import *

class NodeNorm(nn.Module):
    def __init__(
        self,
        unbiased: Optional[bool] = False,
        eps: Optional[float] = 1e-5,
        root_power: Optional[float] =3
    ):
        super(NodeNorm, self).__init__()
        self.unbiased = unbiased
        self.eps = eps
        self.power = 1 / root_power

    def forward(self, x: torch.Tensor):
        std = (torch.var(x, unbiased=self.unbiased, dim=-1, keepdim=True) + self.eps).sqrt()
        x = x / torch.pow(std, self.power)
        return x

    def __repr__(self):
        return f'{self.__class__.__name__}()'

class Conv(GNCA):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.velNorm = 1.0*self.settings.scale/self.velocity_scale
        self.attrNorm = 1.0*self.settings.scale/self.settings.radius_food

        self.hidden_after_size = self.hidden_size + 4
        #if self.model_type == ModelType.WithGlobalNode: self.hidden_after_size += self.hidden_size

        mlp_after_size = 22
        if self.model_type == ModelType.WithGlobalNode: mlp_after_size += 10
        self.mlp_after = nn.Sequential(
            nn.Linear(mlp_after_size, 22),
            nn.Tanh(),
            nn.Linear(22, 7),
        )

        self.mlp_hidden = nn.Sequential(
            nn.Linear(mlp_after_size+10, 10),
            nn.Tanh(),
        )

        self.gConvGRU = gconv_gru.GConvGRU(in_channels=8, out_channels=8, K=1).to(self.device)

        self.conv_layer_cell = GATConv(self.hidden_size, self.output_channels, edge_dim=self.edge_dim-1)
        self.edge_conv_food = GATEdgeConv(3, 2, edge_dim=self.edge_dim-1)
        self.edge_conv_wall = GATEdgeConv(3, 2, edge_dim=self.edge_dim-1)
        if self.model_type == ModelType.WithGlobalNode:
            self.conv_layer_global = GCN(10, 10, 1, self.hidden_size)
        
        self.mlp_x = nn.Sequential(
            nn.Linear(self.hidden_size-1, self.hidden_size-1),
            nn.Tanh(),
            nn.Linear(self.hidden_size-1, 1),
            nn.Tanh(),
        )

        self.H = None
        self.pair_norm = pair_norm.PairNorm(scale=1.0)
        self.node_norm = NodeNorm(root_power=2.0)
        for param in self.parameters():
            param.grad = None

    def gru(self, edges, x):
        if self.H is None:
            self.H = torch.zeros_like(x, device=self.device)
        if self.node_indices_to_keep is not None:
            self.H = self.H[self.node_indices_to_keep].view(self.node_indices_to_keep.shape[0], self.H.shape[1])
        if self.node_indices_to_create is not None:
            ...
            self.node_indices_to_create = None
        self.H = torch.tanh(self.gConvGRU(x, edges, H=self.H))
        #TODO find some way of allowing a new mask for adding nodes inbetween the others...
        #index the new nodes?
        #index of all the old nodes?
        return self.H

    def nodeNorm(self, x):
        x_mean = torch.mean(x, dim=1, keepdim=True)
        x_std = torch.std(x, dim=1, keepdim=True)
        return torch.tanh(((x-x_mean) / x_std)*2 - 1)

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
        x_food = self.edge_conv_food(x=x, edge_index=food_edges, edge_attr=food_attr)[c_mask]
        x_wall = self.edge_conv_wall(x=x, edge_index=wall_edges, edge_attr=wall_attr)[c_mask]
        x_cell = self.conv_layer_cell(x=x, edge_index=cell_edges, edge_attr=cell_attr)[c_mask]
        x_x = self.mlp_x( torch.cat( (torch.norm(x[c_mask, :2], dim=1).unsqueeze(dim=1), x[c_mask, 2:]), dim=1)) * x[c_mask, :2]

        output = torch.zeros((x.shape[0], self.output_channels), device=self.device)

        x_magnitude = torch.norm(x_x, dim=1, keepdim=True)
        cell_magnitude = torch.norm(x_cell[:, :2], dim=1, keepdim=True)
        food_magnitude = torch.norm(x_food, dim=1, keepdim=True)
        wall_magnitude = torch.norm(x_wall, dim=1, keepdim=True)

        if self.model_type == ModelType.WithGlobalNode:
            #c_mask = torch.bitwise_or(c_mask, graph.x[:,4] == NodeType.GlobalCell)
            #x_global = self.conv_layer_global(x=x, edge_index=global_edges, edge_attr=global_attr)
            x_global = torch.tanh(self.conv_layer_global(x=x[:, -10:], edge_index=global_edges))
            g_mask = graph.x[:, 4] == NodeType.GlobalCell
            #graph.x[g_mask, 2:4] = x_global[g_mask, :2] #update global
            #graph.x[g_mask, 5:] = x_global[g_mask, 2:] #update global
            graph.x[g_mask, -10:] = x_global[g_mask, -10:]
            #x = torch.concat((x_food, x_cell[:, :2], x_wall, x_global), dim=1)

            #TODO broadcast the global node
            global_hidden = graph.x[g_mask, -10:].repeat(c_mask.sum(), 1)
            #TODO ensure that only edges towards the global node is added - we don't want to update the other nodes from the global node...

            input = torch.cat((x_magnitude, cell_magnitude, food_magnitude, wall_magnitude, x_x, x_cell[:, :2], x_food, x_wall, x_origin[c_mask, 3:], global_hidden), dim=1)
        else: 
            input = torch.cat((x_magnitude, cell_magnitude, food_magnitude, wall_magnitude, x_x, x_cell[:, :2], x_food, x_wall, x_origin[c_mask, 3:]), dim=1)

        x = self.mlp_after(input)
        output[c_mask, :2] = torch.tanh(x[:, 0:1] * x_x + x[:, 1:2] * x_cell[:, :2] + x[:, 2:3] * x_food + x[:, 3:4] * x_wall + x[:, 4:5] * torch.clamp(x[:, 5:6], -1., 1.))

        h = self.mlp_hidden(torch.cat((x_cell[:, 2:], input), dim=1)) + x_origin[c_mask, 3:]

        x = output
        x[c_mask, 2:] = self.node_norm(h)
        #x[c_mask, 2:] = self.pair_norm(h)
        #x[c_mask, 2:] = h

        #x[c_mask, 2:] = torch.tanh(x[c_mask, 2:]/10 + x_origin[c_mask, 3:]*0.75)

        #x[:, 2:] = torch.tanh(self.pair_norm(x[:, 2:] + x_origin[:, 3:]))
        #x[:, 2:] = self.pair_norm(x[:, 2:] + x_origin[:, 3:])
        #x[c_mask, 2:] = self.pair_norm.forward(x[c_mask, 2:] + x_origin[c_mask, 3:]) #TODO compute this correctly

        if(torch.any(torch.isnan(x))):
            print('conv results in nan...')
            x[torch.isnan(x)] = 0

        return x 

    def forward(self, *args):
        self.H = None
        self.node_indices_to_keep = None
        self.node_indices_to_create = None
        self.conv_layer_cell = self.conv_layer_cell.to(self.device)
        self.edge_conv_food = self.edge_conv_food.to(self.device)
        self.edge_conv_wall = self.edge_conv_wall.to(self.device)
        if self.model_type == ModelType.WithGlobalNode:
            self.conv_layer_global = self.conv_layer_global.to(self.device)
        return super().forward(*args)