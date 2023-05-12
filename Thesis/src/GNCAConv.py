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

class Conv(GNCA):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.velNorm = 1.0*self.settings.scale/self.velocity_scale
        self.attrNorm = 1.0*self.settings.scale/self.settings.radius_food

        self.hidden_after_size = self.hidden_size + 4
        if self.model_type == ModelType.WithGlobalNode: self.hidden_after_size += self.hidden_size

        self.mlp_after = nn.Sequential(
            nn.Linear(5, 5),
            nn.Tanh(),
            nn.Linear(5, 2),
            nn.Tanh(),
        )

        #self.conv_layer_cell = CustomConv(self.hidden_size, dim=self.edge_dim-2, aggr='mean')
        self.conv_layer_cell = GATConv(self.hidden_size, self.output_channels, edge_dim=self.edge_dim-1)

        #self.mean_conv = MeanEdgeConv(2, dim=self.edge_dim-1)
        #self.edge_conv_food = EdgeConv(2, dim=self.edge_dim-1)
        #self.edge_conv_wall = EdgeConv(2, dim=self.edge_dim-1)
        self.edge_conv_food = GATEdgeConv(3, 2, edge_dim=self.edge_dim-1)
        self.edge_conv_wall = GATEdgeConv(3, 2, edge_dim=self.edge_dim-1)
        if self.model_type == ModelType.WithGlobalNode:
            #self.conv_layer_global = CustomConvSimple(self.hidden_size, dim=self.edge_dim-1, aggr='mean')
            self.conv_layer_global = GCN(self.hidden_size, self.hidden_size, 1, self.hidden_size)
        #self.gConvGRU = gconv_gru.GConvGRU(in_channels=2, out_channels=2, K=1).to(self.device)

        self.mlp_x = nn.Sequential(
            nn.Linear(self.hidden_size-1, self.hidden_size-1),
            nn.Tanh(),
            nn.Linear(self.hidden_size-1, 1),
            nn.Tanh(),
        )

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
        #x_food = self.mean_conv(x=x, edge_index=food_edges, edge_attr=food_attr)
        #x_wall = self.mean_conv(x=x, edge_index=wall_edges, edge_attr=wall_attr)
        #TODO use cell mask already here to limit computations
        x_food = torch.tanh(self.edge_conv_food(x=x, edge_index=food_edges, edge_attr=food_attr)[c_mask])
        x_wall = torch.tanh(self.edge_conv_wall(x=x, edge_index=wall_edges, edge_attr=wall_attr)[c_mask])
        x_cell = torch.tanh(self.conv_layer_cell(x=x, edge_index=cell_edges, edge_attr=cell_attr)[c_mask])

        x_x = self.mlp_x( torch.cat( (torch.norm(x[c_mask, :2], dim=1).unsqueeze(dim=1), x[c_mask, 2:]), dim=1)) * x[c_mask, :2]

        #h = x_cell[c_mask, 2:] + x_origin[c_mask, 3:]
        h = x_cell[:, 2:] + x_origin[c_mask, 3:]

        #having no edges in a specific type now results in these being 0 all across the board
        #x = x_food + x_cell #could consider catting this instead?
        if self.model_type == ModelType.WithGlobalNode:
            #c_mask = torch.bitwise_or(c_mask, graph.x[:,4] == NodeType.GlobalCell)
            #x_global = self.conv_layer_global(x=x, edge_index=global_edges, edge_attr=global_attr)
            x_global = torch.tanh(self.conv_layer_global(x=x, edge_index=global_edges))
            g_mask = graph.x[:, 4] == NodeType.GlobalCell
            graph.x[g_mask, 2:4] = x_global[g_mask, :2] #update global
            graph.x[g_mask, 5:] = x_global[g_mask, 2:] #update global
            x = torch.concat((x_food, x_cell[:, :2], x_wall, x_global), dim=1)
        else: 
            #x = torch.concat((x_food, x_cell[:, :2], x_wall), dim=1)
            #x = torch.concat((
            #    torch.norm(x_food, dim=1).unsqueeze(dim=1), 
            #    torch.norm(x_cell[:, :2], dim=1).unsqueeze(dim=1), 
            #    torch.norm(x_wall, dim=1).unsqueeze(dim=1), 
            #    torch.norm(x_x, dim=1).unsqueeze(dim=1)
            #), dim=1)
            ...

        output = torch.zeros((x.shape[0], self.output_channels), device=self.device)

        #somehow subtract the direction from each type - give the magnitude of each - how important is each
        #compute the relative direction of each - something like the dot...
        #give mlp the magnitudes and relative direction of food and obstacle compared to the cell
        #let mlp output direction - that is output of size 2
        #multiply this with direction of cells
        #cell_dir, food_dir, obstacle_dir
        #cell_magni, food_magni, obstacle_magni, cell_food_rot, cell_obstacle_rot
        #where do we want to go - invariant...
        #multiply by cell_dir
        x_cell_vel = x_x+x_cell[:, :2]
        cell_magnitude = torch.norm(x_cell_vel, dim=1, keepdim=True)
        food_magnitude = torch.norm(x_food, dim=1, keepdim=True)
        wall_magnitude = torch.norm(x_wall, dim=1, keepdim=True)
        cell_norm = F.normalize(x_cell_vel, dim=1)
        #food_norm = F.normalize(x_food, dim=1)
        #wall_norm = F.normalize(x_wall, dim=1)

        #cell_norm = x_cell_vel / cell_magnitude #would need to handle case of 0
        #food_norm = x_food / food_magnitude
        #wall_norm = x_wall / wall_magnitude

        #cell_food_angle = torch.acos(torch.dot(cell_norm, food_norm))
        #cell_wall_angle = torch.acos(torch.dot(cell_norm, wall_norm))
        #cell_food_angle = torch.acos(torch.clamp(torch.sum(cell_norm * food_norm, dim=1), -0.99999, 0.99999)).unsqueeze(dim=1) #sum is only used to broadcast here...
        #cell_wall_angle = torch.acos(torch.clamp(torch.sum(cell_norm * wall_norm, dim=1), -0.99999, 0.99999)).unsqueeze(dim=1)
        cell_food_angle = torch.acos(torch.clamp((x_cell_vel * x_food).sum(dim=1, keepdim=True) / (cell_magnitude * food_magnitude + 1e-7), -1.0, 1.0))
        cell_wall_angle = torch.acos(torch.clamp((x_cell_vel * x_wall).sum(dim=1, keepdim=True) / (cell_magnitude * wall_magnitude + 1e-7), -1.0, 1.0))
        input = torch.cat((cell_magnitude, food_magnitude, wall_magnitude, cell_food_angle, cell_wall_angle), dim=1)
        output[c_mask, :2] = self.mlp_after(input) * cell_norm


        #could we rotate all of them in some way to make the output E(n) variant before rotating them back?
        #could we take them norm of each type and input to the MLP
        #then take the output and multiply with each of them - 3 input, 6 output
        #and then sum all of them together

        #x_scale = self.mlp_after(x[c_mask])
        #output[c_mask, :2] = torch.tanh(x_scale[:, :1]*x_food[c_mask] + 
        #                                x_scale[:, 1:2]*x_cell[c_mask, :2] + 
        #                                x_scale[:, 2:3]*x_wall[c_mask] + 
        #                                x_scale[:, 3:4]*x_x[c_mask]
        #                                )
        #output[c_mask, :2] = self.mlp_after(x[c_mask])
        x = output
        #x[:, :2] += self.gru(cell_edges, x[:, :2])

        #... and normalize hidden features H
        #h = x[c_mask, 2:] + x_origin[c_mask, 3:]
        x[c_mask, 2:] = self.nodeNorm(h)
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
        self.conv_layer_cell = self.conv_layer_cell.to(self.device)
        self.edge_conv_food = self.edge_conv_food.to(self.device)
        self.edge_conv_wall = self.edge_conv_wall.to(self.device)
        if self.model_type == ModelType.WithGlobalNode:
            self.conv_layer_global = self.conv_layer_global.to(self.device)
        return super().forward(*args)