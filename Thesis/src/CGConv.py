from GNCAmodel import GNCA
from torch_geometric.nn import CGConv
import torch.nn as nn
import torch


class CGConv1(GNCA):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.conv_layer_cells = CGConv(self.input_channels, dim=self.edge_dim)
        self.conv_layer_food = CGConv(self.input_channels, dim=self.edge_dim)

        self.mlp = nn.Sequential(
            nn.ReLU(), 
            nn.Linear(self.input_channels*2, self.input_channels*2),
            nn.ReLU(), 
            nn.Linear(self.input_channels*2, self.output_channels),
            nn.ReLU()
        )

        self.mlp_before = nn.Sequential(
            nn.Linear(self.input_channels, self.input_channels),
            nn.ReLU(),
            nn.Linear(self.input_channels, self.input_channels),
            nn.ReLU()
        )

    def message_pass(self, graph):
        x = graph.x.clone()
        mask = x[:, :2].abs() > 1.0
        x[:, :2] = x[:, :2] * mask
        h = self.mlp_before(x)

        cell_edges = graph.edge_index[:, torch.nonzero(graph.edge_attr[:, 3] == 1).flatten()]
        food_edges = graph.edge_index[:, torch.nonzero(graph.edge_attr[:, 3] == 0).flatten()]
        cell_attr = graph.edge_attr[torch.nonzero(graph.edge_attr[:, 3] == 1).flatten()]
        food_attr = graph.edge_attr[torch.nonzero(graph.edge_attr[:, 3] == 0).flatten()]
        

        h1 = self.conv_layer_cells(x=h, edge_index=cell_edges, edge_attr=cell_attr)
        h2 = self.conv_layer_food(x=h, edge_index=food_edges, edge_attr=food_attr)
        h = torch.concat((h1,h2), dim=1)
        h = self.mlp(h)
        h = h*2 - 1 #forces acceleration to be between -1 and 1 while using ReLU instead of Tanh
        return h
