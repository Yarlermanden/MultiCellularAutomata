from GNCAmodel import GNCA
from torch_geometric.nn import CGConv
import torch.nn as nn


class CGConv1(GNCA):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.conv_layers = CGConv(self.input_channels, dim=self.edge_dim, aggr='mean')

        self.mlp = nn.Sequential(
            nn.ReLU(), 
            nn.Linear(self.input_channels, self.input_channels),
            nn.ReLU(), 
            nn.Linear(self.input_channels, self.output_channels),
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
        h = self.conv_layers(x=h, edge_index=graph.edge_index, edge_attr=graph.edge_attr)
        h = self.mlp(h)
        h = h*2 - 1 #forces acceleration to be between -1 and 1 while using ReLU instead of Tanh
        return h
