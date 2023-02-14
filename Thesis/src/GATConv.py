from GNCAmodel import GNCA
from torch_geometric.nn import GATv2Conv
import torch.nn as nn

class GATConv(GNCA):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.conv_layers = GATv2Conv(self.input_channels, self.input_channels, heads=1, concat=False, edge_dim=self.edge_dim, add_self_loops=False)
        self.mlp = nn.Sequential(
            nn.ReLU(), 
            nn.Linear(self.input_channels, self.input_channels),
            nn.ReLU(), 
            nn.Linear(self.input_channels, self.output_channels),
            nn.ReLU())

    def message_pass(self, graph):
        '''Convolves the graph for message passing'''
        x = graph.x.clone()
        mask = x[:, :2].abs() > 0.9
        x[:, :2] = x[:, :2] * mask
        h = self.conv_layers(x=x, edge_index=graph.edge_index, edge_attr=graph.edge_attr)
        h = self.mlp(h)
        h = h*2 - 1 #forces acceleration to be between -1 and 1 while using ReLU instead of Tanh
        return h