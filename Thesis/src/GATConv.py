from GNCAmodel import GNCA
from torch_geometric.nn import GATv2Conv
import torch.nn as nn

class GATConv(GNCA):
    def __init__(self, device, channels=5):
        super().__init__(device, channels)
        self.conv_layers = GATv2Conv(channels, channels, heads=1, edge_dim=2)
        self.mlp = nn.Sequential(
            nn.ReLU(), 
            nn.Linear(self.input_channels, self.input_channels),
            nn.ReLU(), 
            nn.Linear(self.input_channels, self.output_channels),
            nn.ReLU())

    def message_pass(self, graph):
        '''Convolves the graph for message passing'''
        h = self.conv_layers(x=graph.x, edge_index=graph.edge_index, edge_attr=graph.edge_attr)
        h = self.mlp(h)
        h = h*2 - 1 #forces acceleration to be between -1 and 1 while using ReLU instead of Tanh
        return h