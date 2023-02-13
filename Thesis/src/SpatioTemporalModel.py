from GNCAmodel import GNCA
import torch.nn as nn
import torch
from torch_geometric_temporal.nn.recurrent.attentiontemporalgcn import A3TGCN

class SpatioTemporal(GNCA):
    def __init__(self, device, channels=5):
        super().__init__(device, channels)
        self.periods = 10
        self.conv_layers = A3TGCN(channels, channels, periods=self.periods)
        self.mlp = nn.Sequential(
            nn.Linear(self.input_channels, self.input_channels),
            nn.ReLU(), 
            nn.Linear(self.input_channels, self.output_channels),
            nn.ReLU()
            )
        self.h = None
        self.X = None

    def message_pass(self, graph):
        '''Convolves the graph for message passing'''
        if self.X is None:
            zero = torch.zeros_like(graph.x)
            self.X = torch.cat((graph.x.unsqueeze(dim=2), torch.zeros(*graph.x.shape, self.periods-1)), dim=2)
            self.h = zero.clone()

        if self.node_indices_to_keep is not None:
            self.X = self.X[self.node_indices_to_keep].view(self.node_indices_to_keep.shape[0], self.X.shape[1], self.periods)
            self.X = torch.cat((graph.x.unsqueeze(dim=2), self.X[:, :, :self.periods-1]), dim=2)
            self.h = self.h[self.node_indices_to_keep].view(self.node_indices_to_keep.shape[0], self.h.shape[1])

        #h = self.conv_layers(x=graph.x, edge_index=graph.edge_index, edge_attr=graph.edge_attr)
        h = self.conv_layers(X=self.X, edge_index=graph.edge_index, H=self.h)
        self.h = h

        x = self.mlp(h)
        x = x*2 - 1 #forces acceleration to be between -1 and 1 while using ReLU instead of Tanh
        return x

    def forward(self, *args):
        self.h = None
        self.X = None
        self.node_indices_to_keep = None
        return super().forward(*args)