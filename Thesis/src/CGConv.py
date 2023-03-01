from GNCAmodel import GNCA
from torch_geometric.nn import CGConv
from torch_geometric_temporal.nn.recurrent import gconv_gru
import torch.nn as nn
import torch


class CGConv1(GNCA):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.input_channels = 2
        self.output_channels = 2
        self.hidden_size = self.input_channels*2
        self.conv_layer_cells = CGConv(self.hidden_size, dim=self.edge_dim, aggr='mean')
        self.conv_layer_food = CGConv(self.hidden_size, dim=self.edge_dim, aggr='mean')

        self.mlp_before = nn.Sequential(
            #nn.Linear(self.input_channels, self.input_channels*2),
            #nn.ReLU(),
            nn.Linear(self.input_channels, self.hidden_size),
            nn.Tanh(),
        )

        self.mlp_middle = nn.Sequential(
            nn.Tanh(),
            nn.Linear(self.hidden_size*2, self.hidden_size),
            nn.Tanh(),
        )

        self.mlp = nn.Sequential(
            nn.Tanh(), 
            #nn.Linear(self.hidden_size, self.input_channels*2),
            #nn.ReLU(), 
            #nn.Linear(self.input_channels*2, self.input_channels*2),
            #nn.ReLU(), 
            #nn.Linear(self.input_channels*2, self.output_channels),
            nn.Linear(self.hidden_size, self.output_channels),
            nn.Tanh()
        )

        #self.lstm1 = nn.LSTM(input_size=self.input_channels*2, hidden_size=self.input_channels*2, num_layers=1)
        #self.rnn = nn.RNN(input_size=self.input_channels*8, hidden_size=self.hidden_size)
        self.gConvGRU = gconv_gru.GConvGRU(in_channels=self.hidden_size, out_channels=self.hidden_size, K=1).to(self.device)
        self.H = None
        #self.C = None

    def message_pass(self, graph):
        #x = self.mlp_before(torch.concat((graph.x[:, 2:4], graph.x[:, 5:7]), dim=1))
        x = self.mlp_before(graph.x[:, 2:4])

        #if self.H is None:
            #self.H = torch.zeros(1, self.hidden_size)

        #if self.node_indices_to_keep is not None:
        #    self.H = self.H[self.node_indices_to_keep].view(self.node_indices_to_keep.shape[0], self.H.shape[1])
        #    self.C = self.C[self.node_indices_to_keep].view(self.node_indices_to_keep.shape[0], self.C.shape[1])

            #self.H = torch.cat((self.H.unsqueeze(dim=2), self.H[:, :, :self.periods-1]), dim=2)
            #self.X = torch.cat((x.unsqueeze(dim=2), self.X[:, :, :self.periods-1]), dim=2)

        cell_edges = graph.edge_index[:, torch.nonzero(graph.edge_attr[:, 3] == 1).flatten()]
        food_edges = graph.edge_index[:, torch.nonzero(graph.edge_attr[:, 3] == 0).flatten()]
        cell_attr = graph.edge_attr[torch.nonzero(graph.edge_attr[:, 3] == 1).flatten()]
        food_attr = graph.edge_attr[torch.nonzero(graph.edge_attr[:, 3] == 0).flatten()]
        
        x1 = self.conv_layer_cells(x=x, edge_index=cell_edges, edge_attr=cell_attr)
        x2 = self.conv_layer_food(x=x, edge_index=food_edges, edge_attr=food_attr)

        x = torch.concat((x1,x2), dim=1)
        x = self.mlp_middle(x)

        if self.H is None:
            self.H = torch.zeros_like(x, device=self.device)
        if self.node_indices_to_keep is not None:
            self.H = self.H[self.node_indices_to_keep].view(self.node_indices_to_keep.shape[0], self.H.shape[1])

        #x, self.H = self.rnn(x, self.H)
        self.H = self.gConvGRU(x, graph.edge_index, H=self.H)

        x = self.mlp(self.H)
        return x

    def forward(self, *args):
        self.H = None
        #self.C = None
        self.node_indices_to_keep = None
        self.mlp_before = self.mlp_before.to(self.device)
        self.mlp_middle = self.mlp_middle.to(self.device)
        self.mlp = self.mlp.to(self.device)
        #self.conv_layer_cells = self.conv_layer_cells.to(self.device)
        #self.conv_layer_food = self.conv_layer_food.to(self.device)
        return super().forward(*args)