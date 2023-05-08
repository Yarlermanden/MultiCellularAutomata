from typing import Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import BatchNorm1d, Linear
from torch import nn

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj, OptTensor, PairTensor

class CustomConv(MessagePassing):
    def __init__(self, channels: Union[int, Tuple[int, int]], dim: int = 0,
                aggr: str = 'add', batch_norm: bool = False,
                bias: bool = True, **kwargs):
        super().__init__(aggr=aggr, **kwargs)
        self.channels = channels
        self.dim = dim
        self.batch_norm = batch_norm

        if isinstance(channels, int):
            channels = (channels, channels)

        self.lin = Linear(sum(channels) + dim, channels[1], bias=bias)
        input = sum(channels) + dim
        self.mlp = nn.Sequential(
            Linear(input, input),
            nn.Tanh(),
            Linear(input, channels[1])
        )
        if batch_norm:
            self.bn = BatchNorm1d(channels[1])
        else:
            self.bn = None

        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        if self.bn is not None:
            self.bn.reset_parameters()

    def forward(self, x: Union[Tensor, PairTensor], edge_index: Adj,
                edge_attr: OptTensor = None) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            x: PairTensor = (x, x)

        # propagate_type: (x: PairTensor, edge_attr: OptTensor)
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=None)
        out = out if self.bn is None else self.bn(out)
        out = out + x[1] #removing this - removes the adding of xi
        return out

    def message(self, x_i, x_j, edge_attr: OptTensor) -> Tensor:
        if edge_attr is None:
            z = torch.cat([x_i, x_j], dim=-1)
        else:
            scale = 1/edge_attr[:, 0]
            edge_attr = scale.view(-1, 1)*edge_attr[:, 1:3]
            z = torch.cat([x_i, x_j, edge_attr], dim=-1)
        return torch.tanh(self.mlp(z))

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.channels}, dim={self.dim})'

class EdgeConv(MessagePassing):
    def __init__(self, channels: Union[int, Tuple[int, int]], dim: int = 0, **kwargs):
        super().__init__(aggr='mean', **kwargs)
        self.channels = channels
        self.dim = dim

        if isinstance(channels, int):
            channels = (channels, channels)
        self.reset_parameters()

        input = sum(channels)
        self.mlp = nn.Sequential(
            Linear(channels[0], channels[0]),
            nn.Tanh(),
            Linear(channels[0], channels[1])
        )

    def reset_parameters(self):
        ... 

    def forward(self, x: Union[Tensor, PairTensor], edge_index: Adj,
                edge_attr: OptTensor = None) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            x: PairTensor = (x, x)

        # propagate_type: (x: PairTensor, edge_attr: OptTensor)
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=None)
        return out

    def message(self, x_i, x_j, edge_attr: OptTensor) -> Tensor:
        z = edge_attr
        scale = 1/(z[:, 0]+0.0001)
        z = z[:, 1:3]*scale.view(-1,1)
        return torch.tanh(self.mlp(z))

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.channels}, dim={self.dim})'

class MeanEdgeConv(MessagePassing):
    def __init__(self, channels: Union[int, Tuple[int, int]], dim: int = 0, **kwargs):
        super().__init__(aggr='mean', **kwargs)
        self.channels = channels
        self.dim = dim

        if isinstance(channels, int):
            channels = (channels, channels)
        self.reset_parameters()

    def reset_parameters(self):
        ...

    def forward(self, x: Union[Tensor, PairTensor], edge_index: Adj,
                edge_attr: OptTensor = None) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            x: PairTensor = (x, x)

        # propagate_type: (x: PairTensor, edge_attr: OptTensor)
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=None)
        return out

    def message(self, x_i, x_j, edge_attr: OptTensor) -> Tensor:
        z = edge_attr
        scale = 1/z[:, 0]
        return torch.tanh(scale.view(-1, 1)*z[:, 1:3])

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.channels}, dim={self.dim})'