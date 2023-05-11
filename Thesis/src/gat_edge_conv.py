from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter
from torch_sparse import SparseTensor, set_diag

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.typing import Adj, OptTensor, PairTensor
from torch_geometric.utils import add_self_loops, remove_self_loops, softmax

class GATEdgeConv(MessagePassing):
    _alpha: OptTensor

    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        negative_slope: float = 0.2,
        edge_dim: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.negative_slope = negative_slope
        self.edge_dim = edge_dim

        self.att = Parameter(torch.Tensor(1, 1, out_channels))
        self.mlp_edge = torch.nn.Sequential(
        #    Linear(edge_dim, edge_dim, weight_initializer='glorot'),
        #    torch.nn.Tanh(),
            Linear(edge_dim, out_channels, weight_initializer='glorot')
        )

        self.register_parameter('bias', None)
        self._alpha = None
        self.reset_parameters()

    def reset_parameters(self):
        self.mlp_edge[0].reset_parameters()
        #self.mlp_edge[2].reset_parameters()
        glorot(self.att)

    def forward(self, x: Union[Tensor, PairTensor], edge_index: Adj,
                edge_attr: OptTensor = None):

        if isinstance(x, Tensor):
            x: PairTensor = (x, x)

        out = self.propagate(edge_index, x=x, edge_attr=edge_attr,
                             size=None)
        self._alpha = None

        out = out.view(-1, self.out_channels)
        if torch.any(torch.isnan(out)):
            print('GATEdgeConv is nan')
            out = torch.zeros_like(out)

        return out

    def message(self, x_i, x_j, edge_attr: OptTensor,
                index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:
        edge_attr = edge_attr.view(-1, 3)

        scale = 1/(edge_attr[:, :1]+0.001)
        x = torch.concat([edge_attr[:, :1], scale*edge_attr[:, 1:3]], dim=1)
        x = self.mlp_edge(x).view(-1, 1, self.out_channels)

        x = F.leaky_relu(x, self.negative_slope)
        alpha = (x * self.att).sum(dim=-1)
        alpha = softmax(alpha, index, ptr, size_i)
        self._alpha = alpha
        return x * alpha.unsqueeze(-1)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={1})')


class GATConv(MessagePassing):
    _alpha: OptTensor

    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        negative_slope: float = 0.2,
        add_self_loops: bool = True,
        edge_dim: Optional[int] = None,
        fill_value: Union[float, Tensor, str] = 'mean',
        **kwargs,
    ):
        super().__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.negative_slope = negative_slope
        self.add_self_loops = add_self_loops
        self.edge_dim = edge_dim
        self.fill_value = fill_value

        self.att = Parameter(torch.Tensor(1, 1, out_channels))

        input = in_channels*2+edge_dim
        self.mlp = torch.nn.Sequential(
            #Linear(input, input, weight_initializer='glorot'),
            #torch.nn.Tanh(),
            Linear(input, out_channels, weight_initializer='glorot')
        )

        self.register_parameter('bias', None)

        self._alpha = None

        self.reset_parameters()

    def reset_parameters(self):
        self.mlp[0].reset_parameters()
        #self.mlp[2].reset_parameters()
        glorot(self.att)

    def forward(self, x: Union[Tensor, PairTensor], edge_index: Adj,
                edge_attr: OptTensor = None):

        out = self.propagate(edge_index, x=x, edge_attr=edge_attr,
                             size=None)
        out = out.view(-1, self.out_channels)
        if torch.any(torch.isnan(out)):
            print('GATCONV is nan')
            out = torch.zeros_like(out)

        self._alpha = None
        return out

    def message(self, x_j: Tensor, x_i: Tensor, edge_attr: OptTensor,
                index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:
        edge_attr = edge_attr.view(-1, 3)
        scale = 1/(edge_attr[:, :1]+0.001)
        edge_attr = torch.cat([edge_attr[:, :1], scale*edge_attr[:, 1:3]], dim=1)
        z = torch.cat([x_i, x_j, edge_attr], dim=1)

        x = self.mlp(z).view(-1, 1, self.out_channels)
        x = F.leaky_relu(x, self.negative_slope)
        alpha = (x * self.att).sum(dim=-1)
        alpha = softmax(alpha, index, ptr, size_i)
        self._alpha = alpha
        return x * alpha.unsqueeze(-1)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={1})')
