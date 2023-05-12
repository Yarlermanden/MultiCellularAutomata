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
            #Linear(edge_dim, edge_dim, weight_initializer='glorot'),
            Linear(6, 6),
            torch.nn.Tanh(),
            #Linear(edge_dim, out_channels, weight_initializer='glorot')
            Linear(6, 1),
            torch.nn.Tanh(),
        )
        self.lin = Linear(1, 1)

        self.register_parameter('bias', None)
        self._alpha = None
        self.reset_parameters()

    def reset_parameters(self):
        ...
        #self.mlp_edge[0].reset_parameters()
        #self.mlp_edge[2].reset_parameters()
        #glorot(self.att)

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
        #x_i is the cell
        edge_attr = edge_attr.view(-1, 3)
        input = torch.cat((edge_attr[:, :1], x_i[:, -5:]), dim=1)

        #scale = 1/(edge_attr[:, :1]+0.0001)
        #x = torch.concat([edge_attr[:, :1], scale*edge_attr[:, 1:3]], dim=1)
        #x = torch.abs(x)
        #mij = self.mlp_edge(x).view(-1, 1, self.out_channels)
        mij = self.mlp_edge(input)

        x = self.lin(mij).unsqueeze(dim=1)
        x = F.leaky_relu(x, self.negative_slope)
        alpha = (x * self.att).sum(dim=-1)
        alpha = softmax(alpha, index, ptr, size_i)
        self._alpha = alpha
        x = (mij * alpha).view(-1, 1) * edge_attr[:, 1:3]
        return x.unsqueeze(dim=1)

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

        self.att = Parameter(torch.Tensor(1, 1, 6))

        input = in_channels*2+1
        self.mlp = torch.nn.Sequential(
            #Linear(input, input, weight_initializer='glorot'),
            Linear(input, input),
            torch.nn.Tanh(),
            #Linear(input, out_channels, weight_initializer='glorot')
            Linear(input, 6),
            torch.nn.Tanh(),
        )
        self.lin = Linear(6, 6)

        self.register_parameter('bias', None)

        self._alpha = None

        self.reset_parameters()

    def reset_parameters(self):
        #self.mlp[0].reset_parameters()
        #self.mlp[2].reset_parameters()
        #glorot(self.att)
        ...

    def forward(self, x: Union[Tensor, PairTensor], edge_index: Adj,
                edge_attr: OptTensor = None):
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr,
                             size=None)
        out = out.view(-1, self.out_channels)
        #TODO should we add conv on x_i itself on top of this?
        if torch.any(torch.isnan(out)):
            print('GATCONV is nan')
            out = torch.zeros_like(out)

        self._alpha = None
        return out

    def message(self, x_j: Tensor, x_i: Tensor, edge_attr: OptTensor,
                index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:
        edge_attr = edge_attr.view(-1, 3)
        #scale = 1/(edge_attr[:, :1]+0.0001)
        #edge_attr1 = torch.cat([edge_attr[:, :1], scale*edge_attr[:, 1:3]], dim=1)
        #edge_attr1 = torch.abs(edge_attr1)
        #z = torch.cat([x_i, x_j, edge_attr1], dim=1)
        z = torch.cat([x_i, x_j, edge_attr[:, :1]], dim=1)

        mij = self.mlp(z)
        if torch.any(torch.isnan(mij)):
            print("mlp return nan")
            print('z contain nan: ', torch.any(torch.isnan(z)))
            #print('scale contains nan: ' + torch.any(torch.isnan(scale)))
            print('x_i contains nan: ', torch.any(torch.isnan(x_i)))
            print('x_j contains nan: ', torch.any(torch.isnan(x_j)))
            print(edge_attr)
            #print(scale)
            print(x_i)
            print(x_j)

        x = self.lin(mij).unsqueeze(dim=1)
        x = F.leaky_relu(x, self.negative_slope)
        alpha = (x * self.att).sum(dim=-1)
        alpha = softmax(alpha, index, ptr, size_i)
        self._alpha = alpha

        x = mij * alpha
        x = torch.cat((x[:, :1] * edge_attr[:, 1:3], x[:, 1:]), dim=1)
        return x.unsqueeze(dim=1)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={1})')
