import torch
import torch.nn as nn
import torch.nn.functional as F

from models.st_gcn.utils.tgcn import ConvTemporalGraphical
from models.st_gcn.utils.graph import Graph

from gt_pyg.nn.model import GraphTransformerNet
from gt_pyg.nn.gt_conv import GTConv
from torch_geometric.nn.conv import GATConv

from typing import List, Union, Optional, Dict, Any

from torch import Tensor
from torch_geometric.data import Batch
from torch_geometric.nn.aggr import MultiAggregation

from graph_transformer_pytorch import GraphTransformer

import time



class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.hidden_layer = nn.Linear(input_dim, hidden_dim)  # Hidden layer
        self.activation = nn.ReLU()                          # Non-linear activation
        self.output_layer = nn.Linear(hidden_dim, output_dim) # Output layer

    def forward(self, x):
        x = self.hidden_layer(x)  # Pass through hidden layer
        x = self.activation(x)    # Apply activation
        x = self.output_layer(x)  # Pass through output layer
        return x

class GraphTransformerEmbedding(nn.Module):
    r"""
    Graph Transformer-based embedding model for skeleton data.

    Args:
        in_channels (int): Number of input channels
        graph_args (dict): Arguments to build the graph
        edge_importance_weighting (bool): If True, learnable weights are added to graph edges
        num_heads (int): Number of attention heads in transformer layers
        embedding_layer_size (int): Dimension of the final embedding layer
        hidden_dim (int): Dimension of hidden layers
        num_layers (int): Number of transformer layers
    """
    def __init__(self, node_in_dim=3, hidden_dim=32, num_gt_layers=4, heads = 8):

        super(GraphTransformerEmbedding, self).__init__()      
 
        # Load graph
        A = torch.tensor([[0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                        [1., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                        [1., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                        [0., 1., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                        [0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                        [0., 0., 0., 1., 0., 0., 1., 1., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
                        [0., 0., 0., 0., 1., 1., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 0.],
                        [0., 0., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
                        [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
                        [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                        [0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
                        [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0.],
                        [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 1., 0., 0.],
                        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 1., 0.],
                        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 1.],
                        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
                        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.]])#.to_sparse()

        # A = torch.tensor(neighbor_base, dtype=torch.long, requires_grad=False)
        self.register_buffer('A', A)

        self.gt = GraphTransformer(
                    dim = hidden_dim,
                    depth = num_gt_layers,
                    dim_head = hidden_dim,
                    edge_dim = 1,             # optional - if left out, edge dimensions is assumed to be the same as the node dimensions above
                    with_feedforwards = True,   # whether to add a feedforward after each attention layer, suggested by literature to be needed
                    gated_residual = True,      # to use the gated residual to prevent over-smoothing
                    rel_pos_emb = True          # set to True if the nodes are ordered, default to False
                )

        self.nodes_emb = nn.Linear(3, hidden_dim)
        self.final_embedding = MLP(hidden_dim, hidden_dim*2, hidden_dim)



    def forward(self, x):

        B, _, C, T, N = x.size()  # Extract dimensions

        # Reshape input to (B * T, V, C) for processing
        x = x.permute(0, 3, 4, 2, 1).reshape(B * T, N, C)  # (B*T, V, C)


        edges = self.A.repeat(x.shape[0], 1, 1, 1).permute(0, 2, 3, 1)
        mask = torch.ones(x.shape[0], 17).bool().to("cuda")

        x = self.nodes_emb(x)

        out = self.gt (x, edges, mask = mask)

        x = out.view(B, T, N, -1).mean(dim=2)

        x = x.mean(dim=1)  # (B, hidden_dim)

        feature = self.final_embedding(x)

        return feature



    def get_embedding(self, x):
        return self.forward(x)


class STGCNEmbedding(nn.Module):
    r"""Spatial temporal graph convolutional networks.

    Args:
        in_channels (int): Number of channels in the input data
        graph_args (dict): The arguments for building the graph
        edge_importance_weighting (bool): If ``True``, adds a learnable
            importance weighting to the edges of the graph
        **kwargs (optional): Other parameters for graph convolution units

    Shape:
        - Input: :math:`(N, in_channels, T_{in}, V_{in})`
        - Output: :math:`(N, num_class)` where
            :math:`N` is a batch size,
            :math:`T_{in}` is a length of input sequence,
            :math:`V_{in}` is the number of graph nodes
    """

    def __init__(self, in_channels, graph_args, edge_importance_weighting=False, temporal_kernel_size=9,
                 embedding_layer_size=256, **kwargs):
        super().__init__()

        # load graph
        self.graph = Graph(**graph_args)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)

        # build networks
        spatial_kernel_size = A.size(0)
        # temporal_kernel_size = 9

        kernel_size = (temporal_kernel_size, spatial_kernel_size)

        self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))
        kwargs0 = {k: v for k, v in kwargs.items() if k != 'dropout'}

        self.st_gcn_networks = nn.ModuleList((
            st_gcn(in_channels, 64, kernel_size, 1, residual=False, **kwargs0),
            st_gcn(64, 64, kernel_size, 1, **kwargs),
            st_gcn(64, 64, kernel_size, 1, **kwargs),
            st_gcn(64, 64, kernel_size, 1, **kwargs),
            st_gcn(64, 128, kernel_size, 2, **kwargs),
            st_gcn(128, 128, kernel_size, 1, **kwargs),
            st_gcn(128, 128, kernel_size, 1, **kwargs),
            st_gcn(128, 256, kernel_size, 2, **kwargs),
            # st_gcn(256, 256, kernel_size, 1, **kwargs),
            st_gcn(256, 256, kernel_size, 1, **kwargs),
        ))

        # initialize parameters for edge importance weighting
        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones(self.A.size()), requires_grad=True)
                for _ in self.st_gcn_networks
            ])
        else:
            self.edge_importance = [1] * len(self.st_gcn_networks)

        self.fcn = nn.Conv2d(256, embedding_layer_size, kernel_size=1)

    def forward(self, x, hint=None):
        # data normalization
        N, C, T, V = x.size()
        x = x.permute(0, 3, 1, 2).contiguous()
        x = x.view(N, V * C, T)

        x = self.data_bn(x)
        x = x.view(N, V, C, T)

        x = x.permute(0, 2, 3, 1).contiguous()
        x = x.view(N, C, T, V)
        # forward
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A * importance)

        # Adding average pooling as in the original model
        x = F.avg_pool2d(x, x.size()[2:])

        feature = self.fcn(x)

        # L2 normalization
        feature = F.normalize(feature, dim=1, p=2)

        feature = feature.view(N, -1)

        return feature

    # Alias for model.forward()
    def get_embedding(self, x):
        return self.forward(x)


class st_gcn(nn.Module):
    r"""Applies a spatial temporal graph convolution over an input graph sequence.

    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (tuple): Size of the temporal convolving kernel and graph convolving kernel
        stride (int, optional): Stride of the temporal convolution. Default: 1
        dropout (int, optional): Dropout rate of the final output. Default: 0
        residual (bool, optional): If ``True``, applies a residual mechanism. Default: ``True``

    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Output graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format

        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.

    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dropout=0,
                 residual=True):
        super().__init__()

        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0] - 1) // 2, 0)

        self.gcn = ConvTemporalGraphical(in_channels, out_channels,
                                         kernel_size[1])

        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                (kernel_size[0], 1),
                (stride, 1),
                padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )

        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, A):

        res = self.residual(x)
        x, A = self.gcn(x, A)
        x = self.tcn(x) + res

        return self.relu(x), A
