import dgl
from dgl.nn import GraphConv
from dgl import DGLGraph
import torch
import torch.nn as nn
import torch.nn.functional as F

from backend.object.graph import Graph

from logging import getLogger

logger = getLogger(__name__)
    
class GraphConvNetwork(nn.Module):    
    def __init__(self, graph: Graph):
        """initialize Graph Convolution layer with given params
        
        Args
        ----
        graph : Graph
            
        """
        super().__init__()
        self.graph = graph        
        
    def define_forward(self):
        input_dim = self.graph.graph.ndata['tensor'].shape[1]
        hidden_dim= 8
        output_dim= 1

        self.conv1 = GraphConv(input_dim, hidden_dim)
        self.conv2 = GraphConv(hidden_dim, output_dim)

    def forward(self, graph: DGLGraph, inputs):
        """define the forward pass of the model
        
        Args
        ----
        graph : DGLGraph
            input graph
        inputs : 
            input tensor
        """
        h = self.conv1(graph, inputs)
        h = F.relu(h)
        h = self.conv2(graph, h)
        return h
