import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GraphConv

from logging import getLogger

logger = getLogger(__name__)
    
# Define a simple GCN model
class GraphConvNetwork(nn.Module):    
    def __init__(self, in_feats, hidden_feats, out_feats):
        super(self).__init__()
        self.conv1 = GraphConv(in_feats, hidden_feats)
        self.conv2 = GraphConv(hidden_feats, out_feats)

    def forward(self, graph, inputs):
        h = self.conv1(graph, inputs)
        h = F.relu(h)
        h = self.conv2(graph, h)
        return h
