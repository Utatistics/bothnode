import dgl
from dgl import DGLGraph
from dgl.nn import GraphConv
from dgl.nn import SAGEConv
from dgl.dataloading import MultiLayerNeighborSampler

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

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
    
class GraphSAGE(nn.Module):
    def __init__(self, in_feats: int, hidden_feats: int, out_feats: int) -> None:
        """
        Initialize the GraphSAGE model with two SAGEConv layers.

        Args
        ----
        in_feats : int
            Number of input features for each node.
        hidden_feats : int
            Number of hidden units in the first SAGEConv layer.
        out_feats : int
            Number of output features for the final layer (embedding dimension).
        """
        super().__init__()
        self.input_dim = in_feats
        self.hidden_dim = hidden_feats
        self.output_dim = out_feats
        
        self.layer1 = dgl.nn.SAGEConv(in_feats, hidden_feats, 'mean')  # First GraphSAGE layer with mean aggregation.
        self.bn1 = nn.BatchNorm1d(hidden_feats)  # Batch normalization for layer 1
        self.layer2 = dgl.nn.SAGEConv(hidden_feats, out_feats, 'mean')  # Second GraphSAGE layer with mean aggregation.
        self.bn2 = nn.BatchNorm1d(out_feats)  # Batch normalization for layer         
        
        self._weights_initializer()

    def _weights_initializer(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                logger.info(f'{m=}: Applying Xavier Initializer...')
                init.xavier_uniform_(m.weight)  # Xavier initialization
                if m.bias is not None:
                    init.zeros_(m.bias)  # Bias initialized to 0


    def forward(self, graph: dgl.DGLGraph, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to compute node embeddings.

        Args
        ----
        graph : dgl.DGLGraph
            The input graph.
        features : torch.Tensor
            Node features (e.g., one-hot encoding or predefined embeddings).

        Returns
        -------
        torch.Tensor
            Final node embeddings of shape (num_nodes, out_feats).
        """
        h = self.layer1(graph, features) 
        h = self.bn1(h)
        h = torch.relu(h)
        h = self.layer2(graph, h)
        h = self.bn2(h)

        return h
    
    def learn_embedding(self, graph: dgl.DGLGraph, features: torch.Tensor, labels: torch.Tensor, epochs: int=20, learning_rate: float=0.01) -> torch.Tensor:
        """ 
        Train GraphSAGE to learn node embeddings with similarity matrix as labels.

        Args
        ----
        graph : dgl.DGLGraph
            Input graph.
        labels : torch.Tensor
            Similarity matrix as labels.
        features : torch.Tensor, optional
            Node features. If None, uses one-hot encoding.
        epochs : int, optional
            Number of training epochs.
        learning_rate : float, optional
            Learning rate for optimizer.

        Returns
        -------
        torch.Tensor
            Learned embeddings.
        """

        # Calculate total number of parameters
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"Total parameters: {total_params}")
        logger.info(f"Trainable parameters: {trainable_params}")

        num_nodes = graph.num_nodes()
        if features is None:
            features = torch.eye(num_nodes) # one-hot encoding (i.e. identiry matrix) if features are not provided

        # Training loop
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()

        for epoch in range(epochs):
            self.train()
            optimizer.zero_grad()

            embeddings = self(graph, features) # callable: delegate to 'forward'
            
            #pred_similarity = embeddings @ embeddings.T  # Dot product for predicted similarity matrix
            pred_similarity = torch.sum(embeddings.unsqueeze(1) * embeddings, dim=2) 

            # Compute loss
            loss = criterion(pred_similarity, labels)
            loss.backward(retain_graph=True)
            
            for name, param in self.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.norm(2).item()  # Compute L2 norm of the gradient
                    logger.warning(f"Gradient norm for {name}: {grad_norm}")
            
            optimizer.step()

            logger.info(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")

        return embeddings.detach()
    
