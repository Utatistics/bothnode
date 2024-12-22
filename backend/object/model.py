import dgl
from dgl import DGLGraph
from dgl.nn import GraphConv
from dgl.nn import SAGEConv
from dgl.dataloading import MultiLayerNeighborSampler

import torch
import torch.nn as nn
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

class GraphSAGE(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats, num_layers, aggregator_type='mean'):
        super(GraphSAGE, self).__init__()
        self.layers = nn.ModuleList()
        # Input layer
        self.layers.append(SAGEConv(in_feats, hidden_feats, aggregator_type))
        # Hidden layers
        for _ in range(num_layers - 2):
            self.layers.append(SAGEConv(hidden_feats, hidden_feats, aggregator_type))
        # Output layer
        self.layers.append(SAGEConv(hidden_feats, out_feats, aggregator_type))

    def forward(self, graph, features):
        h = features
        for layer in self.layers:
            h = layer(graph, h)
            h = F.relu(h)
        return h

class UnsupervisedGraphSAGE:
    def __init__(self, graph, in_feats, hidden_feats, out_feats, num_layers):
        self.graph = graph
        self.model = GraphSAGE(in_feats, hidden_feats, out_feats, num_layers)
        self.classifier = NodePairClassifier(out_feats)
        self.optimizer = torch.optim.Adam(
            list(self.model.parameters()) + list(self.classifier.parameters()),
            lr=0.01 
        )

    def generate_node_pairs(self, num_walks=10, walk_length=5, neg_samples=5):
        """Generate positive and negative node pairs."""
        positive_pairs = []
        for _ in range(num_walks):
            for node in range(self.graph.num_nodes()):
                walk = dgl.sampling.random_walk(self.graph, [node], length=walk_length)[0][0]
                for i in range(len(walk) - 1):
                    positive_pairs.append((walk[i].item(), walk[i + 1].item()))

        # Generate negative pairs
        neg_nodes = torch.arange(self.graph.num_nodes())
        negative_pairs = [
            (np.random.choice(neg_nodes), np.random.choice(neg_nodes)) for _ in range(len(positive_pairs) * neg_samples)
        ]

        return positive_pairs, negative_pairs

    def train(self, features, epochs=10, batch_size=64, neg_samples=5):
        """Train the unsupervised GraphSAGE model."""
        self.model.train()
        self.classifier.train()

        for epoch in range(epochs):
            positive_pairs, negative_pairs = self.generate_node_pairs(neg_samples=neg_samples)
            all_pairs = positive_pairs + negative_pairs
            labels = torch.cat(
                [torch.ones(len(positive_pairs)), torch.zeros(len(negative_pairs))]
            ).to(torch.float32)

            # Shuffle pairs
            indices = torch.randperm(len(all_pairs))
            all_pairs = [all_pairs[i] for i in indices]
            labels = labels[indices]

            # Minibatch training
            total_loss = 0
            for i in range(0, len(all_pairs), batch_size):
                batch_pairs = all_pairs[i:i + batch_size]
                batch_labels = labels[i:i + batch_size]

                u, v = zip(*batch_pairs)
                u = torch.tensor(u)
                v = torch.tensor(v)

                # Forward pass
                u_emb = self.model(self.graph, features)[u]
                v_emb = self.model(self.graph, features)[v]
                predictions = self.classifier(u_emb, v_emb).squeeze()

                # Compute loss
                loss = F.binary_cross_entropy(predictions, batch_labels)
                total_loss += loss.item()

                # Backward pass and optimization
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            logger.info(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(all_pairs):.4f}")

class NodePairClassifier(nn.Module):
    def __init__(self, embedding_dim):
        super(NodePairClassifier, self).__init__()
        self.fc = nn.Linear(embedding_dim * 2, 1)  # Concatenated embeddings as input

    def forward(self, u_emb, v_emb):
        pair_emb = torch.cat([u_emb, v_emb], dim=1)  # Concatenate node embeddings
        return torch.sigmoid(self.fc(pair_emb))  # Binary classification output
