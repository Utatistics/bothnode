import torch
import dgl
import torch.nn as nn
import torch.optim as optim
import numpy as np


from backend.util.config import Config
from logging import getLogger

logger = getLogger(__name__)

config = Config()

import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Tuple
import numpy as np
import logging

logger = logging.getLogger(__name__)

class Node2Vec(nn.Module):
    def __init__(self, num_nodes: int, embedding_dim: int) -> None:
        """
        Initialize Node2Vec model.
        
        Args
        ----
        num_nodes : int 
            Number of nodes in the graph.
        embedding_dim : int
            Dimension of the node embeddings.
        """
        super(Node2Vec, self).__init__()
        self.node_embeddings = nn.Embedding(num_nodes, embedding_dim)

    def forward(self, nodes: torch.Tensor) -> torch.Tensor:
        """
        Get embeddings for a batch of nodes.
        
        Args
        ----
        nodes : torch.Tensor
            Tensor of node IDs
        
        Returns
        -------
            torch.Tensor: Node embeddings.
        """
        return self.node_embeddings(nodes)

    def get_similarity(self, node_u: torch.Tensor, node_v: torch.Tensor) -> torch.Tensor:
        """
        Compute similarity score (i.e. dot product) between two nodes.
        
        Args
        ----
        node_u : torch.Tensor)
            Tensor of source node IDs.
        node_v : torch.Tensor): Tensor of target node IDs.
        
        Returns
        -------
        torch.Tensor:
            Similarity scores.
        """
        emb_u = self.node_embeddings(node_u)
        emb_v = self.node_embeddings(node_v)
        return torch.sum(emb_u * emb_v, dim=-1)

    def _biased_random_walk(self, graph: dgl.DGLGraph, nodes: torch.Tensor, walk_length: int, p: float, q: float) -> List[List[int]]:
        """
        Perform the randarm walk strategy for Node2Vec (i.e. 2nd order biased r.w.) on the given graph.
        
        Args
        ----
        graph: dgl.DGLGraph
            Graph object with a `successors` method.
        nodes : torch.Tensor
            Starting nodes for the random walks.
        walk_length : int
            Length of each walk.
        p : float
            Return hyperparameter.
        q : float
            In-out hyperparameter.
        
        Returns
        -------
        List[List[int]]: List of random walks.
        """
        walks = []
        for start_node in nodes:
            walk = [start_node.item()]
            while len(walk) < walk_length:
                cur = walk[-1]
                neighbors = list(graph.successors(cur).numpy())
                if len(neighbors) == 0:
                    break
                if len(walk) == 1:
                    next_node = np.random.choice(neighbors)
                else:
                    prev = walk[-2]
                    prob = []
                    for neighbor in neighbors:
                        if neighbor == prev:
                            prob.append(1 / p)
                        elif graph.has_edges_between(prev, neighbor):
                            prob.append(1)
                        else:
                            prob.append(1 / q)
                    prob = np.array(prob) / sum(prob)
                    next_node = np.random.choice(neighbors, p=prob)
                walk.append(next_node)
            walks.append(walk)
        return walks

    def train_node2vec(
        self,
        graph: dgl.DGLGraph,
        walk_length: int,
        num_walks: int,
        window_size: int,
        p: float,
        q: float,
        epochs: int,
        learning_rate: float,
    ) -> None:
        """
        Train Node2Vec embeddings using random walks and skip-gram.
        
        Args
        ----
        graph : dgl.DGLGraph
            Graph object with a `num_nodes` method.
        walk_length : int
            Length of each walk.
        num_walks : int
            Number of walks per node.
        window_size : int 
            Context window size for skip-gram.
        p : float
            Return hyperparameter.
        q : float
            In-out hyperparameter.
        epochs : int
            Number of training epochs.
        learning_rate : float
            Learning rate for optimization.
        """
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        # Generate random walks
        all_walks = []
        for _ in range(num_walks):
            start_nodes = torch.arange(graph.num_nodes())
            walks = self._biased_random_walk(graph, start_nodes, walk_length, p, q)
            all_walks.extend(walks)

        # Convert walks to skip-gram pairs
        skip_gram_pairs = []
        for walk in all_walks:
            for i, target in enumerate(walk):
                for j in range(max(0, i - window_size), min(len(walk), i + window_size + 1)):
                    if i != j:
                        skip_gram_pairs.append((target, walk[j]))
        skip_gram_pairs = torch.tensor(skip_gram_pairs)

        # Training loop
        for epoch in range(epochs):
            self.train()
            optimizer.zero_grad()

            u, v = skip_gram_pairs[:, 0], skip_gram_pairs[:, 1]
            u_emb, v_emb = self(u), self(v)
            scores = torch.sum(u_emb * v_emb, dim=1)
            loss = -torch.log(torch.sigmoid(scores)).mean()

            loss.backward()
            optimizer.step()

            logger.info(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")

    def compute_similarity_matrix(self) -> torch.Tensor:
        """
        Compute pairwise similarity scores for all nodes.
        
        Returns:
            torch.Tensor: Pairwise similarity matrix.
        """
        embeddings = self.node_embeddings.weight
        similarity_matrix = embeddings @ embeddings.T
        return similarity_matrix
