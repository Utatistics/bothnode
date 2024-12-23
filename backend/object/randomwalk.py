import torch
import dgl
import torch.nn as nn
import torch.optim as optim
import numpy as np

from backend.util.config import Config

import logging
from logging import getLogger
from typing import List, Tuple

logger = logging.getLogger(__name__)

class Node2Vec(nn.Module):
    def __init__(self, num_nodes: int, embedding_dim: int) -> None:
        """
        Initialize Node2Vec model as a single Embedding layer.
        
        Args
        ----
        num_nodes : int 
            Number of nodes in the graph.
        embedding_dim : int
            Dimension of the node embeddings.
        """
        super().__init__()
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

    def _biased_random_walk(self, graph: dgl.DGLGraph, init_nodes: torch.Tensor, walk_length: int, p: float, q: float) -> List[List[int]]:
        """
        Perform the randarm walk strategy for Node2Vec (i.e. 2nd order biased r.w.), starting from each node on graph 

        Args
        ----
        graph: dgl.DGLGraph
            Graph object with a `successors` method.
        init_nodes : torch.Tensor
            Starting nodes for the random walks (i.e. all nodes)
        walk_length : int
            Length of each walk.
        p : float
            BFS penilizing term (i.e. Return Parameter)
        q : float
            DFS penilizing term (i.e. In-out Parameter)
        
        Returns
        -------
        walks : List[List[int]]
            List of the obtraind random walks
        """
        walks = []
        for init_node in init_nodes:
            walk = [init_node.item()]
            while len(walk) < walk_length:
                cur = walk[-1]
                neighbors = list(graph.successors(cur).numpy()) # list of the possible nexe nodes
                if len(neighbors) == 0:
                    break
                if len(walk) == 1: # 1st step 
                    next_node = np.random.choice(neighbors) # randomly choose from neighbors
                else: # subsequest step
                    prev = walk[-2]
                    prob = []
                    for neighbor in neighbors:
                        if neighbor == prev:
                            prob.append(1 / p)
                        elif graph.has_edges_between(prev, neighbor):
                            prob.append(1)
                        else:
                            prob.append(1 / q)
                    prob = np.array(prob) / sum(prob) # normalizing the distribution
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
            BFS penilizing term (i.e. Return Parameter)
        q : float
            DFS penilizing term (i.e. In-out Parameter)
        epochs : int
            Number of training epochs.
        learning_rate : float
            Learning rate for optimization.
        """
        # Generate random walks
        all_walks = []
        for _ in range(num_walks):
            init_nodes = torch.arange(graph.num_nodes()) # a vector with 
            walks = self._biased_random_walk(graph=graph, init_nodes=init_nodes, walk_length=walk_length, p=p, q=q)
            all_walks.extend(walks) # 'extend' appends the given list by items

        # Convert walks to skip-gram pairs
        skip_gram_pairs = []
        for walk in all_walks:
            for i, target in enumerate(walk):
                for j in range(max(0, i - window_size), min(len(walk), i + window_size + 1)):
                    if i != j:
                        skip_gram_pairs.append((target, walk[j]))
        skip_gram_pairs = torch.tensor(skip_gram_pairs)

        # Training loop
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
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
