import torch
import dgl
import torch.nn as nn
import torch.optim as optim
import numpy as np

class Node2Vec(nn.Module):
    def __init__(self, num_nodes, embedding_dim):
        super(Node2Vec, self).__init__()
        self.node_embeddings = nn.Embedding(num_nodes, embedding_dim)

    def forward(self, nodes):
        return self.node_embeddings(nodes)

    def get_similarity(self, node_u, node_v):
        # Compute the similarity score (dot product of embeddings)
        emb_u = self.node_embeddings(node_u)
        emb_v = self.node_embeddings(node_v)
        return torch.sum(emb_u * emb_v, dim=-1)

    def biased_random_walk(self, graph, nodes, walk_length, p, q):
        walks = []
        for start_node in nodes:
            walk = [start_node]
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
                        elif graph.has_edge_between(prev, neighbor):
                            prob.append(1)
                        else:
                            prob.append(1 / q)
                    prob = np.array(prob) / sum(prob)
                    next_node = np.random.choice(neighbors, p=prob)
                walk.append(next_node)
            walks.append(walk)
        return walks

    def train_node2vec(self, graph, walk_length, num_walks, window_size, p, q, epochs, learning_rate):
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        # Generate random walks
        all_walks = []
        for _ in range(num_walks):
            start_nodes = torch.arange(graph.num_nodes())
            walks = self.biased_random_walk(graph, start_nodes, walk_length, p, q)
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

    def compute_similarity_matrix(self):
        # Compute the pairwise similarity scores for all nodes
        embeddings = self.node_embeddings.weight
        similarity_matrix = embeddings @ embeddings.T
        return similarity_matrix
