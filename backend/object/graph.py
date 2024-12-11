import json
from typing import List, Dict
import matplotlib.pyplot as plt

import dgl
import torch
import networkx as nx

from backend.util.config import Config
from logging import getLogger

logger = getLogger(__name__)

config = Config()

class NodeFeature(object):
    def __init__(self, block_data: List[Dict]):
        self._feature_extractor(block_data=block_data)
        
    def _feature_extractor(self, block_data: List[Dict]) -> None:
        """
        Extract node features from the block data.

        Args
        -------
        block_data : List[Dict]
        """
        self.nodes = {}
        for block in block_data:
            timestamp = int(block.get("timestamp", "0x0"), 16)

            # Process transactions to extract node features
            transactions = block.get("transactions", [])
            for txn in transactions:
                sender = txn.get("from")
                recipient = txn.get("to")
                value = int(txn.get("value", "0x0"), 16) / 1e18  # Convert Wei to Ether
                gas_used = int(txn.get("gas", "0x0"), 16)

                # Update sender's features
                if sender:
                    if sender not in self.nodes:
                        self.nodes[sender] = {"address": sender, "total_sent": 0, "total_received": 0, "total_gas_used": 0, "last_active": 0}
                    self.nodes[sender]["total_sent"] += value
                    self.nodes[sender]["total_gas_used"] += gas_used
                    self.nodes[sender]["last_active"] = max(self.nodes[sender]["last_active"], timestamp)

                # Update recipient's features
                if recipient:
                    if recipient not in self.nodes:
                        self.nodes[recipient] = {"address": recipient, "total_sent": 0, "total_received": 0, "total_gas_used": 0, "last_active": 0}
                    self.nodes[recipient]["total_received"] += value
                    self.nodes[recipient]["last_active"] = max(self.nodes[recipient]["last_active"], timestamp)
                                        
    def write_to_json(self, path_to_json: str) -> None:
        """
        Save the extracted edges to a JSON file.

        Args
        ----
        edges : List[Dict]
            The list of edges to save.
        output_file : str
            Path to the output JSON file.
        """
        with open(path_to_json, "w") as jf:
            json.dump(self.nodes, jf, indent=2)
        
class EdgeFeature(object):
    def __init__(self, block_data: List[Dict]):
        self._feature_extractor(block_data=block_data)
        
    def _feature_extractor(self, block_data: List[Dict]) -> None:
        """
        Extract edge features from the block data.

        Args
        -------
        List[Dict]
            A list of edges with features such as source, target, value, and timestamp.
        """
        self.edges = []
        for block in block_data:
            block_hash = block.get("hash")
            timestamp = int(block.get("timestamp", "0x0"), 16)

            # Extract transactions
            transactions = block.get("transactions", [])
            for txn in transactions:
                self.edges.append({
                    "edge_id": txn.get("hash"),  # Transaction hash
                    "from": txn.get("from"),  # Sender address
                    "to": txn.get("to"),  # recipient address
                    "value": int(txn.get("value", "0x0"), 16) / 1e18,  # Value in Ether (from Wei)
                    "gas_used": int(txn.get("gas", "0x0"), 16),  # Gas used
                    "timestamp": timestamp,  # Block timestamp
                    "block_hash": block_hash  # Hash of the block
                })

    def write_to_json(self, path_to_json: str) -> None:
        """
        Save the extracted edges to a JSON file.

        Args
        ----
        edges : List[Dict]
            The list of edges to save.
        output_file : str
            Path to the output JSON file.
        """
        with open(path_to_json, "w") as jf:
            json.dump(self.edges, jf, indent=2)

class Graph(object):
    def __init__(self, node_feature: NodeFeature, edge_feature: EdgeFeature):
        """
        Args
        ----
        node_feature : NodeFeature
        edge_feature : EdgeFeature
        
        """
        self.node_feature = node_feature
        self.edge_feature = edge_feature
        
        try:
            self._node_link_generator()
            self._tensor_generator()
            
            logger.info("Successfully constructed graph structure.")
            self.get_graph_attr()
        except Exception as e:
            logger.error(f"Graph construction failed: {e}") 
             
    def _node_link_generator(self):
        """create DGL graph object
        """
        self.address_to_index = {features['address']: i for i, features in enumerate(self.node_feature.nodes.values())}
        
        src = []
        dst = []
        for edge in self.edge_feature.edges:
            from_address = edge['from']
            to_address = edge['to']            
            from_index = self.address_to_index.get(from_address)
            to_index = self.address_to_index.get(to_address)            
            if from_index != None and to_index != None:
                src.append(from_index)
                dst.append(to_index)
        self.graph = dgl.graph((src, dst))

    def _tensor_generator(self):
        """extract tensors from node and edge features
        """        
        node_features = []
        for node in self.node_feature.nodes.values():
            node_features.append([
                node['total_sent'],
                node['total_received'],
                node['total_gas_used'],
                node['last_active']
            ])
        edge_features = []
        for edge in self.edge_feature.edges:
            edge_features.append([
                edge['value'],
                edge['gas_used'],
                edge['timestamp']
            ])
        
        self.graph.ndata['tensor'] = torch.tensor(node_features, dtype=torch.float32)
        self.graph.edata['tensor'] = torch.tensor(edge_features, dtype=torch.float32)
    
    def get_graph_attr(self):
        """querying graph structure 
        """
        logger.info(f'{self.graph.ntypes=}')
        logger.info(f'{self.graph.etypes=}')
        logger.info(f'{self.graph.srctypes=}')
        logger.info(f'{self.graph.dsttypes=}')
        logger.info(f'{self.graph.canonical_etypes=}')
        logger.info(f'{self.graph.metagraph()=}')
        
        logger.info(f'{self.graph.num_nodes()=}')
        logger.info(f'{self.graph.num_edges()=}')

        logger.info(f'{self.graph.is_unibipartite=}')
        logger.info(f'{self.graph.is_multigraph=}')
        logger.info(f'{self.graph.is_homogeneous=}')
        
        #logger.info(f'{graph.graph.in_degrees()=}')
        #logger.info(f'{graph.graph.out_degrees()=}')
        #logger.info(f'{graph.graph.adj()=}')
        
    def draw_graph(self):
        
        nx_g = self.graph.to_networkx()
        
        #matplotlib.use("TkAgg")
        plt.figure(figsize=(10, 10))
        
        pos = nx.spring_layout(nx_g, seed=42)  # Layout for better visualization
        nx.draw(
            nx_g, pos, node_size=20, node_color="blue", edge_color="gray", alpha=0.7, with_labels=False
        )
        plt.show()
        plt.savefig(config.PRIVATE_DIR / "graph_visualization.png", format="PNG")  # You can change the filename and format
        plt.close()  # Close the plot to avoid it showing up

        
        