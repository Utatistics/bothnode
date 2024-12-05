import json
from typing import List, Dict

from logging import getLogger

logger = getLogger(__name__)

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
                receiver = txn.get("to")
                value = int(txn.get("value", "0x0"), 16) / 1e18  # Convert Wei to Ether
                gas_used = int(txn.get("gas", "0x0"), 16)

                # Update sender's features
                if sender:
                    if sender not in self.nodes:
                        self.nodes[sender] = {"total_sent": 0, "total_received": 0, "total_gas_used": 0, "last_active": 0}
                    self.nodes[sender]["total_sent"] += value
                    self.nodes[sender]["total_gas_used"] += gas_used
                    self.nodes[sender]["last_active"] = max(self.nodes[sender]["last_active"], timestamp)

                # Update receiver's features
                if receiver:
                    if receiver not in self.nodes:
                        self.nodes[receiver] = {"total_sent": 0, "total_received": 0, "total_gas_used": 0, "last_active": 0}
                    self.nodes[receiver]["total_received"] += value
                    self.nodes[receiver]["last_active"] = max(self.nodes[receiver]["last_active"], timestamp)
                                        
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
                    "to": txn.get("to"),  # Receiver address
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
        self.node_feature_dict = {node_id: idx for idx, node_id in enumerate(node_feature.nodes.keys())}        
        
        node_features = []
        for i in self.node_feature_dict:
            d = self.node_feature_dict[i]
            
            logger.warning(f'{i=}')
            logger.warning(f'{d=}')

            node_features.append([
                d["total_sent"],
                d["total_received"],
                d["total_gas_used"],
                d["last_active"]
            ])        
        
        edge_features = []
        for edge in edge_feature:
            edge_features.append([
                edge["value"],
                edge["gas_used"],
                edge["timestamp"]
            ])
        
        self._graph_constructor(node_features=node_features, edge_features=edge_features)
    
    def _graph_constructor(self, node_features: List[List], edge_features: List[List]):   
        src = [self.node_feature_dict[edge['from']] for edge in edge_feature.edges]
        dst = [self.node_feature_dict[edge['to']] for edge in edge_feature.edges]
        
        self.graph = dgl.graph((src, dst))    
        self.graph.ndata['feature'] = torch.tensor(node_features, dtype=torch.float32)
        self.graph.edata['feature'] = torch.tensor(edge_features, dtype=torch.float32)

    
