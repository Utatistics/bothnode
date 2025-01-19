import json
from pathlib import Path
from typing import List, Dict
import matplotlib.pyplot as plt

import dgl
import torch
import numpy as np
import networkx as nx

from backend.object.network import Network
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
        for block in block_data.values():
            timestamp = int(block.get("timestamp", "0x0"), 16)

            transactions = block.get("transactions", [])
            for tx in transactions:
                sender = tx.get("from")
                recipient = tx.get("to")
                value = int(tx.get("value", "0x0"), 16) / 1e18  # Convert Wei to Ether
                gas_used = int(tx.get("gas", "0x0"), 16)

                if sender:
                    if sender not in self.nodes:
                        self.nodes[sender] = {"address": sender, "total_sent": 0, "total_received": 0, "total_gas_used": 0, "last_active": 0}
                    self.nodes[sender]["total_sent"] += value
                    self.nodes[sender]["total_gas_used"] += gas_used
                    self.nodes[sender]["last_active"] = max(self.nodes[sender]["last_active"], timestamp)

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
        for block in block_data.values():
            timestamp = int(block.get("timestamp", "0x0"), 16)

            transactions = block.get("transactions", [])
            for tx in transactions:
                self.edges.append({
                    "edge_id": tx.get("hash"),  # Transaction hash
                    "from": tx.get("from"),  # Sender address
                    "to": tx.get("to"),  # recipient address
                    "value": int(tx.get("value", "0x0"), 16) / 1e18,  # Value in Ether (from Wei)
                    "gas_used": int(tx.get("gas", "0x0"), 16),  # Gas used
                    "timestamp": timestamp,  # Block timestamp
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
    def __init__(self, node_feature: NodeFeature, edge_feature: EdgeFeature) -> None:
        """custom graph object
        
        Args
        ----
        node_feature : NodeFeature
        edge_feature : EdgeFeature
        
        """
        self.node_feature = node_feature
        self.edge_feature = edge_feature            
        self.index_to_address = {i: features['address'] for i, features in enumerate(self.node_feature.nodes.values())}
        self.address_to_index = {features['address']: i for i, features in enumerate(self.node_feature.nodes.values())}
                                        
        try:
            self._node_link_generator()
            logger.info("Successfully constructed graph object.")
        except Exception as e:
            logger.error(f"node link generation failed: {e}") 

        try:
            self._tensor_generator()         
            logger.info("Successfully added features to the graph.")
        except Exception as e:
            logger.error(f"tensor generation failed: {e}") 
          
        logger.info(f'number of graph nodes: {self.graph.num_nodes()}')
        logger.info(f'number of graph edges: {self.graph.num_edges()}')
   
    def _node_link_generator(self) -> None:
        """create DGL graph object
        """
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

    def _tensor_generator(self) -> None:
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
            if not edge['to']:
                continue
            else:
                edge_features.append([
                edge['value'],
                edge['gas_used'],
                edge['timestamp']
            ])
        
        self.graph.ndata['tensor'] = torch.tensor(node_features, dtype=torch.float32)
        self.graph.edata['tensor'] = torch.tensor(edge_features, dtype=torch.float32)

    def _update_node_feature(self, address_set: set) -> None:
        """update node_feature attribute given the current graph
        
        Args
        ----
        address_set : set
        
        """
        self.node_feature.nodes = {address: self.node_feature.nodes[address] for address in address_set}
                
    def _update_edge_feature(self, address_set: set) -> None:
        """update edge_feature attribute given the current graph

        Args
        ----
        address_set : set
        
        """
        edge_id_set = set() 
        for edge in self.edge_feature.edges:
            edge_id = edge['edge_id']
            if edge['from'] in address_set or edge['to'] in address_set:
                edge_id_set.add(edge_id) 
    
        logger.debug(f'{len(edge_id_set)=}')
        self.edge_feature.edges = [edge for edge in self.edge_feature.edges if edge['edge_id'] in edge_id_set]
            
    def graph_sampler(self, base_num: int, base_ratio: float) -> None:
        """sampling 
        
        Args
        ----
        base_num : int
            number of nodes from normal graph to sample from, in propotion to the given base_ratio
        base_ratio : float
            sampling ratio

        """
        logger.info(f"Appying Graph Sampling")
    
        p = base_ratio * base_num / ((1 - base_ratio) * self.graph.num_nodes())
        logger.debug(f'{base_num=}')
        logger.debug(f'{base_ratio=}')
        logger.debug(f'{self.graph.num_nodes()=}')
        logger.debug(f'{p=}')
        
        # node selection logic
        in_degrees = self.graph.in_degrees()
        out_degrees = self.graph.out_degrees()
        degrees = in_degrees + out_degrees
        node_indices = torch.argsort(degrees, descending=True) # sorted in the order of highest degrees
        sampled_nodes = node_indices[:int(p * len(node_indices))]  # top i-th node to be selected 
        subgraph = self.graph.subgraph(sampled_nodes.tolist()) 

        if '_ID' in subgraph.ndata:
            del subgraph.ndata['_ID']
        if '_ID' in subgraph.edata:
            del subgraph.edata['_ID']
        
        self.graph = subgraph
        
        address_set = set([self.index_to_address[int(i)] for i in sampled_nodes])
        logger.warning(f'{len(address_set)=}')

        self._update_node_feature(address_set=address_set)
        self._update_edge_feature(address_set=address_set)
        
        logger.info(f'number of graph nodes: {self.graph.num_nodes()}')
        logger.info(f'number of graph edges: {self.graph.num_edges()}')
        logger.warning(f'{len(self.node_feature.nodes)=}')
        logger.warning(f'{len(self.edge_feature.edges)=}')
      
    def get_node_addresses(self, node_index: list) -> list:
        """obtaine list of node addresses based on the given indices
        
        Args
        ----
        node_index : list
            list of node indices 
        
        Returns
        -------
        node_address : list
            list of node addresses
        """
        return [self.index_to_address[i] for i in node_index]

    def draw_graph(self, path_to_png: Path, anomaly_dict: dict) -> None:
        """visuallize graph structure
        
        Args
        ----
        path_to_png : Path
            the path to the output png.
        """
                
        nx_g = self.graph.to_networkx()
        
        #matplotlib.use("TkAgg")

        plt.figure(figsize=(10, 10))

        pos = nx.spring_layout(nx_g, seed=42)
        
        if anomaly_dict:
            node_colors = ["red" if node in anomaly_dict else "blue" for node in nx_g.nodes()]

        else:
            node_colors = 'blue'
                
        nx.draw(
            nx_g, pos, node_size=50, node_color=node_colors, edge_color="gray", alpha=0.7, with_labels=False
        )
        
        plt.savefig(path_to_png, format="PNG")
        plt.close()  # Close the plot to avoid it showing up


def graph_merger(graph_normal: Graph, *args: Graph) -> Graph:
    """Merges multiple DGLGraph objects into a single graph.
    
    Args
    ----
    graph_normal : graph
        graph with normal nodes (i.e. obtained via RPC)
    *args : Graph
        abnorml graphs *contained in tuple
        graphs with abnormal nodes (i.e. obtrained from external source via REST)
        Variable number of Graph objects to be merged.
    
    Returns
    -------
    graph : Graph
        A single merged Graph object
    """
    logger.info("Nerging the graphs...")
    try:
        dgl_graphs = [graph.graph for graph in args] # abnormal graphs
        attr = [graph.index_to_address for graph in args]
        
        '''
        dgl_graph = dgl.batch(dgl_graphs)
        graph = Graph(graph=dgl_graph) # call '_load_from_dglGraph'
        '''
        
        node_feature = graph_normal.node_feature
        edge_feature = graph_normal.edge_feature
             
        nodes = node_feature.nodes
        edges = edge_feature.edges
        node_list = [graph.node_feature.nodes for graph in args]
        edge_list = [graph.edge_feature.edges for graph in args]
        
        for node, edge in zip(node_list, edge_list):
            nodes = {**node, **nodes}
            edges += edge
        
        # overwrite node/edge data
        node_feature.nodes = nodes
        edge_feature.edges = edges         
        graph = Graph(node_feature=node_feature, edge_feature=edge_feature)

        nx_graph = graph.graph.to_networkx()
        components = list(nx.weakly_connected_components(nx_graph))
        num_subgraphs = len(components)       
        component_sizes = [len(component) for component in components]
        percentiles = np.percentile(component_sizes, [0, 25, 50, 75, 100]).astype(int)

        if num_subgraphs == 1:
            logger.info("Graph is connected.") 
        else:
            logger.warning(f"Graph contains {num_subgraphs} disconnected subgraphs.")
            logger.warning(
                f"Subgraph size distribution: "
                f"min={percentiles[0]}, Q1={percentiles[1]}, median={percentiles[2]}, "
                f"Q3={percentiles[3]}, max={percentiles[4]}"
            )
            
                
        logger.info("Graph Merge Successful.")
        return graph
    
    except Exception as e:
        logger.error(f"Graph Merge Failed: {e}")
        return None
    