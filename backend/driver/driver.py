import json
import datetime

from backend.object.network import Network
from backend.object.account import Account
from backend.object.wallet import Wallet
from backend.object.contract import Contract
from backend.object.agent import FrontRunner, target_criteria
from backend.object.block import Block
from backend.object.graph import NodeFeature, EdgeFeature, Graph
from backend.object.model import GraphConvNetwork
from backend.object.crowler import CryptoScamDBCrowler
from backend.object.random_walk import Node2Vec

from backend.util.config import Config
from logging import getLogger

from backend.object.db import MongoDBClient, add_auth_to_mongo_connection_string

logger = getLogger(__name__)

config = Config()
db_config = config.DB_CONFIG
extl_config = config.EXTL_CONFIG

connection_string = add_auth_to_mongo_connection_string(connection_string=db_config['connection_string'],
                                                        username=db_config['init_username'],
                                                        password=db_config['init_password'])

def init_net_instance(net_name: str, protocol: str) -> None:
    logger.info(f"Creating Network instance of {net_name}...")
    net_config = config.NET_CONFIG[net_name.upper()] 
    net = Network(net_config=net_config)

    return net

def query_handler(net: Network, target: str, query_params: dict) -> None:
    logger.debug(f'Querying {target}...')
    logger.debug(f'{query_params=}')

    if target == 'nonce':
        try:
            address = query_params['address']
        except:
            logger.error('address parameter cannot be found.')
            raise ValueError('address parameter cannot be found.')
        return net.get_nonce(address=address)
        
    elif target =='block_info':
        try:
            net.get_block_info(number=query_params['block_number'], hash=None)
        except KeyError:
            net.get_block_info(number=None, hash=query_params['block_hash'])
        except TypeError:
            net.get_block_info(number=None, hash=None)

    elif target == 'chain_info':
        net.get_chain_info()

    elif target == 'gas_price':
        net.get_gas_price()

    elif target == 'queue':
        return net.get_queue()
    
    else:
        raise ValueError(f'Invalid target: {target}')
    
def send_transaction(net: Network, sender_address: str, recipient_address: str, amount: int, contract_address: str, func_name: str, func_params: dict) -> None:
    """Send a transaction, which may involve interacting with a smart contract.

    Args
    ----
    net : Network
        The network object used to interact with the blockchain.
    sender_address : str
        The address of the sender's account.
    recipient_address : str
        The address of the recipient's account.
    amount : int
        The amount of cryptocurrency to send (in wei).
    contract_address : str
        The address of the smart contract to interact with
    func_name : str
        The name of the function to call on the smart contract
    func_params : dict
        The parameters for the contract function call, if applicable.

    Returns
    -------
    None
    """
    logger.info("Sending TX...")
    sender = Account(sender_address, private_key=None, chain_id=net.chain_id)
  
    if contract_address:
        recipient = None
        contract = Contract(contract_address=contract_address, provider=net.provider)
        try:
            db_client = MongoDBClient(uri=connection_string, database_name='contract_db')
            document = db_client.find_document(collection_name='deployment', query={"address": contract_address})
            contract.load_from_document(document=document)
        except Exception as e:
            logger.error(f"Failed to load data from MongoDB with address={contract_address}: {e}")

    else:
        contract = None
        recipient = Account(recipient_address, private_key=None, chain_id=net.chain_id)
    
    payload = net.create_payload(sender=sender, recipient=recipient, amount=amount, contract=contract, func_name=func_name, func_params=func_params)
    net.send_tx(sender=sender, payload=payload, contract=contract)     

def check_wallet_balance(net: Network, wallet_address: str) -> None:
    wallet = Wallet(provider=net.provider, wallet_address=wallet_address)
    wallet.get_balance()
    
def front_runner(net: Network, sender_address: str) -> None:
    logger.info("Sending TX...")

    sender = Account(sender_address, private_key=None, chain_id=net.chain_id)
    agent = FrontRunner(net=net, sender=sender)
    mempool = agent.query_mempool 
    target_tx = agent.lockon_tx(mempool=mempool, target_criteria=target_criteria)
    payload = agent.create_payload(target_tx=target_tx)
    agent.execute_frontrun(payload=payload)

    # Prepare the data to be stored
    logger.info("DB ingestion")
    transaction_data = {
        "sender_address": sender_address,
        "target_transaction": target_tx,
        "payload": payload,
        "timestamp": datetime.datetime.now()
    }
    # Insert data into MongoDB
    try:
        db_client = MongoDBClient(uri=connection_string, database_name='transaction_db')
        db_client.insert_document(collection_name='transactions', document=transaction_data)
    except Exception as e:
        logger.error(f"Failed to store data in MongoDB: {e}")

def run_label_crowler() -> None:
    """Fetching blacklisted accounts and restore them to DB as training signal 
    """
    logger.info("Fetching blacklists from external service provider.")
    crowler = CryptoScamDBCrowler(extl_config=extl_config)
    black_list = crowler.get_black_list()
    logger.info(black_list)
    logger.info("DB ingestion")
    '''
    try:
        db_client = MongoDBClient(uri=connection_string, database_name='')
    except Exception as e:
        logger.error(f"Failed to store data in MongoDB: {e}")
    '''

def detect_anamolies(net: Network, method: str, block_num: int, block_len: int) -> None:
    """detect anamolies in the network with the specified method
    
    Args
    ----
    net : Network
        target network to run the detection algorithm
    method : str
        to be implemented
    block_num : int
        the latest number of blocks to aggregate the transactions from.
    block_len : int
        the length of blocks to aggregate the transactions from.
    """
    logger.info("Retriving data from node.")
    if not block_num:
        block_num = net.get_latest_block_num() 
    block = Block(net_name=net.name)
    block.query_blocks(block_num=block_num, block_len=block_len)
    block.write_to_json(path_to_json=config.PRIVATE_DIR / 'blockdata.json')


    logger.info("Graph construction")
    node_feature = NodeFeature(block_data=block.block_data)
    edge_feature = EdgeFeature(block_data=block.block_data)
    # node_feature.write_to_json(path_to_json=config.PRIVATE_DIR / 'node.json')
    # edge_feature.write_to_json(path_to_json=config.PRIVATE_DIR / 'edge.json')
    graph = Graph(node_feature=node_feature, edge_feature=edge_feature)
    # graph.draw_graph()

    
    logger.info("DB ingestion")
    '''
    try:
        db_client = MongoDBClient(uri=connection_string, database_name='')
    except Exception as e:
        logger.error(f"Failed to store data in MongoDB: {e}")
    '''
    
    logger.info("Scoring node similarity via Randam Walk") 
    num_nodes = graph.num_nodes()
    embedding_dim = 16
    walk_length = 10
    num_walks = 80
    window_size = 2
    p = 1  # Return parameter
    q = 1  # In-out parameter
    epochs = 10
    learning_rate = 0.01

    rw = Node2Vec(num_nodes=num_nodes, embedding_dim=embedding_dim)
    embeddings = rw.train_node2vec(graph, walk_length, num_walks, window_size, p, q, epochs, learning_rate)
    similarity_matrix = model.compute_similarity_matrix()
    logger.info("Similarity Matrix:\n", similarity_matrix)

    
    logger.info('Learning the embedding via GNN')    
    gcn = GraphConvNetwork(graph=graph)
    gcn.define_forward()    
    result = gcn(graph.graph, graph.graph.ndata['tensor'])
    
