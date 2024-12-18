import json
import datetime

from backend.object.network import Network
from backend.object.account import Account
from backend.object.contract import Contract
from backend.object.agent import FrontRunner, target_criteria
from backend.object.db import MongoDBClient, add_auth_to_mongo_connection_string

from backend.util.config import Config
from logging import getLogger

logger = getLogger(__name__)

config = Config()
db_config = config.DB_CONFIG
connection_string = add_auth_to_mongo_connection_string(connection_string=db_config['connection_string'], username=db_config['init_username'], password=db_config['init_password'])

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
            logger.error(f"Failed to load data from MongoDB: {e}")

    else:
        contract = None
        recipient = Account(recipient_address, private_key=None, chain_id=net.chain_id)
    
    payload = net.create_payload(sender=sender, recipient=recipient, amount=amount, contract=contract, func_name=func_name, func_params=func_params)
    net.send_tx(sender=sender, payload=payload, contract=contract)

def front_runner(net: Network, sender_address: str) -> None:
    logger.info("Sending TX...")

    sender = Account(sender_address, private_key=None, chain_id=net.chain_id)
    agent = FrontRunner(net=net, sender=sender)
    mempool = agent.query_mempool 
    target_tx = agent.lockon_tx(mempool=mempool, target_criteria=target_criteria)
    payload = agent.create_payload(target_tx=target_tx)
    agent.execute_frontrun(payload=payload)

    # Prepare the data to be stored
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

def detect_anamolies(method: str):
    logger.info("Let there be light")
    