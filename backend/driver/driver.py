import json
from logging import getLogger

logger = getLogger(__name__)

with open('config.json') as f:
    config_json = json.load(f)

from backend.object.network import Network
from backend.object.account import Account
from backend.object.contract import Contract
from backend.object.agent import FrontRunner, target_criteria

def init_net_instance(net_name: str, protocol: str):
    logger.info(f"Creating Network instance of {net_name}...")
    net_config = config_json['NETWORK'][net_name.upper()] 
    net = Network(net_config=net_config)

    return net

def query_handler(net: Network, target: str, query_params: dict):
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
    
def send_transaction(net: Network, sender_address: str, recipient_address: str, amount: int, contract_name: str, build: bool, contract_params: dict, func_name: str, func_params: dict):
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
    contract_name : str
        The name of the smart contract to interact with, or an empty string if not using a contract.
    build : bool
        Whether to build and deploy the contract or just interact with an existing contract.
    contract_params : dict
        The parameters required for contract deployment or interaction.
    func_name : str
        The name of the function to call on the smart contract, or an empty string if not using a contract function.
    func_params : dict
        The parameters for the contract function call, if applicable.

    Returns
    -------
    None
    """
    logger.info("Sending TX...")
    sender = Account(sender_address, private_key=None, chain_id=net.chain_id)

    if contract_name:
        recipient = None
        contract = Contract(contract_name=contract_name, provider=net.provider, contract_params=contract_params)
        if build:
            contract.contract_builder()
            contract.load_from_build()
        else:
            contract.load_from_contract()

    else:
        contract = None
        recipient = Account(recipient_address, private_key=None, chain_id=net.chain_id)
    
    payload = net.create_payload(sender=sender, recipient=recipient, amount=amount, contract=contract, build=build, func_name=func_name, func_params=func_params)
    net.send_tx(sender=sender, payload=payload, contract=contract, build=build)

def front_runner(net: Network, sender_address: str):   
    logger.info("Sending TX...")
    
    sender = Account(sender_address, private_key=None, chain_id=net.chain_id)
    agent = FrontRunner(net=net, sender=sender)
    mempool = agent.query_mempool 
    target_tx = agent.lockon_tx(mempool=mempool, target_criteria=target_criteria)
    payload = agent.create_payload(target_tx=target_tx)
    agent.execute_frontrun(payload=payload)

def detect_anamolies(method: str):
    logger.info("Let there be light")