import json
from logging import getLogger

logger = getLogger(__name__)

with open('config.json') as f:
    config_json = json.load(f)

from backend.object.network import Network
from backend.object.account import Account
from backend.object.contract import Contract
from backend.object.agent import FrontRunner

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
    
def send_transaction(net: Network,
                     sender_address: str,
                     recipient_address: str,
                     amount: int,
                     contract_name: str, 
                     build: bool, 
                     contract_params: dict,
                     func_name: str,
                     func_params: dict):
    
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
    
    net.send_tx(sender=sender, recipient=recipient, amount=amount, contract=contract, build=build, func_name=func_name, func_params=func_params)

def front_runner(net: Network):
    agent = FrontRunner(net=net) 

def detect_anamolies(method: str):
    logger.info("Let there be light")