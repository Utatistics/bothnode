import json
from logging import getLogger

logger = getLogger(__name__)

with open('config.json') as f:
    config_json = json.load(f)

import web3
from backend.object.network import Network
from backend.object.account import Account
from backend.object.contract import Contract

def init_net_instance(net_name: str, protocol: str):
    logger.info(f"Creating Network instance of {net_name}...")
    net_config = config_json['NETWORK'][net_name.upper()] 
    net = Network(net_config=net_config)

    return net

def nonce_getter(net: Network, address: str):
    nonce = net.get_nonce(address=address)
    return nonce

def queue_getter(net: Network):
    net.get_queue()

def send_transaction(net: Network, sender_address: str, recipient_address: str, amount: int, contract_name: str, build: bool):
    logger.info("Sending TX...")

    sender = Account(sender_address, private_key=None, chain_id=net.chain_id)
    recipient = Account(recipient_address, private_key=None, chain_id=net.chain_id)

    if contract_name:
        contract = Contract(contract_name=contract_name, provider=net.provider)
        if build:
            contract.contract_builder()
            logger.info(f'Smart Contract build completed: {contract_name}')
        contract.contract_generator()
    else:
        contract = None
    
    net.send_tx(sender=sender, recipient=recipient, amount=amount, contract=contract, build=build)

def detect_anamolies(method: str):
    logger.info("Let there be light")