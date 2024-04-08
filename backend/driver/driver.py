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

    logger.info(f"The instance connecting to {net_name}...")
    net.connector()
    return net

def nounce_getter(net: Network, address: str):
    nounce = net.get_nonce(address=address)
    return nounce

def queue_getter(net: Network):
    net.get_queue()

def send_transaction(net: Network, sender_address: str, recipient_address: str, contract_name: str):
    logger.info("Sending TX...")
    sender = Account(sender_address, chain_id=net.chain_id, private_key=None)
    recipient = Account(recipient_address, chain_id=net.chain_id, private_key=None)
    if contract_name:
        contract = Contract(contract_name=contract_name)
    else:
        contract = None
    net.send_tx(sender=sender, recipient=recipient, contract=contract)

def detect_anamolies(method: str):
    logger.info("Let there be light")