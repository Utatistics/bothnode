import json
from logging import getLogger

logger = getLogger(__name__)

with open('config.json') as f:
    config_json = json.load(f)

import web3
from backend.object.network import Network

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

def send_transaction(net: Network, tx_type: bool):
    logger.info("Sending TX...")
    net.send_tx(tx_type)

def detect_anamolies(method: str):
    logger.info("Let there be light")