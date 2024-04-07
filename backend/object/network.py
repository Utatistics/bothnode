import json
from pathlib import Path
from web3 import Web3
from backend import config

from logging import getLogger

logger = getLogger(__name__)

class Network(object):
    def __init__(self, net_config: dict) -> None:
        """

        Args
        ----
        net_config : json
        
        """
        self.name = net_config['name']
        self.url = net_config['url']
        self.chain_id = net_config['chain_id']

    def connector(self):
        self.provider = Web3(Web3.HTTPProvider(self.url))
        if self.provider.is_connected():
            logger.info(f"Successfully connected to {self.name}!")
        else:
            raise ConnectionError(f'{self.name} NW not connected')
    
    def _get_network_attributes(self):
        if self.name == 'GANACHE':
            with open(config.PRIVATE_DIR / 'pk.json') as f:
                json = json.load(f)
    
    def get_nonce(self, address):
        return self.provider.eth.get_transaction_count(address)
