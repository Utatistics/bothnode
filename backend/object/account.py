import json
from pathlib import Path
import web3
from web3 import Web3

from backend.util.config import Config
from logging import getLogger

config = Config()
logger = getLogger(__name__)

class Account(object):
    def __init__(self, address: str, private_key: str, chain_id: str) -> None:
        self.chain_id = chain_id
        self.address = address
        if private_key:
            self.private_key = private_key
        else:
            self.private_key = self._get_private_key()
        
        logger.debug(f'Created an account: {self.address}')
        logger.debug(f'Private key: {self.private_key}')
    
    def _get_private_key(self):
        private_key = None
        if self.chain_id == 1337:
            with open(config.PRIVATE_DIR / 'ganache_pk.json') as jf:
                account_keys = json.load(jf)
                private_key = account_keys["private_keys"][self.address]

        elif self.net.chain_id == 1:
            pass
        else:
            pass
        self.private_key = private_key
        
        return private_key    

def create_account(chain_id: str) -> Account:
    instance = web3.eth.account.create()
    address = instance.address
    private_key = instance.privateKey.hex()
    account = Account(address=address, chain_id=chain_id, private_key=private_key)

    return account

