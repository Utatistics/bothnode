import json
from pathlib import Path
import web3
from web3 import Web3
from backend import config

from logging import getLogger
from backend.config import PRIVATE_DIR

logger = getLogger(__name__)

class Account(object):
    def __init__(self, address: str, chain_id: str, private_key: str) -> None:
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
            with open(PRIVATE_DIR / 'ganache_pk.json') as jf:
                primary_keys = json.load(jf)
                # logger.debug(primary_keys["private_keys"].keys())
                private_key = primary_keys["private_keys"][self.address]
        elif self.chain_id == 1:
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

