import json
from pathlib import Path
import web3
from web3 import Web3
from backend import config

from logging import getLogger

logger = getLogger(__name__)

class Account(object):
    def __init__(self, address) -> None:
        self.address = address
        self.private_key = None
    
    def _get_private_key(self, address):
        pass
    

