from logging import getLogger
from web3 import Web3

logger = getLogger(__name__)

import json 
import subprocess
from pathlib import Path
from backend.util.config import Config

config = Config()


class Contract(object):
    def __init__(self, contract_address: str, provider) -> None:
        self.contract_address = contract_address
        self.provider = provider
            
    def _to_dict(self) -> dict:
        return {
            "contract_address": self.contract_name,
            "contract_name": getattr(self, 'contract_name', None),
            "abi": getattr(self, 'abi', None),
            "bytecode": getattr(self, 'bytecode', None)
        }

    def write_to_json(self, contract_address: str):
        self.contract_address = contract_address
        data = self._to_dict()
        with open(self.path_to_contract_json, 'w') as json_file:
            json.dump(data, json_file, indent=4)
        logger.info(f'contract info saved at: {self.path_to_contract_json}')
        
    def load_from_document(self, document: dict) -> None:
        self.abi = document['abi']
        self.bytecode = document['bytecode']
        
        