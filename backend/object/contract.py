from logging import getLogger
from web3 import Web3

logger = getLogger(__name__)

import json 
import subprocess
from pathlib import Path
from backend.util.config import Config

config = Config()


class Contract(object):
    def __init__(self, contract_name: str, provider, contract_params: dict) -> None:
        self.contract_name = contract_name
        self.provider = provider
        self.contract_params = contract_params
        self.path_to_contract = config.SOLC_DIR / contract_name
        self.path_to_contract_json = self.path_to_contract / 'contract_info.json'
            
    def _to_dict(self) -> dict:
        return {
            "contract_name": self.contract_name,
            "contract_params": self.contract_params,
            "path_to_contract": str(self.path_to_contract),
            "path_to_sh": str(self.path_to_sh),
            "contract_address": getattr(self, 'contract_address', None),
            "abi": getattr(self, 'abi', None),
            "bytecode": getattr(self, 'bytecode', None)
        }

    def write_to_json(self, contract_address: str):
        self.contract_address = contract_address
        data = self._to_dict()
        with open(self.path_to_contract_json, 'w') as json_file:
            json.dump(data, json_file, indent=4)
        logger.info(f'contract info saved at: {self.path_to_contract_json}')