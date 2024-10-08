from logging import getLogger

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
        self.path_to_sh = config.SOLC_DIR / 'build.sh'
        self.path_to_contract_json = self.path_to_contract / 'contract_info.json'

    def contract_builder(self):
        logger.info(f"Building the contract: {self.contract_name}")
        self.path_to_build = config.SOLC_DIR / self.contract_name / 'build_info.json'
        logger.info(f'{self.path_to_build=}')
        cmd = f"{self.path_to_sh} {self.contract_name}"
  
        try:
            result = subprocess.run(cmd, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if result.stderr:
                logger.error(f"Build errors: {result.stderr}")
            logger.info(f'Generated build information: {self.path_to_build}')
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Contract build failed with error: {e}")
            print(e.stderr)
            
    def _to_dict(self) -> dict:
        return {
            "contract_name": self.contract_name,
            "contract_params": self.contract_params,
            "path_to_contract": str(self.path_to_contract),
            "path_to_sh": str(self.path_to_sh),
            "path_to_build": str(getattr(self, 'path_to_build', None)),
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
    
    def load_from_build(self):
        if not self.path_to_build.exists():
            logger.error(f'JSON file not found: {self.path_to_build}')
            return

        with open(self.path_to_build, 'r') as jf:
            build_info = json.load(jf)
            k = next(iter(build_info['contracts']))
            self.abi = build_info['contracts'][k]['abi']
            self.bytecode = build_info['contracts'][k]["bin"]
            self.contract_address = None # address not neccesary for contract creation
            
        self.contract = self.provider.eth.contract(abi=self.abi, bytecode=self.bytecode, address=self.contract_address)
        logger.info(f'Contract info loaded from: {self.path_to_build}')

    def load_from_contract(self):
        if not self.path_to_contract_json.exists():
            logger.error(f'JSON file not found: {self.path_to_contract_json}')
            return

        with open(self.path_to_contract_json, 'r') as jf:
            contract_info = json.load(jf)
            self.abi = contract_info['abi']
            self.bytecode = contract_info['bytecode']
            self.contract_address = contract_info['contract_address']
        
        self.contract = self.provider.eth.contract(abi=self.abi, bytecode=self.bytecode, address=self.contract_address)
        logger.info(f'Contract info loaded from: {self.path_to_contract_json}')

