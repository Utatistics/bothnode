from logging import getLogger

logger = getLogger(__name__)

import json 
import subprocess
from backend.util import config

class Contract(object):
    def __init__(self, contract_name: str, provider, contract_params: dict) -> None:
        self.contract_name = contract_name
        self.provider = provider
        self.contract_params = contract_params
        self.path_to_contract = config.SOLC_DIR / contract_name
        self.path_to_sh = config.SOLC_DIR / 'build.sh'

    def contract_builder(self):
        logger.info(f"Building the contract: {self.contract_name}")
        self.path_to_build = config.SOLC_DIR / self.contract_name / 'build_info.json'
        logger.debug(f'{self.path_to_build=}')
        cmd = f"{self.path_to_sh} {self.contract_name}"
  
        try:
            result = subprocess.run(cmd, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if result.stderr:
                logger.error(f"Build errors: {result.stderr}")
            logger.info(f'Generated build information: {self.path_to_build}')
        except subprocess.CalledProcessError as e:
            logger.error(f"Contract build failed with error: {e}")
        
    def contract_generator(self):
        logger.info(f"Generating the contract...")
  
        with open(self.path_to_build, 'r') as f:
            build_info = json.load(f)
            k = next(iter(build_info['contracts']))
            self.contract_address = None # address not neccesary for contract creation
            self.abi = build_info['contracts'][k]['abi']
            self.bytecode = build_info['contracts'][k]["bin"]
            
        # define contract creation transaction
        self.contract = self.provider.eth.contract(abi=self.abi, bytecode=self.bytecode)
        logger.info(f"Build completed: {self.contract_name}")
    
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
        self.path_to_contract_json = self.path_to_contract / 'contract_info.json'
        self.contract_address = contract_address
        data = self._to_dict()
        with open(self.path_to_contract_json, 'w') as json_file:
            json.dump(data, json_file, indent=4)
        logger.info(f'contract info saved at: {self.path_to_contract_json}')
