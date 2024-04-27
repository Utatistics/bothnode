from logging import getLogger

logger = getLogger(__name__)

import json 
import subprocess
from backend.util import config

class Contract(object):
    def __init__(self, contract_name: str, provider) -> None:
        self.contract_name = contract_name
        self.provider = provider
        self.path_to_contract = config.SOLC_DIR / contract_name
        self.path_to_sh = config.SOLC_DIR / 'build.sh'

    def contract_builder(self):
        logger.info(f"Building the contract: {self.contract_name}")
        self.path_to_build = config.SOLC_DIR / self.contract_name / 'build.json'
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
