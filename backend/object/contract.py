from logging import getLogger

logger = getLogger(__name__)

import json 
import subprocess
from backend import config


class Contract(object):
    def __init__(self, contract_name: str) -> None:
        self.contract_name = contract_name
        self.path_to_contract = config.SOLC_DIR / contract_name
        self.path_to_sh = config.SOLC_DIR / 'build.sh'

    def contract_builder(self):
        logger.info("Building the contract...")
        self.path_to_build / 'build.json'        
        logger.info()

        cmd = ["bash", str(self.path_to_sh), self.contract_name]
        subprocess.run(["bash", cmd], check=True)

    def contract_generator(self):
        logger.info(f"Generating the contract...")
        with open(self.path_to_build, 'r') as f:
            build_info = json.load(f)
            contract_address = None # address not neccesary for contract creation
            self.abi = build_info['contracts'][self.contract_name]['abi']
            self.bytecode = build_info['contracts'][self.contract_name]["bin"]

        # define contract creation transaction
        self.contract = self.provider.eth.contract(address=contract_address, abi=self.abi, bytecode=self.bytecode)

