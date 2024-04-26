import json
from pathlib import Path
import web3
from web3 import Web3
from backend import config

from logging import getLogger
from backend.object.account import Account
from backend.object.contract import Contract

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
            with open(config.PRIVATE_DIR / 'ganache_pk.json') as f:
                json = json.load(f)
    
    def get_nonce(self, address):
        return self.provider.eth.get_transaction_count(address)
    
    def get_queue(self):
        web3.geth.txpool.content
        web3.geth.txpool.inspect
        web3.geth.txpool.status
 
    def send_tx(self, sender: Account, recipient: Account, amount: int, contract: Contract, build: bool):
        if contract:
            logger.info('>> Smart Contract Transaction')
            if build:
                constructor_args = (sender, recipient)   
                nonce = self.get_nonce(address=sender)
                payload = contract.contract.constructor(*constructor_args).build_transaction(
                    {
                        "nonce": nonce,  # Include the nonce here
                    }
                )
            else:
                # UPDATE REQUIRED!
                payload = {
                    'from' : sender.address,
                    'to': recipient.address,
                    'value': amount,  # Value in Wei (for Ethereum), usually 0 for token transfers
                    'data': contract.encode_function_call('transfer', [recipient.address, amount]),  # Encoded function call to transfer tokens
                    'gasPrice': self.provider.eth.gas_price,  # Gas price
                    'gas': 100000  # Gas limit
                }

        else:
            logger.info('>> Regular Transaction')
            payload = {
                'from': sender.address,
                'to': recipient.address,
                'value': amount,  # Amount of cryptocurrency to transfer
                'gasPrice': self.provider.eth.gas_price,  # Gas price
                'gas': 100000  # Gas limit
            }

        # sign & send the transaction
        sender_pk = sender.private_key
        logger.debug(f'{sender_pk=}')
        signed_tx = self.provider.eth.account.sign_transaction(payload, sender_pk)
        hashed_tx = self.provider.eth.send_raw_transaction(signed_tx.rawTransaction)

        # store the result of transaction
        tx_receipt = self.provider.eth.wait_for_transaction_receipt(hashed_tx)
        contract_address = tx_receipt.contractAddress
        
        # update
        logger.info("Post-deployment update of the contract attribute.")
        contract.contract = self.provider.eth.contract(address=contract_address, abi=contract.abi, bytecode=contract.by) # post-deployment update
     