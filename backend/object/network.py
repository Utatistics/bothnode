import json
from pathlib import Path
import web3
from web3 import Web3
from backend.util import config
from hexbytes import HexBytes 

from logging import getLogger
from backend.object.account import Account
from backend.object.contract import Contract
from backend.util.tools import convert_to_serializable

logger = getLogger(__name__)

class Network(object):
    def __init__(self, net_config: dict) -> None:
        """

        Args
        ----
        net_config : json
        
        """
        # set attributes
        self.name = net_config['name']
        self.url = net_config['url']
        self.chain_id = net_config['chain_id']

        # init instance 
        self._connector()
        if self.chain_id == 1337:
            self._modify_ganache_account_keys()

    def _connector(self):
        logger.info(f"The instance connecting to {self.name}...")
        self.provider = Web3(Web3.HTTPProvider(self.url))
        if self.provider.is_connected():
            logger.info(f"Successfully connected to {self.name}!")
        else:
            raise ConnectionError(f'{self.name} NW not connected')
        
    def _modify_ganache_account_keys(self):
        path_to_json = config.PRIVATE_DIR / 'ganache_pk.json'
        addresses = self.provider.eth.accounts
        address_map = {a.lower(): a for a in addresses}
        with open(path_to_json, mode='r') as jf:
            account_keys = json.load(jf)
            d = {
                "addresses": {address_map.get(k,k): address_map.get(v,v) for k, v in account_keys["addresses"].items()},
                "private_keys": {address_map.get(k,k): v for k, v in account_keys["private_keys"].items()},
                }
            with open(path_to_json, mode='w') as jj:
                json.dump(d, jj, indent=4)

    def get_nonce(self, address: str):
        nonce = self.provider.eth.get_transaction_count(address)
        logger.info(f'>> Nonce={nonce} for {address=}')
        return nonce
    
    def get_block_info(self, number: int, hash: str):
        if number:
            block_info = self.provider.eth.get_block(number)
        elif hash:
            block_info = self.provider.eth.get_block(hash)
        else:
            block_info = self.provider.eth.get_block('latest')

        block_info_serializable = {k: convert_to_serializable(v) for k, v in block_info.items()}
        logger.info(">> Block info:\n%s", json.dumps(block_info_serializable, indent=4))
        return block_info

    def get_chain_info(self):
        logger.info(f">> Chain ID: {self.provider.eth.chain_id}")
        logger.info(f">> Chain hashrate: {self.provider.eth.hashrate}")
        logger.info(f">> Chain syncing status: {self.provider.eth.syncing}")

    def get_gas_price(self):
        gas_price = self.provider.eth.gas_price
        max_priority_fee = self.provider.eth.max_priority_fee 
        gas_price_gwei = self.provider.from_wei(gas_price, 'gwei')
        max_priority_fee_gwei = self.provider.from_wei(max_priority_fee, 'gwei')
        
        logger.info(f"Current gas price: {gas_price}")
        logger.info(f"Current gas price (gwei): {gas_price_gwei}")
        logger.info(f"Max priority fee: {max_priority_fee}")
        logger.info(f"Max priority fee (gwei): {max_priority_fee_gwei}")

    def get_queue(self):
        logger.info(f'{web3.geth.txpool.content=}')
        logger.info(f'{web3.geth.txpool.inspect=}')
        logger.info(f'{web3.geth.txpool.status=}')
 
    def _create_payload(self, sender: Account, recipient: Account, amount: int, contract: Contract, build: bool):
        nonce = self.get_nonce(address=sender.address)

        if contract:
            logger.info('>> Smart Contract Transaction')
            if build:
                constructor_args = (sender, recipient)   
                payload = contract.contract.constructor(*constructor_args).build_transaction(
                    {
                        "nonce": nonce,  # Include the nonce here
                    }
                )
            else:
                payload = { # UPDATE REQUIRED!
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
                'gas': 100000,  # Gas limit
                'nonce': nonce
            }

        logger.info(">> Payload content:\n%s", json.dumps(payload, indent=4))                
        return payload

    def send_tx(self, sender: Account, recipient: Account, amount: int, contract: Contract, build: bool):
        # create payload
        payload = self._create_payload(sender=sender, recipient=recipient, amount=amount, contract=contract, build=build)

        # sign & send the transaction
        sender_pk = sender.private_key
        signed_tx = self.provider.eth.account.sign_transaction(payload, sender_pk)
        hashed_tx = self.provider.eth.send_raw_transaction(signed_tx.rawTransaction)

        # store the result of transaction
        tx_receipt = self.provider.eth.wait_for_transaction_receipt(hashed_tx)
        contract_address = tx_receipt.contractAddress
        
        if tx_receipt.status == 1:
            contract_address = tx_receipt.contractAddress  # Get contract address if applicable
            logger.info(f"Successfully completed  the transaction.")
        else:
            # Transaction failed or reverted
            logger.error("The transaction failed or reverted.")

        # update
        if contract and build:
            logger.info("Post-deployment update of the contract attribute.")
            contract.contract = self.provider.eth.contract(address=contract_address, abi=contract.abi, bytecode=contract.by) 