from pathlib import Path
import json
import subprocess
from web3 import Web3
from eth_abi import encode

from backend.util import config
from backend.util.config import Config
from backend.object.account import Account
from backend.object.contract import Contract
from backend.util.tools import logger_hexbytes

from logging import getLogger

config = Config()
logger = getLogger(__name__)


class Network(object):
    def __init__(self, net_config: dict) -> None:
        """Network Interface object that allows user to interact with the network. 

        Args
        ----
        net_config : json
        
        """
        # set attributes
        self.name = net_config['name']
        self.local_rpc = net_config['local_rpc']
        self.chain_id = net_config['chain_id']

        # init instance 
        self._connector()
        if self.chain_id == 1337:
            self._modify_ganache_account_keys()

    def _connector(self):
        logger.info(f"The instance connecting to {self.name}...")
        self.provider = Web3(Web3.HTTPProvider(self.local_rpc))
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
    
    def get_block_number(self):
        number = self.provider.eth.get_block('latest').number
        logger.info(f'>> Block Number={number}')
        return number

    def get_chain_info(self):
        logger.info(f">> Chain ID: {self.provider.eth.chain_id}")
        logger.info(f">> Chain hashrate: {self.provider.eth.hashrate}")
        logger.info(f">> Chain syncing status: {self.provider.eth.syncing}")

    def get_gas_price(self) -> int:
        gas_price = self.provider.eth.gas_price
        return gas_price
    
    def get_max_fee_per_gas(self):
        block = self.provider.eth.get_block('latest')
        base_fee = block['baseFeePerGas']  # Get current block's base fee
        max_fee_per_gas = base_fee * 2  # Set maxFeePerGas to 2x the base fee to be safe
        return max_fee_per_gas
    
    def get_max_priority_fee(self) -> int:
        max_priority_fee = self.provider.eth.max_priority_fee
        return max_priority_fee
            
    def get_queue(self) -> list:
        try:
            mempool_sh = config.SCRIPT_DIR / "geth_mempool.sh"
            # mempool_sh = config.SCRIPT_DIR / "curl_mempool.sh"
            mempool_process = subprocess.run(["bash", str(mempool_sh)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)            
            if mempool_process.returncode == 0:
                mempool_data = json.loads(mempool_process.stdout)                
                pending_txs = mempool_data.get('result', {}).get('pending', {})                
                flattened_txs = [tx for txs in pending_txs.values() for tx in txs.values()]
                logger.info(f"Retrieved {len(flattened_txs)} pending transactions from the mempool.")
                return flattened_txs
            else:
                logger.error(f"Error executing script: {mempool_process.stderr}")
                return []
        except Exception as e:
            logger.error(f"An error occurred while querying the mempool: {str(e)}")
            return []

    def _encode_nested_dict_to_bytes(self, data: dict):
        """Recursively encodes any nested dictionaries with 'types' and 'values'into bytes using solidityPack
        """
        if isinstance(data, dict):
            # Check if the dict contains 'types' and 'values' keys for encoding
            if 'types' in data and 'values' in data:
                # Convert 'types' and 'values' to bytes using solidityPack
                return encode(data['types'], data['values'])
            else:
                # Recursively apply to all values in the dictionary
                return {key: self._encode_nested_dict_to_bytes(value) for key, value in data.items()}
        elif isinstance(data, list):
            # Apply encoding to each item if it's a list
            return [self._encode_nested_dict_to_bytes(item) for item in data]
        else:
            # Return the data as-is if it's neither dict nor list
            return data
        
    def create_payload(self, sender: Account, recipient: Account, amount: int, contract: Contract, func_name: str, func_params: dict) -> dict:
        """Create a payload for a transaction or smart contract interaction.

        Args
        ----
        sender : Account
            The account object representing the sender's address.
        recipient : Account
            The account object representing the recipient's address.
            The amount of cryptocurrency to transfer (in wei).
        contract : Contract
            The contract object for interaction or deployment.
        func_name : str
            The name of the function to call on the smart contract
        func_params : dict
            The parameters for the contract function call, if applicable.

        Returns
        -------
        dict
            A dictionary containing the transaction payload, including fields such as 'from', 'to', 'value', 'gasPrice', 'gas', and 'data'.
        """
        nonce = self.get_nonce(address=sender.address)
        
        if contract:
            logger.info('>> Smart Contract Transaction')
            logger.info(f'{func_name=}')
            logger.info(f'{func_params=}')
        
            encoded_params = self._encode_nested_dict_to_bytes(func_params)
            function_call = contract.contract.encodeABI(fn_name=func_name, args=encoded_params)
            payload = {
                'from': sender.address,
                'to': contract.contract_address,
                'value': 0,  # Value in Wei, typically 0 for function calls
                'data': function_call,  # Encoded function call
                'maxFeePerGas': self.get_max_fee_per_gas(),
                'maxPriorityFeePerGas': self.get_max_priority_fee(),
                'gas': 100000,  # Set an appropriate gas limit for the transaction
                'nonce': nonce,
                'chainId': self.chain_id
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

        try:
            logger.info(">> Payload:\n%s", json.dumps(payload, indent=4))
        except TypeError:
            # First attempt with logger_hexbytes
            try:
                logger_hexbytes(level='info', data=payload)
            except TypeError:
                # If logger_hexbytes also fails, log a warning
                logger.warning(">> Unable to display payload")
                    
        return payload

    def send_tx(self, sender: Account, payload: dict, contract: Contract):
        """Sign and send a transaction, and handle the post-transaction steps.

        Args
        ----
        sender : Account
            The account object representing the sender's address, including private key.
        payload : dict
            The transaction payload to be signed and sent.
        contract : Contract
            The contract object related to the transaction. Used for post-transaction contract updates.

        Returns
        -------
        None
        """
        # sign & send the transaction
        sender_pk = sender.private_key
        signed_tx = self.provider.eth.account.sign_transaction(payload, sender_pk)
        hashed_tx = self.provider.eth.send_raw_transaction(signed_tx.rawTransaction)

        # store the result of transaction
        tx_receipt = self.provider.eth.wait_for_transaction_receipt(hashed_tx)
        
        if tx_receipt.status == 1:
            logger.info(f"Successfully completed the transaction.")
            logger_hexbytes(level='info', data=tx_receipt)
        else:
            logger.error("The transaction failed or reverted.")
            logger_hexbytes(level='error', data=tx_receipt)

        if contract:
            contract_address = tx_receipt.contractAddress