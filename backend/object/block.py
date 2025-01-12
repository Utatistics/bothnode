import json
import requests
from pathlib import Path

from backend.util.config import Config

from logging import getLogger

config = Config()
logger = getLogger(__name__)

class Block(object):
    def __init__(self, net_name: str):
        """Initialize the Block class with network configuration.

        Args
        ----
        net_name : str
            The name of the network (e.g., "mainnet", "testnet") for querying blocks.
        """
        self.net_name = net_name
        self.rpc_url = config.NET_CONFIG[net_name.upper()]['local_rpc']
        self.block_data = {}  # Initialize an empty list to store data for all blocks

    def _query_block_by_num(self, block_num: int) -> dict:
        """Query a specific block by its number.

        Args
        ----
        block_num : int
            The number of the block to query.

        Returns
        -------
        dict
            A dictionary containing the block's details if the query succeeds, otherwise `None`.
        """
        hex_block_number = hex(block_num)  # Convert block number to hexadecimal
        payload = {
            "jsonrpc": "2.0",
            "method": "eth_getBlockByNumber",
            "params": [hex_block_number, True],  # True for including full transaction objects
            "id": block_num
        }
        
        try:
            response = requests.post(self.rpc_url, json=payload)
            if response.status_code == 200:
                logger.info(f"Success: block number={block_num}")
                return response.json()
            else:
                logger.error(f"Failed to fetch block {block_num}. HTTP Status: {response.status_code}")
                return None
        except Exception as e:
            logger.error(f"Error querying block {block_num}: {e}")
            return None
        
    def query_blocks(self, block_num: int, block_len: int) -> None:
        """Direct query for block via RPC: from (block_num - block_len + 1) to block_num.

        Args
        ----
        block_num : int
            The ending block number in the range to query.
        block_len : int
            The number of blocks to query, starting from `block_num - block_len + 1`.

        Returns
        -------
        None
        """
        start_block = block_num - block_len + 1
        end_block = block_num
        logger.info(f'Querying blocks from {start_block} to {end_block}')

        for block_number in range(start_block, end_block + 1):
            rpc_res = self._query_block_by_num(block_number)
            if rpc_res and 'result' in rpc_res:
                payload = rpc_res['result']
                # logger.warning(f"{rpc_res['result']=}")
                block_hash = rpc_res['result']['hash']
                self.block_data[block_hash] = payload  # Append each block's data to the list
                logger.info(f"Fetched Block {block_number}")
            else:
                logger.warning(f"Failed to fetch data for Block {block_number}")

        logger.info(f"Finished querying blocks. Total blocks fetched: {len(self.block_data)}")
    
    def aggregate_from_transactions(self, docs: dict) -> None:
        """Aggregate tx data from external DB into blockdata
        
        Args 
        ----
        docs : dict
            externally obtained transaction data
        """
        for address in docs:
            logger.debug(f'{address=}')
            transactions = docs[address]
            for tx in transactions:
                payload = {
                    'hash': tx.get('hash', None)
                    ,'nonce': tx.get('nonce', None)
                    ,'blockHash': tx.get('blockHash', None)
                    ,'blockNumber': tx.get('blockNumber', None)
                    ,'from': tx.get('from', None)
                    ,'to': tx.get('to', None)
                    ,'value': tx.get('value', None)
                    ,'gas': tx.get('gas', None)
                    ,'gasPrice': tx.get('gasPrice', None)
                    ,'input': tx.get('input', None)
                    ,'cumulativeGasUsed': tx.get('cumulativeGasUsed', None)
                    ,'txreceipt_status': tx.get('txreceipt_status', None)
                    ,'gasUsed': tx.get('gasUsed', None)
                    ,'isError': tx.get('isError', None)
                }           
                block_hash = tx['blockHash']
                if block_hash in self.block_data:
                    self.block_data[block_hash]['transactions'].append(payload)
                else:
                    self.block_data[block_hash]= {
                        'hash': tx['blockHash']
                        ,'number': tx['blockNumber']
                        ,'transactions': [payload]
                    }
        
    def write_to_json(self, path_to_json: Path) -> None:
        with open(path_to_json, 'w', encoding='utf-8') as json_file:
            json.dump(self.block_data, json_file, indent=4, ensure_ascii=False)
        