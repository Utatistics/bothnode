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
        """Query a range of blocks, from (block_num - block_len + 1) to block_num.

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

        self.block_data = []  # Initialize an empty list to store data for all blocks

        for block_number in range(start_block, end_block + 1):
            rpc_res = self._query_block_by_num(block_number)
            if rpc_res and 'result' in rpc_res:
                block_info = rpc_res['result']
                self.block_data.append(block_info)  # Append each block's data to the list
                logger.info(f"Fetched Block {block_number}")
            else:
                logger.warning(f"Failed to fetch data for Block {block_number}")

        logger.info(f"Finished querying blocks. Total blocks fetched: {len(self.block_data)}")
    
    def write_to_json(self, path_to_json: Path) -> None:
        with open(path_to_json, 'w', encoding='utf-8') as json_file:
            json.dump(self.block_data, json_file, indent=4, ensure_ascii=False)
        