from logging import getLogger

logger = getLogger(__name__)

from web3.datastructures import AttributeDict
import json

def convert_to_serializable(data):
    if isinstance(data, bytes):
        return data.hex()
    elif isinstance(data, list):
        return [convert_to_serializable(item) for item in data] # Recursively convert list items
    elif isinstance(data, (dict, AttributeDict)):
        return {k: convert_to_serializable(v) for k, v in data.items()} # Recursively convert dictionary values
    else:
        return data

def logger_hexbytes(level: str, data: object):
    data_serializable = convert_to_serializable(data)

    if level == 'info':
        logger.info(">> \n%s", json.dumps(data_serializable, indent=4))
    if level == 'error':
        logger.error(">> \n%s", json.dumps(data_serializable, indent=4))
