import json
from logging import getLogger

logger = getLogger(__name__)

with open('config.json') as f:
    config_json = json.load(f)

from backend.object.network import Network

def init_net_instance(net_name: str, protocol: str):
    logger.info(f"Creating Network instance of {net_name}...")
    net_config = config_json['NETWORK'][net_name.upper()] 
    net = Network(net_config=net_config)

    logger.info(f"The instance connecting to {net_name}...")
    net.connector()


def send_transactions(is_smart_contract: bool):
    print("Sending TX...")

def detect_anamolies(method: str):
    print("Let there be light")