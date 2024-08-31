import time
import json
import argparse
import traceback

from backend.driver import driver
from backend.util.executor import node_launcher

import logging
from logging import getLogger
from colorlog import ColoredFormatter

logger = getLogger(__name__)
level = logging.DEBUG

formatter = ColoredFormatter(
    "%(log_color)s%(asctime)s [%(levelname)s] %(message)s%(reset)s",
    datefmt='%Y-%m-%d %H:%M:%S',
    reset=True,
    log_colors={
        'DEBUG': 'light_black',
        'INFO': 'green',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'red,bg_white',
    }
)

# Get the root logger
logger = logging.getLogger()
logger.setLevel(level=level)

# Add a stream handler with the colored formatter
stream_handler = logging.StreamHandler()
stream_handler.setLevel(level=level)
stream_handler.setFormatter(formatter)

# Add the handler to the logger
logger.addHandler(stream_handler)
        
with open('config.json') as f:
    jf = json.load(f)
    version = jf['CLI']['version']

class ArgParse(object):
    def __init__(self) -> None:
        # define parsers
        self.parser = argparse.ArgumentParser(description="bothnode CLI")
    
        # command and args
        self.cmd = ["launch", "init", "get", "tx", "detect"]
        self.tgt = ["block_info", "nonce", 'chain_info', 'gas_price', 'queue']
        self.parser.add_argument("command", help="Command to execute", choices=self.cmd)
        self.parser.add_argument("net", help="Network name (e.g., ganache)")
        self.parser.add_argument("target", nargs='?', help="Target for the comamnd", choices=self.tgt)

        # common options
        self.parser.add_argument("-v", "--version", action='version', version=f"bothnode v{version}")
        self.parser.add_argument("-p", "--protocol", default='HTTPS')

        # get related options
        self.parser.add_argument("-q", "--query-params", type=self._dict_parser, help="Query parameters in dictionary format")

        # tx related options
        self.parser.add_argument("-f", "--sender-address", help="The address for the sender")
        self.parser.add_argument("-t", "--recipient-address", help="The address for the recipient")
        self.parser.add_argument("-a", "--amount", type=int)
        self.parser.add_argument("-b", "--build", action='store_true', default=False)
        self.parser.add_argument("--contract-name")
        self.parser.add_argument("--contract-params", type=self._dict_parser, help="Constructor parameters in dictionary format")
        self.parser.add_argument("--func-name", help="name of the function (i.e. method) to call")
        self.parser.add_argument("--func-params", type=self._dict_parser, help="Smart contract method parameters in dictionary format")

        # detect related options
        self.parser.add_argument("-m", "--method", choices=['SVM','GNN'])

        # parse the args
        self._parse_args()

    def _parse_args(self):
        self.args = self.parser.parse_args()
        logger.debug(f'{self.args=}')
    
    def _dict_parser(self, value):
        try:
            parsed_dict = json.loads(value)
            return parsed_dict
        except json.JSONDecodeError:
            raise argparse.ArgumentTypeError(f"Invalid dictionary format: {value}")

def draw_ascii_art():
    pattern = [
        "__.           __  .__                      .___       ",
        "\_ |__   _____/  |_|  |__   ____   ____   __| _/____  ",
        " | __ \ /  _ \   __\  |  \ /    \ /  _ \ / __ |/ __ \ ",
        " | \_\ (  <_> )  | |   Y  \   |  (  <_> ) /_/ \  ___/ ",
        " |___  /\____/|__| |___|  /___|  /\____/\____ |\___  >",
        "     \/                 \/     \/            \/    \/ ",
     ]

    for i, line in enumerate(pattern):
            print(line)
            time.sleep(.025)
    print('\n')
    
def handler(args: argparse.Namespace):
    if args.command == 'launch':
        logger.info(f"Launching the network: {args.net}")
        node_launcher(net_name=args.net)
    
    else:
        logger.info(f"Executing the command: {args.command}")
        net = driver.init_net_instance(net_name=args.net, protocol=args.protocol)

        if args.command == 'get':
            logger.info("Query the network")
            driver.query_handler(net=net, target=args.target, query_params=args.query_params)
            
        elif args.command == 'tx':
            logger.info("Starting a transaction...")
            driver.send_transaction(net=net,
                                    sender_address=args.sender_address,
                                    recipient_address=args.recipient_address,
                                    amount=args.amount,
                                    build=args.build,
                                    contract_name=args.contract_name,
                                    contract_params=args.contract_params,
                                    func_name=args.func_name,
                                    func_params=args.func_params)
     
        elif args.command == 'frun':
            logger.info("Commencing a front-run...")
            driver.front_runner(net=net, sender_address=args.sender_address)
     
        elif args.command == 'detect':
            driver.detect_anamolies(method=args.method)
  
def main():
    argparser = ArgParse()

    if argparser.args.command:
        handler(args=argparser.args)
         
if __name__ == '__main__':
    main()
