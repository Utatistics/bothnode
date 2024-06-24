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

import time
import json
import argparse
from blessed import Terminal
import readline
import traceback

from backend.driver import driver
from backend.util.executor import node_launcher
from backend.object.network import Network
        
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

def clear_screen(term: Terminal):
    print(term.clear)

def draw_ascii_art(term: Terminal):
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

def console_mode(term: Terminal, net: Network):
    print(f"*** welcome to bothnode v{version}")
    draw_ascii_art(term)

    # Enable readline for command history
    readline.parse_and_bind('"\e[A": history-search-backward')
    history = []

    while True:
        inputs = input('>>> ').split()
        history.append(inputs)
        cmd = inputs[0]
        args = inputs[1:]
        try:
            if cmd in ['exit', 'q']:
                break
            
            elif cmd.lower() == 'init':
                net_name = args[0]
                protocol = args[1]
                net = driver.init_net_instance(net_name=net_name, protocol=protocol)
                    
            elif cmd == 'get':
                target = args[0]
                logger.info(f"querying {net.name} for the {target}..")
                if target == 'queue':
                    driver.queue_getter(net)
                
                elif target == 'nonce':
                    address = args[1]
                    nonce = driver.nonce_getter(net=net, address=address)
                    print(f'nonce: {nonce}')

            elif cmd == 'tx':
                sender = input('sender address: ')
                recipient = input('recipient address: ')
                amount = input('amount* press Enter for None: ')
                contract_name = input('contract name* press Enter for None: ')
                if contract_name:
                    build = input('build?: ').lower() in ['y', 'yes']
                else:
                    build = None
                driver.send_transaction(net=net,
                                        sender_address=sender,
                                        recipient_address=recipient,
                                        amount=amount,
                                        contract_name=contract_name,
                                        build=build)
            
            else:
                print(f"Hello, {cmd}")
        except Exception as e:
            logger.info(e)

    print("Thanks for using bothnode.")

def handler(args: argparse.Namespace, term: Terminal):
    if args.command == 'launch':
        node_launcher(net_name=args.net)
    else:
        # initiate the net instance
        status = False
        try:
            net = driver.init_net_instance(net_name=args.net, protocol=args.protocol)
            logger.info(f"Successfully initiated network instance: {net.name}")
            status = True
        except AttributeError as err:
            logger.error('Network not specified.')
            logger.debug(f'Error: {err}')
        except KeyError as err:
            logger.error(f'Invalid network name: {args.net}')
            logger.debug(f'Error: {err}')
            logger.debug(traceback.format_exc())

        # handling the command    
        if status:
            if args.command == 'init':
                logger.info("Opening the console...")
                console_mode(term=term, net=args.net)

            elif args.command == 'get':
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
            
            elif args.command == 'detect':
                if args.method:
                    driver.detect_anamolies(method=args.method)
                else:
                    raise ValueError('Method not specified.')        
        else:
            logger.error('Commmand not executed due to the internal error.')

def main():
    term = Terminal()
    argparser = ArgParse()

    if argparser.args.command:
        handler(args=argparser.args, term=term)
    else:
        console_mode(term=term, net=None)
         
if __name__ == '__main__':
    main()
