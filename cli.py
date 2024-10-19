import os
import sys
import time
import json
import argparse
import threading
import itertools 

from backend.driver import driver
from ethnode.executor import node_launcher
from backend.util.bothnode import start_bothnode, sync_mongodb

import logging
from logging import getLogger
from colorlog import ColoredFormatter

logger = getLogger(__name__)
level = logging.INFO

formatter = ColoredFormatter(
    "%(log_color)s%(asctime)s [%(levelname)s] %(message)s%(reset)s",
    datefmt='%Y-%m-%d %H:%M:%S',
    reset=True,
    log_colors={
        'DEBUG': 'light_black',
        'INFO': 'light_green',
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
        self.cmd = ['run', 'init', 'db_sync', 'get', 'tx', 'frontrun', 'detect']
        self.parser.add_argument("command", help="Command to execute", choices=self.cmd)       
        
        partial_args, _ = self.parser.parse_known_args()

        self.parser.add_argument("-v", "--verbose")
        
        if partial_args.command == 'db_sync':
            self.parser.add_argument("instance_id")            
            self.parser.add_argument("instance_region", help="region name (e.g., eu-east-2)")            
            self.parser.add_argument("db_name", help="database name (e.g., transaction)")               
        elif partial_args.command != 'run':
            self.parser.add_argument("net", help="Network name (e.g., ganache)")
            self.parser.add_argument("-p", "--protocol", default='HTTPS')
            if partial_args.command == 'get':
                tgt = ['block_info', 'nonce', 'chain_info', 'gas_price', 'queue'] 
                self.parser.add_argument("target", nargs='?', help="Target for the comamnd", choices=tgt)
                self.parser.add_argument("-q", "--query-params", type=self._dict_parser, help="Query parameters in dictionary format")
            if partial_args.command == 'tx':
                self.parser.add_argument("-f", "--sender-address", help="The address for the sender")
                self.parser.add_argument("-t", "--recipient-address", help="The address for the recipient")
                self.parser.add_argument("-a", "--amount", type=int)
                self.parser.add_argument("-b", "--build", action='store_true', default=False)
                self.parser.add_argument("--contract-name")
                self.parser.add_argument("--contract-params", type=self._dict_parser, help="Constructor parameters in dictionary format")
                self.parser.add_argument("--func-name", help="name of the function (i.e. method) to call")
                self.parser.add_argument("--func-params", type=self._dict_parser, help="Smart contract method parameters in dictionary format")
            if partial_args.command == 'frontrun':
                self.parser.add_argument("sender_address", help="The address for the sender")
                self.parser.add_argument("-m", "--method", choices=['SVM','GNN'])        
        
          
        # parse the args
        self._parse_args()

    def _parse_args(self):
        self.args = self.parser.parse_args()
        logger.debug(f'{self.args=}')
   
    def _dict_parser(self, value):
            # Check if the input is a valid file path
            if os.path.isfile(value):
                try:
                    with open(value, 'r') as file:
                        parsed_dict = json.load(file)
                        return parsed_dict
                except (IOError, json.JSONDecodeError) as e:
                    raise argparse.ArgumentTypeError(f"Error reading file: {e}")
            
            # Otherwise, try to parse it as a JSON string
            try:
                parsed_dict = json.loads(value)
                return parsed_dict
            except json.JSONDecodeError:
                raise argparse.ArgumentTypeError(f"Invalid dictionary format: {value}")

class Spinner:
    def __init__(self, message="Processing..."):
        self.spinner = itertools.cycle(['|', '/', '-', '\\'])
        self.stop_running = False
        self.message = message

    def start(self):
        def spin():
            while not self.stop_running:
                sys.stdout.write(f'\r{self.message} {next(self.spinner)}')
                sys.stdout.flush()
                time.sleep(0.1)
            sys.stdout.write(f'\r{self.message} {" " * (len(self.message) + 2)}')  # Clear spinner
            sys.stdout.flush()

        self.spinner_thread = threading.Thread(target=spin)
        self.spinner_thread.start()

    def stop(self):
        self.stop_running = True
        self.spinner_thread.join()

def draw_ascii_art():
    pattern = [
        "    )            )      )                  (            ",
        " ( /(         ( /(   ( /(                  )\ )     (   ",
        " )\())   (    )\())  )\())   (       (    (()/(    ))\  ",
        "((_)\    )\  (_))/  ((_)\    )\ )    )\    ((_))  /((_) ",
        "| |(_)  ((_) | |_   | |(_)  _(_/(   ((_)   _| |  (_))   ",
        "| '_ \ / _ \ |  _|  | ' \  | ' \)) / _ \ / _` |  / -_)  ",
        "|_.__/ \___/  \__|  |_||_| |_||_|  \___/ \__,_|  \___|  ",
    ]
    print(f">>> Welcome to bothnode {version}")
    for line in pattern:
        print(line)
        time.sleep(0.1)
    print('\n')
    
def handler(args: argparse.Namespace):
    if args.command == 'run':
        logger.info(f"Starting bothnode application.")       
        draw_ascii_art()
        start_bothnode()
        
    elif args.command == 'init':
        logger.info(f"Launching the network: {args.net}")
        node_launcher(net_name=args.net)
        
    elif args.command == 'db_sync':
        sync_mongodb(instance_id=args.instance_id, region=args.instance_region, container_name='mongodb', db_name=args.db_name)
        
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
     
        elif args.command == 'frontrun':
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
