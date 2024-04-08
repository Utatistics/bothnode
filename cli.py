import logging
from logging import getLogger

logger = getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s-%(levelname)s: %(message)s',
    handlers=[
        logging.StreamHandler()  # Log to console
    ]
)

import time
import json
import argparse
from blessed import Terminal
import readline

from backend.driver import driver
from ethnode.executor import node_launcher
from backend.object.network import Network

with open('config.json') as f:
    jf = json.load(f)
    version = jf['CLI']['version']

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
            time.sleep(.05)
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
                
                elif target == 'nounce':
                    address = args[1]
                    nounce = driver.nounce_getter(net=net, address=address)
                    print(f'nounce: {nounce}')

            elif cmd == 'tx':
                sender = input('sender address: ')
                recipient = input('recipient address: ')
                contract_name = input('contract name: ')
                driver.send_transaction(net=net, sender_address=sender, recipient_address=recipient, contract_name=contract_name)
            
            else:
                print(f"Hello, {cmd}")
        except Exception as e:
            logger.info(e)

    print("Thanks for using bothnode.")

def handler(args: argparse.Namespace, term: Terminal):
    cmd = args.command
    
    if cmd == 'init':
        net = args.net
        protocol = args.protocol
        if net:
            net = driver.init_net_instance(net_name=net, protocol=protocol)
            console_mode(term=term, net=net)
        else:
            raise ValueError('network not specified.')
        
    elif cmd == 'tx':
        issc = args.smart_contract
        driver.send_transaction(tx_type=issc)
    
    elif cmd == 'detect':
        method = args.method
        if method:
            driver.detect_anamolies(method=method)
        else:
            raise Exception('Method not specified.')
    
    elif cmd == 'launch':
        net = args.net
        if net:
            node_launcher(net_name=net)
        else:
            raise ValueError('network not specified.')

    if args.terminal:
        console_mode(term=term)

def main():
    term = Terminal()
    cmd = ["init", "tx", "detect","launch"]
    parser = argparse.ArgumentParser(description="bothnode CLI")
    parser.add_argument("command", nargs='?', help="Command to execute", choices=cmd)
    parser.add_argument("net", help="Network name (e.g., ganache)")

    parser.add_argument("-v", "--version", action='version', version=f"bothnode v{version}")
    parser.add_argument("-t", "--terminal", action='store_true')
    parser.add_argument("-n", "--net")
    parser.add_argument("-p", "--protocol", default='HTTPS')
    parser.add_argument("--smart_contract", action='store_true')
    parser.add_argument("--method", choices=['SVM','GNN'])
    
    args = parser.parse_args()
    if args.command:
        handler(args=args, term=term)
    else:
        console_mode(term=term, net=None)
         
if __name__ == '__main__':
    main()
