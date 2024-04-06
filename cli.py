import logging
from logging import getLogger
import argparse
from blessed import Terminal
import time
import json

from backend import driver

logger = getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s-%(levelname)s: %(message)s',
    handlers=[
        logging.StreamHandler()  # Log to console
    ]
)


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
            time.sleep(.005)
    print('\n')

def console_mode(term: Terminal):
    print(f"welcome to bothnode v{version}")
    draw_ascii_art(term)

    while True:
        cmd = input('>>> ')
        if cmd in ['exit', 'q']:
             break
        elif cmd.lower() == 'init':
            net_name = input("Network? -> ")
            protocol = input("Protocol? -> ")
            net = driver.init_net_instance(net_name=net_name, protocol=protocol)
        elif cmd == 'queue':
             logger.info(f"querying {net.name} for the queue...")
             print(net.url)         
        else:
            print(f"Hello, {cmd}")
            
    print("Thanks for using bothnode.")

def handler(args: argparse.Namespace, term: Terminal):
    cmd = args.command
    
    if cmd == 'init':
        net = args.net
        protocol = args.protocol
        if net:
            driver.init_net_instance(net_name=net, protocol=protocol)
            console_mode(term=term)
        else:
            raise ValueError('network not specified.')
        
    elif cmd == 'tx':
        issc = args.smart_contract
        driver.send_transactions(is_smart_contract=issc)
    
    elif cmd == 'detect':
        method = args.method
        if method:
            driver.detect_anamolies(method=method)
        else:
            raise Exception('Method not specified.')

    if args.terminal:
        console_mode(term=term)

def main():
    term = Terminal()
    cmd = ["init", "tx", "detect"]
    parser = argparse.ArgumentParser(description="bothnode CLI")
    parser.add_argument("command", nargs='?', help="Command to execute", choices=cmd)
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
        console_mode(term=term)
         
if __name__ == '__main__':
    main()
