import json
from pathlib import Path
import subprocess

from logging import getLogger

logger = getLogger(__name__)


with open('config.json') as f:
    config_json = json.load(f)
    SCRIPT_DIR = Path(config_json['CONFIG']['script_dir'])
    PRIVATE_DIR = Path(config_json['CONFIG']['private_dir'])

'''
def node_launcher(net_name: str):
    if net_name.lower() == 'ganache':
        endpoint = 'ganache.sh'
    else:
        endpoint = 'geth.sh'
        
    path_to_sh = SCRIPT_DIR / endpoint
    logger.info(f'Launching {net_name}')
    
    try:
        subprocess.Popen(["bash", path_to_sh, net_name])
        logger.info(f"Executed {path_to_sh} successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error executing shell script: {e}")
'''

def node_launcher(net_name: str):
    if net_name.lower() == 'ganache':
        endpoint = 'ganache.sh'
    else:
        endpoint = 'geth.sh'
        
    path_to_sh = SCRIPT_DIR / endpoint
    logger.info(f'Launching {net_name}')
    
    try:
        clef_process = subprocess.Popen(
            ["clef", "newaccount", "--keystore", f"{PRIVATE_DIR}/keystore"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )

        stdout, stderr = clef_process.communicate()

        if clef_process.returncode != 0:
            logger.error(f"Error creating new account: {stderr.decode('utf-8')}")
            return
        
        logger.info(f"Account created: {stdout.decode('utf-8')}")

        # Run the rest of the shell script
        subprocess.Popen(["bash", str(path_to_sh), net_name])
        logger.info(f"Executed {path_to_sh} successfully.")
    except subprocess.CalledProcessError as e:
        logger.error(f"Error executing shell script: {e}")