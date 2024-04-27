import json
from pathlib import Path
import subprocess

from logging import getLogger

logger = getLogger(__name__)


with open('config.json') as f:
    config_json = json.load(f)
    SCRIPT_DIR = Path(config_json['CONFIG']['script_dir'])

def node_launcher(net_name: str):
    if net_name.lower() == 'ganache':
        endpoint = 'ganache.sh'
    elif net_name.lower() == 'main':
        endpoint = 'start.sh'
    path_to_sh = SCRIPT_DIR / endpoint

    logger.info(f'Launching {net_name}')
    
    try:
        subprocess.run(["bash", path_to_sh], check=True)
        logger.info(f"Executed {path_to_sh} successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error executing shell script: {e}")
