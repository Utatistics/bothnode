import json
from pathlib import Path
import subprocess

from logging import getLogger

logger = getLogger(__name__)


with open('config.json') as f:
    config_json = json.load(f)
    SCRIPT_DIR = Path(config_json['CONFIG']['script_dir'])
    PRIVATE_DIR = Path(config_json['CONFIG']['private_dir'])


def node_launcher(net_name: str):
    logger.info(f'Launching {net_name}')
    
    if net_name.lower() == 'ganache':
        ganache_sh = SCRIPT_DIR / 'ganache.sh'
        subprocess.Popen(["bash", str(ganache_sh), net_name])
    
    else:
        # installation scripts
        install_geth_sh = SCRIPT_DIR / 'install_geth.sh'
        install_geth_process = subprocess.run(["bash", str(install_geth_sh)], check=True)
        if install_geth_process.returncode != 0:
            logger.error(f"Error executing {install_geth_sh}: {install_geth_process.stderr.decode('utf-8')}")
            return
        logger.info(f"Executed {install_geth_sh} successfully.")

        install_lighthouse_sh = SCRIPT_DIR / 'install_lighthouse.sh'
        install_lighthouse_process = subprocess.run(["bash", str(install_lighthouse_sh)], check=True)
        if install_lighthouse_process.returncode != 0:
            logger.error(f"Error executing {install_lighthouse_sh}: {install_lighthouse_process.stderr.decode('utf-8')}")
            return
        logger.info(f"Executed {install_lighthouse_sh} successfully.")

        # Run the clef.sh script to handle interactive commands
        clef_sh = SCRIPT_DIR / "clef.sh"
        clef_process = subprocess.run(["bash", str(clef_sh)], check=True)
        if clef_process.returncode != 0:
            logger.error(f"Error executing {clef_sh}: {clef_process.stderr.decode('utf-8')}")
            return
        logger.info(f"Executed {clef_sh} successfully.")

        # Finally, start geth and lighthouse
        geth_sh = SCRIPT_DIR / 'geth.sh'        
        start_services_process = subprocess.Popen(["bash", str(geth_sh), net_name])
        logger.info(f"Executed {geth_sh} successfully.")