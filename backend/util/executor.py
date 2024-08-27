import os
import shutil
import json
from pathlib import Path
import subprocess

from logging import getLogger

logger = getLogger(__name__)


with open('config.json') as f:
    config_json = json.load(f)
    SCRIPT_DIR = Path(config_json['CONFIG']['script_dir'])
    PRIVATE_DIR = Path(config_json['CONFIG']['private_dir'])
    INSTALL_DIR = SCRIPT_DIR / 'install'
    

def install_service(service_name: str, source_dir: Path, dest_dir: Path):
    install_geth_sh = INSTALL_DIR / 'install_geth.sh'
    install_geth_process = subprocess.run(["bash", str(install_geth_sh)], check=True)
    if install_geth_process.returncode != 0:
        logger.error(f"Error executing {install_geth_sh}: {install_geth_process.stderr.decode('utf-8')}")
        return
    logger.info(f"Executed {install_geth_sh} successfully.")

    install_lighthouse_sh = INSTALL_DIR / 'install_lighthouse.sh'
    install_lighthouse_process = subprocess.run(["bash", str(install_lighthouse_sh)], check=True)
    if install_lighthouse_process.returncode != 0:
        logger.error(f"Error executing {install_lighthouse_sh}: {install_lighthouse_process.stderr.decode('utf-8')}")
        return
    logger.info(f"Executed {install_lighthouse_sh} successfully.")

    source_path = source_dir / f'{service_name}.service'
    dest_path = dest_dir / f'{service_name}.service'

    # Copy the service file to the systemd directory
    shutil.copy2(source_path, dest_path)
    logger.info(f'{service_name}.service copied to {dest_dir}')

    # Set the correct permissions
    os.chmod(dest_path, 0o644)
 
def main():
    # Define paths
    script_dir = Path(__file__).resolve().parent
    service_dir = script_dir / 'service'
    systemd_dir = Path('/etc/systemd/system/')
    
    # List of services to install
    services = ['geth', 'lighthouse']
    
    # Install each service
    for service in services:
        install_service(service, service_dir, systemd_dir)
   
    
def node_launcher(net_name: str):
    logger.info(f'Launching {net_name}')
    
    if net_name.lower() == 'ganache':
        ganache_sh = SCRIPT_DIR / 'ganache.sh'
        subprocess.Popen(["bash", str(ganache_sh), net_name])
    
    else:        
        # Run the clef.sh script to handle interactive commands
        clef_sh = SCRIPT_DIR / "clef.sh"
        clef_process = subprocess.run(["bash", str(clef_sh), net_name], check=True)
        if clef_process.returncode != 0:
            logger.error(f"Error executing {clef_sh}: {clef_process.stderr.decode('utf-8')}")
            return
        logger.info(f"Executed {clef_sh} successfully.")

        # Finally, start geth and lighthouse
        geth_sh = SCRIPT_DIR / 'geth.sh'        
        start_services_process = subprocess.Popen(["bash", str(geth_sh), net_name])
        logger.info(f"Executed {geth_sh} successfully.")