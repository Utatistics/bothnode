import os
import shutil
import json
from pathlib import Path
import subprocess

from logging import getLogger

logger = getLogger(__name__)

SYSTEM_DIR = Path('/etc/systemd/system/')

with open('config.json') as f:
    config_json = json.load(f)
    SCRIPT_DIR = Path(config_json['CONFIG']['script_dir'])
    PRIVATE_DIR = Path(config_json['CONFIG']['private_dir'])
    INSTALL_DIR = SCRIPT_DIR / 'install'
    SERVICE_DIR = SCRIPT_DIR / 'service'

def install_service(service_name: str) -> None:
    install_sh = INSTALL_DIR / f'install_{service_name}.sh'
    install_process = subprocess.run(["bash", str(install_sh)], check=True)
    if install_process.returncode != 0:
        logger.error(f"Error executing {install_sh}: {install_process.stderr.decode('utf-8')}")
        return
    logger.info(f"Executed {install_sh} successfully.")

def setup_service(service_name: str):
    source_path = SCRIPT_DIR / f'{service_name}.service'
    dest_path = SERVICE_DIR / f'{service_name}.service'

    # copying the .service file 
    if dest_path.exists():
        logger.warning(f"{dest_path} already exists.")
    else:
        try:
            subprocess.run(['sudo', 'cp', str(source_path), str(dest_path)], check=True)
            logger.info(f'{service_name}.service copied to {SYSTEM_DIR}')
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to copy {service_name}.service to {SYSTEM_DIR}: {e}")    
    
    # set permissions for the .service file
    try:
        subprocess.run(['sudo', 'chmod', '644', str(dest_path)], check=True)
        logger.info(f'Set permissions for {dest_path} to 644.')
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to set permissions for {dest_path}: {e}")

    # reload the systemd configuration
    try:
        subprocess.run(['sudo', 'systemctl', 'daemon-reload'], check=True)
        logger.info(f'Reloaded systemd configuration successfully.')
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to reload systemd configuration: {e}")
                     
def node_launcher(net_name: str) -> None: 
    logger.info(f'Launching {net_name}')
    
    if net_name.lower() == 'ganache':
        ganache_sh = SCRIPT_DIR / 'ganache.sh'
        subprocess.Popen(["bash", str(ganache_sh), net_name])
    else:
        services = ['geth', 'lighthouse']        
        for service in services:
            install_service(service_name=service)
            setup_service(service_name=service)

        # Run the clef.sh script to handle interactive commands
        clef_sh = SCRIPT_DIR / "clef.sh"
        clef_process = subprocess.run(["bash", str(clef_sh), net_name], check=True)
        if clef_process.returncode != 0:
            logger.error(f"Error executing {clef_sh}: {clef_process.stderr.decode('utf-8')}")
            return
        logger.info(f"Executed {clef_sh} successfully.")

