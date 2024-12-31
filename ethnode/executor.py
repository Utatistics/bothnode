import subprocess

from backend.util.config import Config
from logging import getLogger

config = Config()
logger = getLogger(__name__)

def install_service(service_name: str) -> None:
    """
    Install the service by running its installation script.

    Args
    ----
    service_name : str
        The name of the service to install.
    Returns
    -------
    """
    install_sh = config.INSTALL_DIR / f'install_{service_name}.sh'
    install_process = subprocess.run(["bash", str(install_sh)], check=True)
    if install_process.returncode != 0:
        logger.error(f"Error executing {install_sh}: {install_process.stderr.decode('utf-8')}")
        return
    logger.info(f"Executed {install_sh} successfully.")

def setup_service(service_name: str, net_name: str):
    """
    Set up a systemd service by copying the .service file.

    Args
    ----
    service_name : str
        The name of the service to set up. This should match the .service file in the service directory.

    Returns
    -------
    """

    source_path = config.SERVICE_DIR / f'{service_name}.service'
    dest_path = config.SYSTEM_DIR / f'{service_name}.service'

    # copying the .service file 
    if dest_path.exists():
        logger.warning(f"{dest_path} already exists.")
    else:
        try:
            subprocess.run(['sudo', 'cp', str(source_path), str(dest_path)], check=True)
            logger.info(f'{service_name}.service copied to {config.SYSTEM_DIR}')
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to copy {service_name}.service to {config.SYSTEM_DIR}: {e}")    
    
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
        
    try:
        subprocess.run(['sudo', 'systemctl', 'start' f'`{service_name}.service'], check=True)
        logger.info(f'Enabled systemd configuration successfully.')
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to enable systemd configuration: {e}")

def service_launcher(net_name: str) -> None:
    """
    Launch the appropriate network node based on the specified network name.
    *currently utilize syhstem.d rather than calling .sh (bothnode/ethnode/command)
    
    Args
    ----
    net_name : str
        The name of the network to launch (i.e. ganache, geth & lighthouse)

    Returns
    -------
    """
    logger.info(f'Launching {net_name}')
    
    if net_name.lower() == 'ganache':
        ganache_sh = config.COMMAND_DIR / 'ganache.sh'
        subprocess.Popen(["bash", str(ganache_sh), str(config.ROOT_DIR), net_name])
    else:
        services = ['geth', 'lighthouse']        
        for service in services:
            install_service(service_name=service)
            setup_service(service_name=service)

        # Run the clef.sh script to handle interactive commands
        clef_sh = config.COMMAND_DIR / "clef.sh"        
        try:
            clef_process = subprocess.run(
                ["bash", str(clef_sh), net_name],
                check=True,
                text=True,  # Enables text mode for capturing output as strings
                stdout=subprocess.PIPE,  # Capture standard output
                stderr=subprocess.PIPE   # Capture standard error
                )
            logger.info(f"Output: {clef_process.stdout}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Error executing {clef_sh}: {e.stderr}")            

def node_launcher(net_name: str) -> None: 
    """
    Args
    ----
    net_name : str
        The name of the network to launch (i.e. ganache, geth & lighthouse)

    Returns
    -------
    """
    logger.info(f'Launching {net_name}')
    
    if net_name.lower() == 'ganache':
        ganache_sh = config.COMMAND_DIR / 'ganache.sh'
        subprocess.Popen(["bash", str(ganache_sh), str(config.ROOT_DIR), net_name])
    else:
        # Run the clef.sh script to handle interactive commands
        clef_sh = config.COMMAND_DIR / "clef.sh"        
        try:
            clef_process = subprocess.run(
                ["bash", str(clef_sh), net_name],
                check=True,
                text=True,  # Enables text mode for capturing output as strings
                stdout=subprocess.PIPE,  # Capture standard output
                stderr=subprocess.PIPE   # Capture standard error
                )
            logger.info(f"Output: {clef_process.stdout}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Error executing {clef_sh}: {e.stderr}")
        
        geth_sh = config.COMMAND_DIR / "geth.sh"        
        try:
            geth_process = subprocess.run(
                ["bash", str(geth_sh), net_name],
                check=True,
                text=True,  # Enables text mode for capturing output as strings
                stdout=subprocess.PIPE,  # Capture standard output
                stderr=subprocess.PIPE   # Capture standard error
                )
            logger.info(f"Output: {geth_process.stdout}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Error executing {geth_sh}: {e.stderr}")
            
