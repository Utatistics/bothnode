import sys
import subprocess

from backend.util.config import Config
from logging import getLogger

config = Config()
logger = getLogger(__name__)

def run_uvicorn() -> None:
    uvicorn_sh = config.SCRIPT_DIR / 'start_uvicorn.sh'
    uvicorn_check = subprocess.run(["pgrep", "-f", "uvicorn"], capture_output=True, text=True)
    if uvicorn_check.returncode != 0:
        logger.info("Starting uvicorn...")
        subprocess.run(["bash", str(uvicorn_sh)], check=True)
    else:
        logger.info("Uvicorn is already running.")

def run_docker_compose() -> None:
    docker_sh = config.SCRIPT_DIR / 'start_docker.sh'    
    logger.info("Starting docker-compose services...")
    subprocess.run(["bash", str(docker_sh)], check=True)

def start_bothnode() -> None:
    run_uvicorn()
    run_docker_compose()

def sync_mongodb(instance_id: str, region: str, container_name: str, db_name: str) -> None:
    """
    Synchronizes a MongoDB instance from a remote EC2 server to a local MongoDB container

    Args
    ----
    instance_id : str
        The ID of the EC2 instance
    region : str
        The AWS region where the EC2 instance is located.
    container_name : str
        The name of the MongoDB container *assumed to be identical
    db_name : str
        The name of the MongoDB database to sync.

    Returns
    -------
    """
    db_config = config.DB_CONFIG
    username=db_config['init_username']
    password=db_config['init_password']
    
    mongodb_sh = config.SCRIPT_DIR / 'sync_mongodb.sh'
    logger.info("Sync mongodb instances...")    
    
    try:
        # Running the shell script and capturing output while printing it
        subprocess.run(["bash", str(mongodb_sh), instance_id, region, container_name, db_name, username, password],
            check=True, text=True, stdout=sys.stdout, stderr=sys.stderr
        )
        logger.info("MongoDB sync completed successfully.")
    except subprocess.CalledProcessError as e:
        # Error handling with real-time stderr capture
        logger.error(f"Error occurred during MongoDB sync: {e.stderr}")
        raise
    