import subprocess

from backend.util.config import Config
from logging import getLogger

config = Config()
logger = getLogger(__name__)

def run_uvicorn():
    uvicorn_sh = config.SCRIPT_DIR / 'start_uvicorn.sh'
    uvicorn_check = subprocess.run(["pgrep", "-f", "uvicorn"], capture_output=True, text=True)
    if uvicorn_check.returncode != 0:
        logger.info("Starting uvicorn...")
        subprocess.run(["bash", str(uvicorn_sh)], check=True)
    else:
        logger.info("Uvicorn is already running.")

def run_docker_compose():
    docker_sh = config.SCRIPT_DIR / 'start_docker.sh'    
    logger.info("Starting docker-compose services...")
    subprocess.run(["bash", str(docker_sh)], check=True)

def start_bothnode():
    run_uvicorn()
    run_docker_compose()

def sync_mongodb(instance_id: str, region: str, container_name: str, db_name: str):
    mongodb_sh = config.SCRIPT_DIR / 'sync_mongodb.sh'
    logger.info("Sync mongodb instances...")    
    subprocess.run(["bash", str(mongodb_sh), instance_id, region, container_name, db_name], check=True)
    