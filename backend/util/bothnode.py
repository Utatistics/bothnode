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

if __name__ == "__main__":
    start_bothnode()