import json
from pathlib import Path

ROOT_DIR = Path.cwd()
SCRIPT_DIR = ROOT_DIR / 'ehtnode'
BACKEND_DIR = ROOT_DIR / 'backend'
PRIVATE_DIR = ROOT_DIR / 'private'
SOLC_DIR = BACKEND_DIR / 'solc'

with open(ROOT_DIR / 'config.json') as f:
    NET_CONFIG = json.load(f)
