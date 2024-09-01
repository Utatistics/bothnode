import json
import yaml
from pathlib import Path

class Config(object):
    def __init__(self) -> None:
        self.ROOT_DIR = Path.cwd()
        self.SYSTEM_DIR = Path('/etc/systemd/system')

        self.path_to_config = Path('config.json')
        self.path_to_docker = Path('docker-compose.yml')        
            
        with open(self.path_to_config) as f:
            config_json = json.load(f)
            
            # set path
            self.BACKEND_DIR = self.ROOT_DIR / Path(config_json['CONFIG']['backend_dir'])
            self.INTERFACE_DIR = self.ROOT_DIR / Path(config_json['CONFIG']['interface_dir'])
            self.PRIVATE_DIR = self.ROOT_DIR / Path(config_json['CONFIG']['private_dir'])
            
            self.OBJECT_DIR = self.BACKEND_DIR / 'object'
            self.SOLC_DIR = self.BACKEND_DIR / 'solc'
            self.SCRIPT_DIR = self.OBJECT_DIR / 'scripts'
            
            self.COMMAND_DIR = self.INTERFACE_DIR / 'command'
            self.INSTALL_DIR = self.INTERFACE_DIR / 'install'
            self.SERVICE_DIR = self.INTERFACE_DIR / 'service'

            # load network config data
            self.NET_CONFIG = config_json['NETWORK']
            
            # load the CLI info
            self.CLI_VERSION = config_json['CLI']['version']
            
            # load DB config data
            self.DB_CONFIG = config_json['DB']
            
        with open(self.path_to_docker) as f:
            config_ymal = yaml.load(f, Loader=yaml.SafeLoader)
            mongodb_service = config_ymal['services']['mongodb']
            self.DB_CONFIG['init_username'] = mongodb_service['environment']['MONGO_INITDB_ROOT_USERNAME']
            self.DB_CONFIG['init_password'] = mongodb_service['environment']['MONGO_INITDB_ROOT_PASSWORD']
            self.DB_CONFIG['init_database'] = mongodb_service['environment']['MONGO_INITDB_DATABASE']
            
            