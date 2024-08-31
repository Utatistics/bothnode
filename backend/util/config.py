import json
from pathlib import Path

class Config(object):
    def __init__(self) -> None:
        self.ROOT_DIR = Path.cwd()
        self.SYSTEM_DIR = Path('/etc/systemd/system/')

        with open('config.json') as f:
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

            # load config data
            self.NET_CONFIG = config_json['NETWORK']
            
            # get the version 
            self.CLI_VERSION = config_json['CLI']['version']