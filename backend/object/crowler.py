import json
import requests
from bs4 import BeautifulSoup
from typing import List, Dict

from backend.util.config import Config
from logging import getLogger


logger = getLogger(__name__)
config = Config()
extl_config = config.EXTL_CONFIG


class LabelCrowler(object):
    def __init__(self):
        pass


class CryptoScamDBCrowler(object):
    def __init__(self, extl_config: dict):
        self.endpoint = extl_config['cryptoScamD']
    
    def _get_req_urls(self) -> None:
        """obtain lists of http request URLs
        
        """   
        res = requests.get(self.endpoint)
        logger.debug(res.text)
        soup = BeautifulSoup(response.content, "html.parser")
        
        # parse res to obtrain the list
         
        self.req_url_list - []
    
    def _get_scam_info(self):
        """parse the response to obrain
        """
        
    def crowl_executor(self):
        """
        """
        self._get_req_urls()
        for url in self.req_url_list:
            self._get_scam_info()
        
           

class EtherScanCrowler(object):
    def __init__(self):
        pass
