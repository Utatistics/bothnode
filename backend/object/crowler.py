import re
import json
import requests
from bs4 import BeautifulSoup
from typing import List, Dict

from backend.util.config import Config
from logging import getLogger

logger = getLogger(__name__)


class LabelCrowler(object):
    def __init__(self):
        pass

class CryptoScamDBCrowler(object):
    def __init__(self, extl_config: dict):
        self.endpoint = extl_config['cryptoScamDB']

    def get_black_list(self) -> list:
        """obtain lists of reported_addresses
        
        Returns
        -------
        black_list : list[dict]
            list of addresses reported as scam
        """   
        res_pages = requests.get(self.endpoint)        
        page_url_list = self._pages_js_perser(res_pages)
        logger.debug(res_pages.text)
        #soup = BeautifulSoup(response.content, "html.parser")
        
        black_list = []
        for page_url in page_url_list:
            report_json = self._report_json_parser(page_url)
            black_list.append(report_json)
        
        return black_list
                
    def _pages_js_perser(self, res) -> list:
        """parse the response to obrain
        
        Args
        ----
        res : 
            Response of script type (i.e. JavaScript)
        
        Returns
        -------
        data_path : list
            dictionary contaning
        """
        pattern = r"dataPaths\s*:\s*(\{.*?\})"  # Regex to match the `dataPaths` object
        match = re.search(pattern, res.text, re.DOTALL)  # re.DOTALL allows matching across lines
        if match:
            data_paths_str = match.group(1)  # The JSON-like string for dataPaths

            # Step 3: Parse the JSON-like string
            try:
                data_paths = json.loads(data_paths_str)
                return list(data_paths.values())
            except json.JSONDecodeError as e:
                logger.error("Error parsing dataPaths:", e)
        else:
            logger.info("dataPaths not found in the response.")
        
        def _report_json_parser(self):
            pass
        
class EtherScanCrowler(object):
    def __init__(self):
        pass
