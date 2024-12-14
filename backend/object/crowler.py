import re
import json
import requests
from typing import List, Dict
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from backend.util.config import Config
from logging import getLogger

logger = getLogger(__name__)


class LabelCrowler(object):
    def __init__(self):
        pass

class CryptoScamDBCrowler(object):
    def __init__(self, extl_config: dict):
        """set endpoint and configure session
        
        Args
        ----
        extl_config : dict
            partial config object
        """
        retries = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[500, 502, 503, 504],
            allowed_methods=["GET"]
        )
        adapter = HTTPAdapter(max_retries=retries)
        
        self.session = requests.Session()
        self.session.mount("https://", adapter)
        self.endpoint = extl_config['cryptoScamDB']
    
    def __del__(self):
        """Ensure the session is closed when the object is deleted.
        """
        if hasattr(self, 'session'):
            self.session.close()
            
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
            data_paths_str = re.sub(r'(?<=\{|,)\s*([a-zA-Z0-9_-]+)\s*:', r'"\1":', match.group(1))            
            try:
                data_paths = {
                    key: value for key, value in json.loads(data_paths_str).items()
                    if key.startswith("domain") or key.startswith("address")}
                return list(data_paths.values())
            except json.JSONDecodeError as e:
                logger.error("Error parsing dataPaths:", e)
        else:
            logger.info("dataPaths not found in the response.")
        
        def _report_json_parser(self):
            pass
    
    def _report_json_parser(self, res_report: dict):
        """send GET request for a single scam page
        
        Args
        ----
        page_url : dict
            returned respoinse
        
        Returns
        -------
        res_report : dict
            parsed response
        """
        logger.debug(res_report.status_code)
        logger.debug(res_report.json())
        
        return res_report

    def get_black_list(self) -> list:
        """obtain lists of reported_addresses
        
        Returns
        -------
        black_list : list[dict]
            list of addresses reported as scam
        """   
        # Get page URLs
        res_pages = self.session.get(self.endpoint)
        page_url_params = self._pages_js_perser(res_pages)
        
        # Generate black list
        black_list = []
        for url_param in page_url_params:
            page_url = f'https://cryptoscamdb.org/static/d/{url_param}.json'            
            try:
                logger.info(f'{page_url=}')
                res_report = self.session.get(page_url)
                res_report.raise_for_status()
                report_json = self._report_json_parser(res_report)
                black_list.append(report_json)
            except requests.exceptions.RequestException as e:
                logger.error(f"Request failed for {page_url}: {e}")
                raise  # Reraise the exception to handle it in the calling method
            except ValueError as e:
                logger.error(f"Error fetching data for {page_url}: {e}")
                
        return black_list
        
class EtherScanCrowler(object):
    def __init__(self):
        pass
