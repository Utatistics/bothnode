import re
import json
import requests
from pathlib import Path
from typing import List, Dict
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
from concurrent.futures import ThreadPoolExecutor, as_completed

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
               
        nodes = res_report.json()['data']['allCsdbScamDomains']['edges']
        num_nodes = len(nodes)
        logger.debug(f'{num_nodes=}')
    
        return [i['node'] for i in nodes]

    def _get_reported_nodes(self, page_url: str) -> None:
        """Fetch a single URL and parse its content.
        
        Args
        ----
        page_url : string
            endpoint url
        
        Returns
        -------
        nodes : list
            reported nodes 
        """
        logger.info(f'{page_url=}')

        try:
            res_report = self.session.get(page_url)
            res_report.raise_for_status()
            return self._report_json_parser(res_report)
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed for {page_url}: {e}")
            return []
        except ValueError as e:
            logger.error(f"Error fetching data for {page_url}: {e}")
            return []
        
    def get_black_list(self, concurrent: bool=False) -> None:
        """obtain lists of reported_addresses
        
        Args
        ----
        concurrent : bool
            execute the method concurrently  
        """ 
        
        res_pages = self.session.get(self.endpoint)
        page_url_params = self._pages_js_perser(res_pages)
        page_urls = [f'https://cryptoscamdb.org/static/d/{param}.json' for param in page_url_params]
        logger.info(f"Found {len(page_urls)} URLs.")
        
        self.black_node_list = [] 
        if concurrent:
            with ThreadPoolExecutor(max_workers=10) as executor:  # Adjust max_workers as needed
                future_to_url = {executor.submit(self._get_reported_nodes, url): url for url in page_urls}
                for future in as_completed(future_to_url):
                    url = future_to_url[future]
                    try:
                        nodes = future.result()
                        self.black_node_list.extend(nodes)
                    except Exception as e:
                        logger.error(f"Unexpected error processing {url}: {e}")
        else:                       
            for page_url in page_urls:
                nodes = self._get_reported_nodes(page_url=page_url)
                self.black_node_list.extend(nodes)

            
    def write_to_json(self, path_to_json: Path) -> None:
        """write the black list to a single json file
        
        Args
        ----
        path_to_json : Path
            path to the .json file
        
        """
        with open(path_to_json, mode="w") as file:
            json.dump(self.black_node_list, file, indent=4)
            
class EtherScanCrowler(object):
    def __init__(self):
        pass
