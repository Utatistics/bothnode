import web3
from backend.object.network import Network

class FrontRunner(object):
    def __init__(self, net: Network) -> None:
        self.net = net
    
    def query_mempool(self):
        mempool = self.net.get_queue
        # persisting the mempool data for each frontrunning attempt?
    
    def lockon_tx(self):
        self.data = None
    
    def create_payload(self):
        pass
        # think of the logic to calculate the gas price, etc.
    
    def execute_frontrun(self):
        pass
    
    
