from web3 import Web3

from backend.object.network import Network
from backend.object.account import Account

from logging import getLogger

logger = getLogger(__name__)

def target_criteria(tx):
    return tx['to'] == '0xUniswapContractAddress' and 'swapExactTokensForTokens' in tx['input']

def large_transfer_criteria(tx, min_value_threshold):
    return tx['value'] > min_value_threshold  # for ETH transfers

def arbitrage_opportunity_criteria(tx, exchange_addresses, known_arbitrage_functions):
    is_exchange_transaction = tx['to'] in exchange_addresses
    is_arbitrage_function = any(func in tx['input'] for func in known_arbitrage_functions)    
    return is_exchange_transaction and is_arbitrage_function

class FrontRunner(object):
    def __init__(self, net: Network, sender: Account) -> None:
        """
        Initialize a FrontRunner instance.

        Args
        ----
        net : Network
            The network object used to interact with the blockchain.
        sender : Account
            The account object representing the sender's address.

        Returns
        -------
        None
        """
        self.net = net
        self.account = sender
        self.data = None

    def query_mempool(self) -> list:
        """
        Query the mempool for pending transactions.

        Returns
        -------
        list
            A list of transactions currently in the mempool.
        """
        mempool = self.net.get_queue()  # Assuming get_queue returns the pending transactions
        return mempool

    def lockon_tx(self, mempool: list, target_criteria) -> dict:
        """
        Lock onto a transaction in the mempool that matches the target criteria.

        Args
        ----
        mempool : list
            A list of transactions to search through.
        target_criteria : callable
            A function that takes a transaction (dict) and returns True if it matches the criteria.

        Returns
        -------
        dict
            The transaction that matches the criteria, or None if no match is found.
        """
        mempool = self.query_mempool()
        for tx in mempool:
            if target_criteria(tx):
                self.data = tx
                logger.debug(f"Locked on transaction: {tx['hash']}")
                return tx
        
        logger.warning("No matching transaction found.")
        return None

    def create_payload(self, target_tx: dict) -> dict:
        """
        Create a payload for the front-running transaction.

        Args
        ----
        target_tx : dict
            The target transaction from which to create the front-running payload.

        Returns
        -------
        dict
            The payload for the front-running transaction, or None if no payload could be created.
        """
        if target_tx:
            gas_price = self.net.get_gas_price()
            max_priority_fee = self.net.get_max_priority_fee()
            frontrun_gas_price = gas_price + int(gas_price * 0.2)  # Increase gas price by 20% to front-run

            frontrun_tx = {
                'from': self.account.address,
                'to': target_tx['to'],
                'value': target_tx['value'],
                'gas': target_tx['gas'],
                'gasPrice': frontrun_gas_price,
                'nonce': self.net.get_nonce(),
                'data': target_tx['input']
            }
            logger.info(f"Created payload: {frontrun_tx}")
            return frontrun_tx
        logger.warning("No transaction to create payload from.")
        return None

    def execute_frontrun(self, payload: dict) -> str:
        """
        Sign and send the front-running transaction.

        Args
        ----
        payload : dict
            The transaction payload to be sent.

        Returns
        -------
        None
        """
        if payload:
            self.net.send_tx(sender=self.account.address, payload=payload)
            logger.info(f"Front-running transaction sent.")
        else:
            logger.info('FrontRunner DNS.')

class Arbitrageur(object):
    def __init__(self) -> None:
        pass
