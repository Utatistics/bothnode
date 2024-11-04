from logging import getLogger

logger = getLogger(__name__)

class Wallet:
    def __init__(self, provider, wallet_address: str) -> None:
        self.provider = provider
        self.connected = self._connect()
        self.account = self._set_account(wallet_address=wallet_address)

    def _connect(self) -> bool:
        """Connects to the Web3 provider and checks for MetaMask connection."""
        if self.provider.is_connected():
            logger.info("Connected to Ethereum network")
            self.connected = True
            return True
        else:
            logger.info("Failed to connect to Ethereum network")
            self.connected = False
            return False

    def _set_account(self, wallet_address: str) -> None:
            """Sets the MetaMask account."""
            if not self.web3.is_address(wallet_address):
                raise ValueError("Invalid Ethereum address.")
            self.account = self.web3.to_checksum_address(wallet_address)
            logger.info(f"Account set to {self.account}")

    def get_balance(self) -> float:
        """Fetches the ETH balance of the connected account."""
        if self.account is None:
            raise ValueError("Account not set. Call set_account() first.")
        balance_wei = self.web3.eth.get_balance(self.account)
        balance_eth = self.web3.from_wei(balance_wei, 'ether')
        logger.info(f"Balance: {balance_eth} ETH")
        return balance_eth
