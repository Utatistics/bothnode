import json
from pathlib import Path

from backend.config import NET_CONFIG
from backend.object.network import Network
from web3 import Web3


def main():
    network = Network(net_config=NET_CONFIG.GANACHE)
    address = None
    path_to_solc = NET_CONFIG.SOLC_DIR / 'sample.solc'
    
    # deploy smart contract
    deploy(network=network, address=address, path_to_solc=path_to_solc)

    
def deploy(network: Network, address, path_to_solc: Path):
    nonce = network.get_nonce(address=address)

    # retrieve smart contract infomation
    with open(NET_CONFIG.SOLC_DIR / 'build_info.json', 'r') as file:
        data = json.load(file)

    # name = 'Sample.sol:BalanceChecker'
    # name = 'Faucet.sol:Faucet'
    name = 'Displacement.sol:Displacement'

    contract_address = None # address not neccesary for contract creation
    abi = data['contracts'][name]['abi']
    bytecode = data['contracts'][name]["bin"]

    # define contract creation transaction
    contract = w3.eth.contract(address=contract_address, abi=abi, bytecode=bytecode)
    constructor_args = (agent_address, attacker_address)
    data = contract.constructor(*constructor_args).build_transaction(
        {
            "nonce": nonce,  # Include the nonce here
        }
    )

    # deploy smart contract
    signed_tx = network.provider.eth.account.sign_transaction(data, agent_pk)
    hashed_tx = network.provider.eth.send_raw_transaction(signed_tx.rawTransaction)

    # store the result of transaction
    tx_receipt = network.provider.eth.wait_for_transaction_receipt(hashed_tx)
    contract_address = tx_receipt.contractAddress
    contract = network.provider.eth.contract(address=contract_address, abi=abi, bytecode=bytecode) # post-deployment update
    contract_info = {
        "address": contract.address,
        "abi": contract.abi
    }

    with open(path_to_output / 'contract_info.json', 'w') as json_file:
        json.dump(contract_info, json_file)

    print(f'{tx_receipt.status=}')
