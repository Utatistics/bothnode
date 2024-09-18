# bothnode
Welcome to bothnode. (v0.2.0)
```                                                                                              
    )            )      )                  (            
 ( /(         ( /(   ( /(                  )\ )     (   
 )\())   (    )\())  )\())   (       (    (()/(    ))\  
((_)\    )\  (_))/  ((_)\    )\ )    )\    ((_))  /((_) 
| |(_)  ((_) | |_   | |(_)  _(_/(   ((_)   _| |  (_))   
| '_ \ / _ \ |  _|  | ' \  | ' \)) / _ \ / _` |  / -_)  
|_.__/ \___/  \__|  |_||_| |_||_|  \___/ \__,_|  \___|  
                                                        
```

### Ethereum Smartcontracs (DeFi) Mitigating the Manipulation
What you can do with bothnode:
 - setup a node with the choice of your network
 - interact with a node for basic operation 
 - detect the anamolies and malpractiecs using the variety of methods

### Getting Started 
#### Remote Settings 
*Skip this step if not necessary.

Use [bothnode-infra](https://github.com/Utatistics/bothnode-infra/tree/main) to automate cloud resources setup 

#### Install bothnode
First, clone the repository. 
```bash
git clone https://github.com/your-username/bothnode.git
cd bothnode
```

To start using CLI tool, install bothnode to your machine 
```bash
pip install -e .
```

#### Set up your Node
Then, set up ethereum client by running the following command:
```bash
bothnode init <network_name>
```
The command launch the node of your choice: 
 - ganache: local Ethereum emulator
 - sepolia: Ethereum test network
 - main: Ethereum main net

Run the usual `systemctl` commands to manage the underlying services (e.g., geth, lighthouse)
```bash
sudo systemctl start <service_name>.service
sudo systemctl stop <service_name>.service
sudo systemctl status <service_name>.service

sudo systemctl enable <service_name>.service
sudo journalctl -u <service_name>.service -f
```


After successful execution of `bothnode init`, your node will be running in the background while starting the syncing process at the same time.

### Interact with Node
bothnode implements multiple ways of node interaction: get, tx, detect

#### bothnode get
```bash
bothnode get <network_name> <target> --options
```

#### bothnode tx
To send transaction to the network:
```bash
bothnode tx <network_name> --options

# regular transaction
bothnode tx ganache -f <from_address>-t <to_address> --amount 1

# transaction for smart contract deployment
bothnode tx ganache -f <from_address> -b
--contract-name Tokenization
--contract-params '{
  "name_": "My Token",
  "symbol_": "MTK",
  "decimals_": 18,
  "initialSupply_": 1000000
}'

# transaction for smart contract calling
bothnode tx ganache -f <address>
--contract-name Tokenization
--func-name transfer
--func-params '{"recipient": "0xFaD6bF978fC43DD8Dc6084356012f80CB3Ff1b56", "amount": 1000}'
```

#### bothnode detect
Specify the method and apply detection algorithms to the living network!
```bash
bothnode detect --method <method_name>
```

see --help for the available commands and options.

### Documentation
For further information on how to use bothnode, see the official documentation.
