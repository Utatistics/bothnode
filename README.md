# bothnode
welcome to bothnode.

    __.           __  .__                      .___       
    \_ |__   _____/  |_|  |__   ____   ____   __| _/____  
     | __ \ /  _ \   __\  |  \ /    \ /  _ \ / __ |/ __ \ 
     | \_\ (  <_> )  | |   Y  \   |  (  <_> ) /_/ \  ___/ 
     |___  /\____/|__| |___|  /___|  /\____/\____ |\___  >
         \/                 \/     \/            \/    \/ 

### Ethereum Smartcontracs (DeFi) Mitigating the Manipulation
What you can do with bothnode:
 - setup a node with the choice of your network
 - interact with a node for basic operation 
 - detect the anamolies and malpractiecs using the variety of methods

### Getting Started 
Frist, clone the repository. 
```bash
git clone https://github.com/your-username/bothnode.git
cd bothnode
```

#### Remote Settings 
* Skip this step if not necessary.
To install Terraform, follow the instructions provided in the [official documentation](https://developer.hashicorp.com/terraform/install).

Use terraform to activate cloud resources in accordance with the predefined configuration. 
```bash
terraform init 
terraform plan
terraform apply
```

Connect to remote server with SSH
```bash
ssh 
```

#### Set up your Node
Then, set up ethereum client by running the following command:
```bash
./ethnode/pkg_install.sh
./ethnode/start_geth.sh
./ethnode/start_lighthouse.sh
```

To start using CLI tool, go to the top of the project directory and run:
```bash
pip install -e .
bothnode launch <network_name>
bothnode init <network_name>
```
'launch' will activate the node of your choice, and 'init' will kick start the background process that allows you to interact the node.

### Interact with Node
bothnode implements multiple ways of node interaction. For example, to send transaction to the network:
```bash
bothnode tx
```

Specify the method and apply detection algorithms to the living network!
```bash
bothnode detect --method <method_name>
```

see --help for the available commands and options.

