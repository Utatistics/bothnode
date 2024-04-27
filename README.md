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

#### Remote Settings 
*Skip this step if not necessary.
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

### Getting Started 
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
bothnode launch <network_name>
```
'launch' activates the node of your choice, which will be running in the background. 

### Interact with Node
bothnode implements multiple ways of node interaction. For example, to query network information:
```bash
bothnode get <network_name> <target> --options
```

To send transaction to the network:
```bash
bothnode tx <network_name> --options
```

Specify the method and apply detection algorithms to the living network!
```bash
bothnode detect --method <method_name>
```

see --help for the available commands and options.

### Documentation
For further information on how to use bothnode, see the official documentation.
