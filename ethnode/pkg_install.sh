#!/bin/bash

# update repository 
sudo add-apt-repository ppa:ethereum/ethereum
sudo apt-get update
sudo apt-get upgrade

# common util 
sudo apt-get -y install tree
sudo apt-get -y install network-manager
sudo apt-get -y install net-tools
sudo apt-get -y install unzip

# geth
sudo apt-get install -y ethereum

# lighthouse
c -LO https://github.com/sigp/lighthouse/releases/download/v4.0.1/lighthouse-v4.0.1-x86_64-unknown-linux-gnu.tar.gz
tar -xvf lighthouse-v4.0.1-x86_64-unknown-linux-gnu.tar.gz
sudo mv lighthouse /usr/local/bin/
rm -rf lighthouse-v4.0.1-x86_64-unknown-linux-gnu.tar.gz
