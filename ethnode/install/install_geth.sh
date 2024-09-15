#!/bin/bash

if ! command -v geth &> /dev/null
then
    echo "geth not found, installing..."
    sudo add-apt-repository ppa:ethereum/ethereum
    sudo apt-get update
    sudo apt-get install -y ethereum
else
    echo "geth is already installed"
fi
