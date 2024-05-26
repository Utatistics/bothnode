#!/bin/bash

if ! command -v ganache &> /dev/null
then
    echo "ganache not found, installing..."
    sudo npm install ganache --global
else
    echo "ganache is already installed"
fi
