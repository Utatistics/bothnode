#!/bin/bash

if ! command -v lighthouse &> /dev/null
then
    echo "lighthouse not found, installing..."
    curl -LO https://github.com/sigp/lighthouse/releases/download/v4.0.1/lighthouse-v4.0.1-x86_64-unknown-linux-gnu.tar.gz
    tar -xvf lighthouse-v4.0.1-x86_64-unknown-linux-gnu.tar.gz
    sudo mv lighthouse /usr/local/bin/
    rm -rf lighthouse-v4.0.1-x86_64-unknown-linux-gnu.tar.gz
else
    echo "lighthouse is already installed"
fi
