#!/bin/bash

if ! command -v lighthouse &> /dev/null
then
    LATEST_VERSION=$(curl -s https://api.github.com/repos/sigp/lighthouse/releases/latest | jq -r .tag_name)
    echo "lighthouse not found, installing the latest version($LATEST_VERSION)..."
    curl -LO https://github.com/sigp/lighthouse/releases/download/$LATEST_VERSION/lighthouse-$LATEST_VERSION-x86_64-unknown-linux-gnu.tar.gz
    tar -xvf lighthouse-$LATEST_VERSION-x86_64-unknown-linux-gnu.tar.gz
    sudo mv lighthouse /usr/local/bin/
    rm -rf lighthouse-$LATEST_VERSION-x86_64-unknown-linux-gnu.tar.gz
else
    echo ">>lighthouse is already installed"
fi
