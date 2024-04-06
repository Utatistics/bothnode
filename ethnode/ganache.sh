#!/bin/bash

# set path
ROOT_DIR=$(dirname "$(dirname "$(readlink -f "$0")")")
PRIVATE_DIR="$ROOT_DIR/private"
ACCOUNT_KEYS_PATH="$PRIVATE_DIR/pk.json"

echo $ROOT_DIR
echo $PRIVATE_DIR
echo $ACCOUNT_KEYS_PATH

# load config values from config.json
GANACHE_CHAIN_ID=$(jq -r '.NETWORK.GANACHE.chain_id' "$ROOT_DIR/config.json")
GANACHE_PORT=$(jq -r '.NETWORK.GANACHE.port' "$ROOT_DIR/config.json")

# launch ganache server
ganache --chain.chainId=$GANACHE_CHAIN_ID \
        --port=$GANACHE_PORT \
        --wallet.accountKeysPath=$ACCOUNT_KEYS_PATH \
        --wallet.defaultBalance=1000000 \
        --chain.asyncRequestProcessing=true \
        --miner.blockGasLimit=1000000000000 \
