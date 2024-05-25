#!/bin/bash

# set path
ROOT_DIR=$(dirname "$(dirname "$(readlink -f "$0")")")
PRIVATE_DIR="$ROOT_DIR/private"
ACCOUNT_KEYS_PATH="$PRIVATE_DIR/ganache_pk.json"
PATH_TO_CONFIG="$ROOT_DIR/config.json"
echo ">>> private keys will be stored in: $ACCOUNT_KEYS_PATH"

# load config values from config.json
GANACHE_CHAIN_ID=$(jq -r '.NETWORK.GANACHE.chain_id' "$PATH_TO_CONFIG")
GANACHE_PORT=$(jq -r '.NETWORK.GANACHE.rpc_port' "$ROOT_DIR/config.json")

# launch ganache server
ganache --chain.chainId=$GANACHE_CHAIN_ID \
        --port=$GANACHE_PORT \
        --wallet.accountKeysPath=$ACCOUNT_KEYS_PATH \
        --wallet.defaultBalance=1000000 \
        --chain.asyncRequestProcessing=true \
        --miner.blockGasLimit=1000000000000 \
