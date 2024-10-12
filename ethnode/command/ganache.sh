#!/bin/bash

# set path
ROOT_DIR=$1
PRIVATE_DIR="$ROOT_DIR/private"
INSTALL_DIR="$ROOT_DIR/ethnode/install"
CONFIG_PATH="$ROOT_DIR/config.json"
ACCOUNT_KEYS_PATH="$PRIVATE_DIR/ganache_pk.json"
GANACHE_LOG_PATH="$PRIVATE_DIR/ganache.log"

# create ganache_pk.json
if [ ! -f "$ACCOUNT_KEYS_PATH" ]; then
    echo "ganache_pk.json not found in $ACCOUNT_KEYS_PATH, creating it..."
    mkdir -p "$PRIVATE_DIR"    
    touch "$ACCOUNT_KEYS_PATH"
    echo "{}" > "$ACCOUNT_KEYS_PATH"
    echo "Empty ganache_pk.json file created in $ACCOUNT_KEYS_PATH"
fi

# load config values from config.json
GANACHE_CHAIN_ID=$(jq -r '.NETWORK.GANACHE.chain_id' "$CONFIG_PATH")
GANACHE_PORT=$(jq -r '.NETWORK.GANACHE.rpc_port' "$ROOT_DIR/config.json")

# install
bash "$INSTALL_DIR/install_ganache.sh"

# launch ganache server
nohup ganache --chain.chainId=$GANACHE_CHAIN_ID \
        --port=$GANACHE_PORT \
        --wallet.accountKeysPath=$ACCOUNT_KEYS_PATH \
        --wallet.defaultBalance=1000000 \
        --chain.asyncRequestProcessing=true \
        --miner.blockGasLimit=1000000000000 > $GANACHE_LOG_PATH 2>&1 &
      # --fork https://mainnet.infura.io/v3/YOUR_INFURA_PROJECT_ID@BLOCK_NUMBER
      
echo ">>> private keys have been stored in $ACCOUNT_KEYS_PATH"
echo ">>> run 'tail -f $GANACHE_LOG_PATH' to monitor the process in real-time."
