#!/bin/bash

# set path
ROOT_DIR=$(dirname "$(dirname "$(readlink -f "$0")")")
PATH_TO_CONFIG="$ROOT_DIR/config.json"
PRIVATE_DIR="$ROOT_DIR/private"
INSTALL_DIR="$ROOT_DIR/ethnode/install"

# set parameters
NETWORK=$(echo "$1" | tr '[:lower:]' '[:upper:]')
NETWORK_NAME=$1
AUTHRPC_PORT=$(jq -r --arg NETWORK "$NETWORK" '.NETWORK[$NETWORK].authrpc_port' "$PATH_TO_CONFIG")
SYNC_URL=$(jq -r --arg NETWORK "$NETWORK" '.NETWORK[$NETWORK].checkpoint_sync_url' "$PATH_TO_CONFIG")
CHAIN_ID=$(jq -r --arg NETWORK "$NETWORK" '.NETWORK[$NETWORK].chain_id' "$PATH_TO_CONFIG")

# install 
bash "$INSTALL_DIR/install_geth.sh"
bash "$INSTALL_DIR/install_lighthouse.sh"

# create a JWT secret file
sudo mkdir -p /secrets
openssl rand -hex 32 | tr -d "\n" | sudo tee /secrets/jwt.hex

# generate an account key pair while specifying where to store them
# clef newaccount --keystore $PRIVATE_DIR/keystore

echo '>> Start processes in the background.'
echo '>>> Starting lighthouse...'
nohup lighthouse bn \
  --network $NETWORK_NAME \
  --execution-endpoint http://localhost:$AUTHRPC_PORT \
  --execution-jwt $PRIVATE_DIR/jwtsecret \
  --checkpoint-sync-url $SYNC_URL \
  --http > $HOME/.bothnode/log/lighthouse.log 2>&1 &

echo '>>> Starting clef...'
nohup clef --keystore $PRIVATE_DIR/keystore --configdir $PRIVATE_DIR/clef --chainid $CHAIN_ID > $HOME/.bothnode/log/clef.log 2>&1 &

echo '>>> Starting geth...'
nohup geth --$NETWORK_NAME \
  --datadir $PRIVATE_DIR \
  --authrpc.addr localhost \
  --authrpc.port $AUTHRPC_PORT \
  --authrpc.vhosts localhost \
  --authrpc.jwtsecret $PRIVATE_DIR/jwtsecret \
  --http \
  --http.api eth,net \
  --signer=$PRIVATE_DIR/clef/clef.ipc > $HOME/.bothnode/log/geth.log 2>&1 &

wait
