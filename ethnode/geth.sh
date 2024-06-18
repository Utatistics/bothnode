#!/bin/bash

# set path
ROOT_DIR=$(dirname "$(dirname "$(readlink -f "$0")")")
INSTALL_DIR="$ROOT_DIR/ethnode/install"
PATH_TO_CONFIG="$ROOT_DIR/config.json"

# set parameters
NETWORK=$(echo "$1" | tr '[:lower:]' '[:upper:]')
NETWORK_NAME=$1
AUTHRPC_PORT=$(jq -r --arg NETWORK "$NETWORK" '.NETWORK[$NETWORK].authrpc_port' "$PATH_TO_CONFIG")
SYNC_URL=$(jq -r --arg NETWORK "$NETWORK" '.NETWORK[$NETWORK].checkpoint-sync-url' "$PATH_TO_CONFIG")
CHAIN_ID=$(jq -r --arg NETWORK "$NETWORK" '.NETWORK[$NETWORK].chain_id' "$PATH_TO_CONFIG")

# install 
bash "$INSTALL_DIR/install_geth.sh"
bash "$INSTALL_DIR/install_lighthouse.sh"


# create a JWT secret file
sudo mkdir -p /secrets
openssl rand -hex 32 | tr -d "\n" | sudo tee /secrets/jwt.hex

# start lighthouse
lighthouse bn \
  --network $NETWORK_NAME \
  --execution-endpoint http://localhost:$AUTHRPC_PORT \
  --execution-jwt ~bothnode/private/jwtsecret \
  --checkpoint-sync-url $SYNC_URL \
  --http

# start clef
clef newaccount --keystore ~bothnode/private/keystore
clef --keystore ~bothnode/private/keystore --configdir ~bothnode/private/clef --chainid $CHAIN_ID

# start geth
geth --$NETWORK_NAME \
     --datadir ~bothnode/private \
     --authrpc.addr localhost \
     --authrpc.port $AUTHRPC_PORT \
     --authrpc.vhosts localhost \
     --authrpc.jwtsecret ~bothnode/private/jwtsecret \
     --http \
     --http.api eth,net \
     --signer=~bothnode/private/clef/clef.ipc \
     --http

