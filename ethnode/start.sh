#!/bin/bash

# set path
ROOT_DIR=$(dirname "$(dirname "$(readlink -f "$0")")")
INSTALL_DIR="$ROOT_DIR/ethnode/install"
PATH_TO_CONFIG="$ROOT_DIR/config.json"

# set parameters
NETWORK=$(echo "$1" | tr '[:lower:]' '[:upper:]')
NETWORK_NAME=$1
AUTHRPC_PORT=$(jq -r --arg NETWORK_NAME "$NETWORK" '.NETWORK[$NETWORK].authrpc_port' "$PATH_TO_CONFIG")

echo $ROOT_DIR
echo $INSTALL_DIR
echo $PATH_TO_CONFIG
echo $NETWORK
echo $NETWORK_NAME
echo $AUTHRPC_PORT

# install 
bash "$INSTALL_DIR/install_geth.sh"
bash "$INSTALL_DIR/install_lighthouse.sh"

echo $AUTHRPC_PORT
# start geth
geth --$NETWORK_NAME \
     --datadir ~/geth-tutorial \
     --authrpc.addr localhost \
     --authrpc.port $AUTHRPC_PORT \
     --authrpc.vhosts localhost \
     --authrpc.jwtsecret ~/geth-tutorial/jwtsecret \
     --http \
     --http.api eth,net \
     --signer=~/geth-tutorial/clef/clef.ipc \
     --http

# start lighthouse
lighthouse bn \
  --network $NETWORK_NAME \
  --execution-endpoint http://localhost:$AUTHRPC_PORT \
  --execution-jwt ~/geth-tutorial/jwtsecret \
  --checkpoint-sync- https://$NETWORK_NAME.beaconstate.info \
  --http
