#!/bin/bash

# set path
ROOT_DIR=$(dirname "$(dirname "$(readlink -f "$0")")")
PATH_TO_CONFIG="$ROOT_DIR/config.json"

# set parameters
NETWORK_NAME=$1
AUTHRPC_PORT=$(jq -r --arg NETWORK_NAME "$NETWORK_NAME" '.NETWORK[$NETWORK_NAME].authrpc_port' "$PATH_TO_CONFIG")

# install 
bash "$ROOT_DIR/install_geth.sh"
bash "$ROOT_DIR/install_lighthouse.sh"

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
