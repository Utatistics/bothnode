#!/bin/bash
echo '>>> Starting clef...'

# set path
ROOT_DIR=$(dirname "$(dirname "$(dirname "$(readlink -f "$0")")")")
PRIVATE_DIR=$ROOT_DIR/private
CLEF_DIR="$HOME/.clef"
PATH_TO_CONFIG="$ROOT_DIR/config.json"

# set parameters
NETWORK=$(echo "$1" | tr '[:lower:]' '[:upper:]')
CHAIN_ID=$(jq -r --arg NETWORK "$NETWORK" '.NETWORK[$NETWORK].chain_id' "$PATH_TO_CONFIG")

# create the clef directory
mkdir -p $CLEF_DIR

# generate an account key pair while specifying where to store them
clef --keystore $CLEF_DIR/keystore --configdir $CLEF_DIR/clef --chainid $CHAIN_ID > $HOME/.bothnode/log/clef.log 2>&1 <<EOF &
ok
cat $PRIVATE_DIR/clef_password)
EOF

# start clef
echo "ok" | clef --keystore $CLEF_DIR/keystore --configdir $CLEF_DIR/clef --chainid $CHAIN_ID > $HOME/.bothnode/log/clef.log 2>&1 &