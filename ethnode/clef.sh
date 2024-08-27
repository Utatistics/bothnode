#!/bin/bash

# set path
ROOT_DIR=$(dirname "$(dirname "$(readlink -f "$0")")")
PRIVATE_DIR="$ROOT_DIR/private"

# generate an account key pair while specifying where to store them
clef newaccount --keystore $PRIVATE_DIR/keystore

echo '>>> Starting clef...'
nohup clef --keystore $PRIVATE_DIR/keystore --configdir $PRIVATE_DIR/clef --chainid $CHAIN_ID > $HOME/.bothnode/log/clef.log 2>&1 &
