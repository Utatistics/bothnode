#!/bin/bash

# set path
ROOT_DIR=$(dirname "$(dirname "$(readlink -f "$0")")")
PRIVATE_DIR="$ROOT_DIR/private"

# generate an account key pair while specifying where to store them
clef newaccount --keystore $PRIVATE_DIR/keystore
