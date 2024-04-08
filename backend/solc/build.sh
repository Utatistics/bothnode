#!/bin/bash

# set variable
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <value>"
    exit 1
fi
CONTRACT_NAME="$1"

# set path
BACKEND_DIR=$(dirname "$(dirname "$(readlink -f "$0")")")
SOLC_DIR="$BACKEND_DIR/solc"
CONTRACT_DIR="$SOLC_DIR/$CONTRACT_NAME"

# execute the build command
solc --combined-json abi,bin $CONTRACT_DIR/$CONTRACT_NAME.sol > $CONTRACT_DIR/build.json
