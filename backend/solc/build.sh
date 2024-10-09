#!/bin/bash

# set variable
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <value>"
    exit 1
fi
CONTRACT_NAME="$1"

# set path
ROOT_DIR=$(dirname "$(dirname "$(dirname "$(readlink -f "$0")")")")
BACKEND_DIR="$ROOT_DIR/backend"
SOLC_DIR="$BACKEND_DIR/solc"
CONTRACT_DIR="$SOLC_DIR/$CONTRACT_NAME"
PATH_TO_SOLC="$CONTRACT_DIR/Main.sol"
PATH_TO_JSON="$CONTRACT_DIR/build_info.json"

# execute the build command
solc --combined-json abi,bin \
    --base-path $CONTRACT_DIR \
    --include-path $ROOT_DIR/node_modules \
    $PATH_TO_SOLC | jq --indent 4 '.' > $PATH_TO_JSON
