#!/bin/bash

# start geth
geth --sepolia \
     --datadir ~/geth-tutorial \
     --authrpc.addr localhost \
     --authrpc.port 8551 \
     --authrpc.vhosts localhost \
     --authrpc.jwtsecret ~/geth-tutorial/jwtsecret \
     --http \
     --http.api eth,net \
     --signer=~/geth-tutorial/clef/clef.ipc \
     --http

