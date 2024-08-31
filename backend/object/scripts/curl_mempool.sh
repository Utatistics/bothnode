#!/bin/bash

# Query the mempool using the geth RPC call
curl -X POST \
  -H "Content-Type: application/json" \
  --data '{"jsonrpc":"2.0","method":"txpool_content","params":[],"id":1}' \
  http://localhost:8545
