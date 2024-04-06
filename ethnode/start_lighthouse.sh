#!/bin/bash

# start lighthouse
lighthouse bn \
  --network sepolia \
  --execution-endpoint http://localhost:8551 \
  --execution-jwt ~/geth-tutorial/jwtsecret \
  --checkpoint-sync-url https://sepolia.beaconstate.info \
  --http

