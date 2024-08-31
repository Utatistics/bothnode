#!/bin/bash

# Query the mempool using the geth command
geth --exec 'txpool.content' attach http://localhost:8545
