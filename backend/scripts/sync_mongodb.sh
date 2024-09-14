#!/bin/bash

# Load variables from the .env file
ENV_FILE_PATH="./.env"   # Specify the path to your .env file here
if [ -f "$ENV_FILE_PATH" ]; then
    source "$ENV_FILE_PATH"
else
    echo "Error: .env file not found at $ENV_FILE_PATH"
    exit 1
fi

# Set the variables 
INSTANCE_ID="$1"
REGION="$2"
MONGODB_CONTAINER_NAME=="$3"
MONGODB_DB_NAME="$4"

EC2_IP_ADDRESS=$(./get_ec2_ip.sh "$INSTANCE_ID" "$REGION")
EC2_USER="ubuntu" # Change this to your EC2 user
EC2_DUMP_PATH="/tmp/mongo_backup" 

# Dump the remote MongoDB database
echo "Dumping MongoDB from remote EC2 instance..."
ssh -i "$SSH_KEY_PATH" "$EC2_USER@$EC2_IP_ADDRESS" "docker exec $MONGODB_CONTAINER_NAME mongodump --db $MONGO_DB_NAME --out $EC2_DUMP_PATH"

# Transfer the dump to local machine
echo "Transferring MongoDB dump to local machine..."
scp -i "$SSH_KEY_PATH" -r "$EC2_USER@$EC2_IP_ADDRESS:$EC2_DUMP_PATH/$MONGO_DB_NAME" /tmp/

# Restore the database to local MongoDB container
echo "Restoring MongoDB dump to local MongoDB container..."
docker exec -i "$MONGODB_CONTAINER_NAME" mongorestore --db $MONGO_DB_NAME --drop /tmp/$MONGO_DB_NAME

# Clean up
echo "Cleaning up remote dump files..."
ssh -i "$SSH_KEY_PATH" "$EC2_USER@$EC2_IP_ADDRESS" "rm -rf $EC2_DUMP_PATH/$MONGO_DB_NAME"

echo "Synchronization complete!"
