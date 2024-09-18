#!/bin/bash

ROOT_DIR=$(dirname "$(dirname "$(dirname "$(readlink -f "$0")")")")
PRIVATE_DIR="$ROOT_DIR/private"
SCRIPT_DIR="$ROOT_DIR/backend/scripts"

# Load variables from the .env file
ENV_FILE_PATH="$PRIVATE_DIR/.env"   # Specify the path to your .env file here
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
MONGODB_USERNAME="$5"
MONGODB_PASSWORD="$6"

# Select the correct SSH key based on the region
if [ "$REGION" == "ap-northeast-1" ]; then
    SSH_KEY_PATH="$SSH_KEY_TOKYO"
elif [ "$region" == "eu-west-2" ]; then
    SSH_KEY_PATH="$SSH_KEY_LONDON"
else
    echo "Error: Unsupported region $region"
    exit 1
fi

EC2_IP_ADDRESS=$($SCRIPT_DIR/get_ec2_ip.sh "$INSTANCE_ID" "$REGION")
EC2_USER="ubuntu" # Change this to your EC2 user
EC2_DUMP_PATH="/tmp/mongo_backup" 

# ----- DEBUG ----- #
echo $SSH_KEY_PATH
echo $EC2_USER
echo $EC2_IP_ADDRESS
# ----- DEBUG ----- #

# Dump the remote MongoDB database
echo "Dumping MongoDB from remote EC2 instance..."
ssh -i "$SSH_KEY_PATH" "$EC2_USER@$EC2_IP_ADDRESS" "docker exec $MONGODB_CONTAINER_NAME mongodump --db $MONGO_DB_NAME --out $EC2_DUMP_PATH --username $MONGODB_USERNAME --password $MONGODB_PASSWORD --authenticationDatabase admin"

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
