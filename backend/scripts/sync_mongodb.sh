#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Function to handle errors
handle_error() {
    echo "Error occurred in the script at line $1. Exiting."
    echo "Error: $2"
    exit 1
}

# Trap any error and call the handle_error function
trap 'handle_error $LINENO "$BASH_COMMAND"' ERR

ROOT_DIR=$(dirname "$(dirname "$(dirname "$(readlink -f "$0")")")")
PRIVATE_DIR="$ROOT_DIR/private"
SCRIPT_DIR="$ROOT_DIR/backend/scripts"

# Load variables from the .env file
ENV_FILE_PATH="$PRIVATE_DIR/.env"
if [ -f "$ENV_FILE_PATH" ]; then
    source "$ENV_FILE_PATH"
else
    echo "Error: .env file not found at $ENV_FILE_PATH"
    exit 1
fi

# Source the external file with functions
SCRIPT_FILE_PATH="$SCRIPT_DIR/get_ec2_ip.sh"
if [ -f "$SCRIPT_FILE_PATH" ]; then
    source "$SCRIPT_FILE_PATH"
else
    echo "Error: functions.sh file not found at $SCRIPT_FILE_PATH"
    exit 1
fi

# Set the variables 
INSTANCE_ID="$1"
REGION="$2"
MONGODB_CONTAINER_NAME="$3"
MONGODB_DB_NAME="$4"
MONGODB_USERNAME="$5"
MONGODB_PASSWORD="$6"

# Select the correct SSH key based on the region
if [ "$REGION" == "ap-northeast-1" ]; then
    SSH_KEY_PATH="$SSH_KEY_TOKYO"
elif [ "$REGION" == "eu-west-2" ]; then
    SSH_KEY_PATH="$SSH_KEY_LONDON"
else
    echo "Error: Unsupported region $REGION"
    exit 1
fi

# Call the get_ec2_ip_address function from the external file
EC2_IP_ADDRESS=$(get_ec2_ip_address "$INSTANCE_ID" "$REGION")

# Check if the IP was retrieved
if [ -z "$EC2_IP_ADDRESS" ]; then
    echo "Failed to retrieve IP address from get_ec2_ip_address"
    exit 1
else
    echo "EC2 IP address retrieved: $EC2_IP_ADDRESS"
fi

EC2_USER="ubuntu"
EC2_DUMP_PATH="/tmp/mongo_backup" 
LOCAL_DUMP_PATH="/tmp/mongo_backup"

# Remote: MongoDB -> Remote: container 
echo "--> Dumping MongoDB dump into remote container..."
ssh -i "$SSH_KEY_PATH" "$EC2_USER@$EC2_IP_ADDRESS" \
  "docker exec $MONGODB_CONTAINER_NAME mongodump \
  --db $MONGODB_DB_NAME \
  --out $EC2_DUMP_PATH \
  --username $MONGODB_USERNAME \
  --password $MONGODB_PASSWORD \
  --authenticationDatabase admin"

# Remote: container -> Remote: VM
echo "--> Copying MongoDB dump from remote container to EC2 instance..."
ssh -i "$SSH_KEY_PATH" "$EC2_USER@$EC2_IP_ADDRESS" \
  "docker cp \
  $MONGODB_CONTAINER_NAME:$EC2_DUMP_PATH/. \
  $EC2_DUMP_PATH"

# Remote: VM -> Local: VM
echo "--> Transferring MongoDB dump from EC2 instance into local machine..."
scp -i "$SSH_KEY_PATH" \
  -r "$EC2_USER@$EC2_IP_ADDRESS:$EC2_DUMP_PATH" \
  "$LOCAL_DUMP_PATH"

# Local: VM -> Local: container
echo "--> Copying MongoDB dump from local machine into local container..."
docker exec "$MONGODB_CONTAINER_NAME" mkdir -p "$LOCAL_DUMP_PATH"
docker cp $LOCAL_DUMP_PATH/. "$MONGODB_CONTAINER_NAME:$LOCAL_DUMP_PATH"

echo "--> Restoring the database from the dump"
docker exec -i "$MONGODB_CONTAINER_NAME" \
  mongorestore --drop \
  --nsInclude="$MONGODB_DB_NAME.*" \
  --username $MONGODB_USERNAME \
  --password $MONGODB_PASSWORD \
  --authenticationDatabase admin \
  "$LOCAL_DUMP_PATH"

# Clean up local storage 
echo "--> Cleaning up dump files..."
rm -rf "$LOCAL_DUMP_PATH"
docker exec "$MONGODB_CONTAINER_NAME" rm -rf "$LOCAL_DUMP_PATH"

echo "--> Synchronization complete!"
