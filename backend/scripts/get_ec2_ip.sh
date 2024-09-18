#!/bin/bash

echo "get_ec2_ip.sh called with instance_id=$1 and region=$2" >&2

# Function to retrieve EC2 instance IP or DNS
get_ec2_ip_address() {
    local instance_id="$1"
    local region="$2"

    # Use AWS CLI to describe the EC2 instance and extract the public IP address
    local ec2_ip_address=$(aws ec2 describe-instances --instance-ids "$instance_id" --region "$region" --query "Reservations[*].Instances[*].PublicIpAddress" --output text)

    # If public IP is empty, try getting the public DNS name as fallback
    if [ -z "$ec2_ip_address" ]; then
        ec2_ip_address=$(aws ec2 describe-instances --instance-ids "$instance_id" --region "$region" --query "Reservations[*].Instances[*].PublicDnsName" --output text)
    fi

    # Check if public IP or DNS name was successfully retrieved
    if [ -z "$ec2_ip_address" ]; then
        echo "Error: Unable to retrieve public IP address or DNS name for instance $instance_id"
        exit 1
    fi

    # Return the IP or DNS name
    echo "$ec2_ip_address"
}
