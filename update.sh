#!/bin/bash

# Define the config file and tag prefix
CONFIG_FILE="config.json"
README_FILE="README.md"
TAG_PREFIX="v"

# Function to increment the version number
increment_version() {
    local version="$1"
    local type="$2"

    IFS='.' read -r -a version_parts <<< "$version"

    case "$type" in
        major)
            version_parts[0]=$((version_parts[0] + 1))
            version_parts[1]=0
            version_parts[2]=0
            ;;
        minor)
            version_parts[1]=$((version_parts[1] + 1))
            version_parts[2]=0
            ;;
        patch)
            version_parts[2]=$((version_parts[2] + 1))
            ;;
        *)
            echo "Invalid update type. Use 'major', 'minor', or 'patch'."
            if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
                return 1  # Graceful exit for sourced script
            else
                exit 1
            fi
            ;;
    esac

    echo "${version_parts[0]}.${version_parts[1]}.${version_parts[2]}"
}

# Check if the correct number of arguments is provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <update_type> <commit_message>"
    echo "Update types: major, minor, patch"
    if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
        return 1  # Graceful exit for sourced script
    else
        exit 1
    fi
fi

UPDATE_TYPE="$1"
COMMIT_MESSAGE="$2"

# Extract the current version from config.json
current_version=$(jq -r '.CLI.version' "$CONFIG_FILE")

# Increment the version based on the type to get the targeted version
new_version=$(increment_version "$current_version" "$UPDATE_TYPE")

# Prompt the user with the targeted version
read -p "Have you updated the README.md file for version ${new_version}? (yes/no): " user_input
if [[ "$user_input" != "yes" ]]; then
    echo "Please update the README.md file before running this script. Exiting."
    if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
        return 0  # Graceful exit for sourced script
    else
        exit 0
    fi
fi

# Update README.md
sed -i "s/Welcome to bothnode.(v[0-9]*\.[0-9]*\.[0-9]*)/Welcome to bothnode.(v$new_version)/" "$README_FILE"

# Update the version in config.json
jq --arg version "$new_version" '.CLI.version = $version' "$CONFIG_FILE" > temp.json && mv temp.json "$CONFIG_FILE"

# Add all changes to git
git add .

# Commit the changes
git commit -m "update: $COMMIT_MESSAGE"

# Tag the commit with the new version
git tag -a "${TAG_PREFIX}${new_version}" -m "Tagging version ${new_version}"

# Push the commit and tags to the remote repository
git push
git push origin "${TAG_PREFIX}${new_version}"

echo "Version updated to $new_version and tagged successfully."
