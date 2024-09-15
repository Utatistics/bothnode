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
            exit 1
            ;;
    esac

    echo "${version_parts[0]}.${version_parts[1]}.${version_parts[2]}"
}

# Check if the correct number of arguments is provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <update_type> <commit_message>"
    echo "Update types: major, minor, patch"
    exit 1
fi

UPDATE_TYPE="$1"
COMMIT_MESSAGE="$2"

# Extract the current version from config.json
current_version=$(jq -r '.CLI.version' "$CONFIG_FILE")

# Increment the version based on the type
new_version=$(increment_version "$current_version" "$UPDATE_TYPE")

# update README.md
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
