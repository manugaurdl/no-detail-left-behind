#!/bin/bash

# Check if the argument is provided
if [ -z "$1" ]; then
  echo "Error: No commit message provided."
  echo "Usage: ./git_push.sh <commit_message>"
  exit 1
fi

# Assign the argument to a variable
commit_message="$1"

# Run the Git commands
git add .
git commit -m "$commit_message"
git push origin main
