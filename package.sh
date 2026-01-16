#!/bin/bash
# Packaging script for model.tar.gz
# Assumes model.pth is in the current directory

tar -czvf model.tar.gz model.pth requirements.txt code/ README.md
echo "Created model.tar.gz"