#!/usr/bin/env bash

# Exit immediately if any command fails
set -e

# 1. Check if python3 is installed
if ! command -v python3 &> /dev/null; then
    echo "Python3 not found. Please install Python 3.10+"
    exit 1
fi

# 2. Create a virtual environment named .venv
python3 -m venv .venv

# 3. Activate the virtual environment
source .venv/bin/activate

# 4. Upgrade pip inside the venv
pip install --upgrade pip

# 5. Install all packages from requirements.txt
pip install -r requirements.txt

# 6. Print confirmation message
echo "Environment ready. Run: source .venv/bin/activate"
