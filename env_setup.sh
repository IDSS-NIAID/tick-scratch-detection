#!/bin/bash --login
#
# Creates a Conda environment and uses pip to install TensorFlow with its
# bundled CUDA libraries. It will remove any existing environment with the
# same name before starting.
#
# USAGE:
#   ./create_conda_env.sh
#   ./create_conda_env.sh <custom_env_name>

# --- Configuration ---
set -eo pipefail # Exit on error

ENV_NAME="${1:-tick}"
PYTHON_VERSION="3.12"

echo "======================================================="
echo "Creating Conda Environment: $ENV_NAME"
echo "Python Version:           $PYTHON_VERSION"
echo "======================================================="

# --- 1. Load Conda Module ---
echo -e "\n--> Loading miniconda3 module..."
module load miniconda3

# --- 2. Check for and Remove Existing Environment ---
echo -e "\n--> Checking for existing environment named '$ENV_NAME'..."
# Use 'conda info --envs' and grep to check if the environment exists.
# The -q flag makes grep quiet, and -w matches the whole word.
if conda info --envs | grep -q -w "$ENV_NAME"; then
    echo "--> Environment '$ENV_NAME' found. Removing it to ensure a fresh install..."
    conda env remove --name "$ENV_NAME" --yes
    echo "--> Existing environment removed."
else
    echo "--> No existing environment found. Proceeding with creation."
fi

# --- 3. Create a Base Conda Environment ---
# We create a minimal environment with Python. The other packages will be added with pip.
echo -e "\n--> Creating base environment '$ENV_NAME'..."
conda create -n "$ENV_NAME" --yes -c conda-forge \
    python="$PYTHON_VERSION"

echo "--> Base environment created."

# --- 4. Install Packages with Pip ---
# Using the new 'tensorflow[and-cuda]' target which bundles CUDA libraries.
# This command is run inside the new environment.
echo -e "\n--> Installing packages with pip. This may take several minutes..."
conda run -n "$ENV_NAME" python -m pip install \
    'tensorflow[and-cuda]' \
    opencv-python \
    scikit-image

echo "--> Package installation complete."

# --- 5. Final Instructions ---
echo -e "\n--- SETUP COMPLETE ---"
echo "Activate your environment with:"
echo "  conda activate $ENV_NAME"
echo ""
echo "In a job script, you only need to load miniconda3 and activate:"
echo "  module load miniconda3"
echo "  conda activate $ENV_NAME"
echo "======================================================="
