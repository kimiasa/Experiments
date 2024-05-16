#!/bin/bash

# Default value for the continue-on-failure flag
ignore_failure=false

# Initialize config directory variable
config_directory=""

# Function to print usage
print_usage() {
    echo "Usage: $0 <config_directory_path> [--ignore-failure]"
}

# Parse the command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --ignore-failure) ignore_failure=true ;;
        *) 
            if [ -z "$config_directory" ]; then
                config_directory=$1
            else
                echo "Unknown parameter passed: $1"
                print_usage
                exit 1
            fi
            ;;
    esac
    shift
done

# Check if the config directory is passed
if [ -z "$config_directory" ]; then
    echo "Config directory is required."
    print_usage
    exit 1
fi

# Ensure the config directory exists
if [ ! -d "$config_directory" ]; then
    echo "Config directory does not exist: $config_directory"
    exit 1
fi


# Create the ./proton_benchmarks directory if it doesn't exist
if [ ! -d "./proton_benchmarks" ]; then
  mkdir ./proton_benchmarks
fi

# Define the relative directory path from the parameter
CONFIG_DIR="$config_directory"

# Extract the last part of the config directory path
DIR_NAME=$(basename "$CONFIG_DIR")

# Find all YAML files in the specified directory, excluding those starting with 'base_'
for file in "$CONFIG_DIR"/*.yaml; 
do
  # Get the base name of the file (without path)
  base_name=$(basename "$file")
  
  # Check if the file name starts with 'base_'
  if [[ "$base_name" != base_* ]]; then
    # Strip the .yaml extension to get the file name
    file_name="${base_name%.yaml}"
    echo Processing $file_name
    echo -------------------------
    # Run the python command with the appropriate arguments
    python run.py experiment="$DIR_NAME"/"$file_name" +exp_name="$DIR_NAME"_"$file_name"
    ret_code=$?

    if [ $ret_code -ne 0 ]; then
        echo "Experiment failed with return code $ret_code"
        if [ "$ignore_failure" = false ]; then
            exit $ret_code
        else
            echo "Continuing despite the failure due to --continue-on-failure flag"
        fi
    else
        echo "Python script executed successfully"
    fi
    echo --------------------------
    # Sleep 2 seconds in between runs
    sleep 2
  fi
done
