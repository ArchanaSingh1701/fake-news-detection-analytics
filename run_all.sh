#!/bin/bash

echo "Running Fake News Detection Pipeline..."

# Check if python3-venv is available, if not, install it
if ! python3 -m venv --help > /dev/null 2>&1; then
    echo "Installing python3-venv..."
    sudo apt update && sudo apt install -y python3-venv
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

if [ $? -ne 0 ]; then 
    echo "Dependency installation failed"
    exit 1
fi

echo "Step 1: Generating synthetic data..."
python3 data_prep.py --synthesize
if [ $? -ne 0 ]; then echo "Step 1 failed"; exit 1; fi

echo "Step 2: Building graph..."
python3 graph_builder.py
if [ $? -ne 0 ]; then echo "Step 2 failed"; exit 1; fi

echo "Step 3: Extracting features..."
python3 feature_extractor.py
if [ $? -ne 0 ]; then echo "Step 3 failed"; exit 1; fi

echo "Step 4: Training ML model..."
python3 train_ml.py
if [ $? -ne 0 ]; then echo "Step 4 failed"; exit 1; fi

echo "Step 5: Evaluating model..."
python3 evaluate.py
if [ $? -ne 0 ]; then echo "Step 5 failed"; exit 1; fi

echo "Pipeline completed!"