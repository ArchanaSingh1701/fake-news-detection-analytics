@echo off
echo Running Fake News Detection Pipeline...

echo Step 1: Generating synthetic data...
python data_prep.py --synthesize

echo Step 2: Building graph...
python graph_builder.py

echo Step 3: Extracting features...
python feature_extractor.py

echo Step 4: Training ML model...
python train_ml.py

echo Step 5: Evaluating model...
python evaluate.py

echo Pipeline completed!
pause