This repository contains a comprehensive suite of scripts for analyzing material bandgap properties. The pipeline processes crystal structure files, calculates features, trains models, and performs advanced analysis.

🛠️ Prerequisites
Ensure you have the following Python packages installed:

pip install pandas numpy scikit-learn xgboost shap matplotlib seaborn pymatgen tqdm

📝 Usage & Execution Order
Execute the scripts in the following sequence:
1. Data Processing
Description: Generate feature dataset from VASP structure files (POSCAR/CONTCAR)
● Input: Structure files in dataset/ folder, optionally model_predictions.csv and atom_importance.csv
● Output: data/material_features.csv
● Command:

python Data_Processing.py

2. Data Verification & Proxy Importance Analysis
Description: Load generated data, perform sensitivity analysis using XGBoost model, compare with DFT data
● Input: data/material_features.csv
● Output: Verification data and comparison plots in results/ folder
● Command:


python Proxy_Importance.py

3. SHAP Analysis (Visualization Preparation)
Description: Train Random Forest proxy model and generate data for Origin plotting
● Input: material_features.csv (in root directory or specified path)
● Output: CSV files for dependence plots in shap_results/
● Command:

python Shap_dependence.py

4. SHAP Analysis (Plot Generation)
Description: Generate final visualizations (Beeswarm, Bar, Heatmap)
● Input: material_features.csv
● Output: PNG images in shap_results/
● Command:

python shap_analysis_alignn.py

📊 Analysis Workflow
1. Feature Generation - Extract structural and atomic features from crystal structures
2. Model Training - Train proxy models for bandgap prediction
3. Feature Importance - Analyze feature importance using various methods
4. SHAP Analysis - Perform SHAP value analysis for model interpretability
5. Sensitivity Verification - Validate results against DFT calculations
🛠️ Configuration
Key configuration settings are managed in Config_1_Settings.py. Modify the following if your file locations differ:
● BASE_DIR: Root directory of the project
● DATA_DIR: Directory for input/output data
● File path constants for all input/output files
● Feature lists and target variables
📌 Notes
● The scripts contain hardcoded relative paths based on the recommended directory structure
● Ensure all required input files are in the correct locations before execution
● The pipeline is designed for materials science applications, specifically bandgap property analysis
● Results are saved in designated output directories for easy access and organization

