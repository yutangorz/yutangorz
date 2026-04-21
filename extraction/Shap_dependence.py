import pandas as pd
import numpy as np
import shap
from sklearn.ensemble import RandomForestRegressor
import os
import warnings

warnings.filterwarnings('ignore')

FEATURES_FILE = 'material_features.csv'
OUTPUT_DIR = 'shap_results'

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

print("1. Loading data and training model...")

df = pd.read_csv(FEATURES_FILE)
feature_cols = [
    'period', 'electronegativity', 'covalent_radius', 'valence_electrons',
    'ionization_energy', 'electron_affinity', 'atomic_volume',
    'bond_length', 'bond_angle'
]
target_col = 'predicted_bandgap'

df_clean = df[feature_cols + [target_col]].dropna()
X = df_clean[feature_cols]
y = df_clean[target_col]

print("   - Training Random Forest proxy model...")
model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
model.fit(X, y)

print("   - Calculating SHAP values (this may take a few minutes)...")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

name_map_file = {
    'period': 'Period',
    'electronegativity': 'Electronegativity',
    'covalent_radius': 'Covalent_Radius',
    'valence_electrons': 'Valence_Electrons',
    'ionization_energy': 'Ionization_Energy',
    'electron_affinity': 'Electron_Affinity',
    'atomic_volume': 'Atomic_Volume',
    'bond_length': 'Bond_Length',
    'bond_angle': 'Bond_Angle'
}

print("\n2. Generating dependence data CSVs for 9 features...")

all_data_list = []

mean_abs_shap = np.abs(shap_values).mean(axis=0)
sorted_feat_indices = np.argsort(mean_abs_shap)[::-1]

for i, feat_name in enumerate(feature_cols):
    print(f"   - Processing feature: {feat_name} ({i + 1}/9)")

    x_vals = X[feat_name].values
    shap_vals = shap_values[:, i]

    other_indices = [idx for idx in range(len(feature_cols)) if idx != i]
    remaining_sorted = [idx for idx in sorted_feat_indices if idx != i]
    interaction_feat_idx = remaining_sorted[0]
    interaction_feat_name = feature_cols[interaction_feat_idx]
    interaction_vals = X[interaction_feat_name].values

    df_dep = pd.DataFrame({
        'Feature_Value': x_vals,
        'SHAP_Value': shap_vals,
        f'Color_By_{interaction_feat_name}': interaction_vals,
        'Sample_ID': df_clean.index
    })

    df_dep = df_dep.sort_values(by='Feature_Value').reset_index(drop=True)

    safe_name = name_map_file.get(feat_name, feat_name)
    file_name = f'Dependence_Data_{safe_name}.csv'
    df_dep.to_csv(os.path.join(OUTPUT_DIR, file_name), index=False)

    df_dep['Feature_Name'] = feat_name
    df_dep['Feature_Display_Name'] = safe_name
    all_data_list.append(df_dep)

    print(f"     ✅ Saved: {file_name} (Color by: {interaction_feat_name})")

print("\n3. Generating summary file (All_Features_Dependence_Data.csv)...")

top1_feat_idx = sorted_feat_indices[0]
top1_feat_name = feature_cols[top1_feat_idx]

list_simple = []
for i, feat_name in enumerate(feature_cols):
    x_vals = X[feat_name].values
    shap_vals = shap_values[:, i]
    top1_vals = X[top1_feat_name].values

    df_temp = pd.DataFrame({
        'Feature_Name': feat_name,
        'Feature_Display_Name': name_map_file.get(feat_name, feat_name),
        'Feature_Value': x_vals,
        'SHAP_Value': shap_vals,
        f'Color_Reference_{top1_feat_name}': top1_vals
    })
    list_simple.append(df_temp)

df_final_all = pd.concat(list_simple, ignore_index=True)
df_final_all.to_csv(os.path.join(OUTPUT_DIR, 'ALL_Features_Dependence_Data.csv'), index=False)
print(f"   ✅ Saved summary file: ALL_Features_Dependence_Data.csv")

print("\n🎉 Export complete!")
print(f"📂 Location: {os.path.abspath(OUTPUT_DIR)}")
print("\n💡 Origin Plotting Guide:")
print("   Method A (Individual Plots):")
print("      Open 'Dependence_Data_XXX.csv', set 'Feature_Value' as X and 'SHAP_Value' as Y for scatter plot.")
print("      Set 'Color_By_XXX' column as color map to show interaction effects.")
print("      Suggest adding a LOESS smoothing curve or B-Spline fit line to show trends.")
print("\n   Method B (Panel Plots):")
print("      Open 'ALL_Features_Dependence_Data.csv'.")
print("      Use Origin's 'Grouped by Column' feature, panel by 'Feature_Display_Name'.")
print("      X-axis: Feature_Value, Y-axis: SHAP_Value, Color: Color_Reference_XXX.")