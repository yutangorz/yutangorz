import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import warnings
import os

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

FEATURES_FILE = 'material_features.csv'
OUTPUT_DIR = 'shap_results'

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

print("1. Loading and cleaning data...")

try:
    df = pd.read_csv(FEATURES_FILE)
    print(f"File loaded. Original columns: {list(df.columns)}")
except FileNotFoundError:
    raise FileNotFoundError(f"File {FEATURES_FILE} not found. Please check the path.")

target_col = 'predicted_bandgap'
feature_cols = [
    'period', 'electronegativity', 'covalent_radius', 'valence_electrons',
    'ionization_energy', 'electron_affinity', 'atomic_volume',
    'bond_length', 'bond_angle'
]

required_cols = [target_col] + feature_cols
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

df_clean = df[required_cols].copy()
df_clean = df_clean.dropna()
print(f"Data cleaning complete. Valid samples: {len(df_clean)} (Original: {len(df)})")

X = df_clean[feature_cols]
y = df_clean[target_col]

print("\n2. Training proxy model (Random Forest)...")
proxy_model = RandomForestRegressor(n_estimators=200, max_depth=None, random_state=42, n_jobs=-1)
proxy_model.fit(X, y)

y_proxy = proxy_model.predict(X)
r2 = r2_score(y, y_proxy)
mae = mean_absolute_error(y, y_proxy)

print(f"Proxy model performance: R² = {r2:.4f}, MAE = {mae:.4f} eV")
if r2 < 0.85:
    print("Warning: Moderate fit. These 9 features may not fully explain the model's predictions.")
else:
    print("Excellent fit. SHAP analysis results are reliable.")

print("\n3. Calculating SHAP values (this may take 1-3 minutes)...")
explainer = shap.TreeExplainer(proxy_model)
shap_values = explainer.shap_values(X)
print("SHAP values calculated.")

print("\n4. Generating high-quality plots...")

sns.set_context("paper", font_scale=1.2)
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.linewidth'] = 1.2

name_map = {
    'period': 'Period Number',
    'electronegativity': 'Electronegativity',
    'covalent_radius': 'Covalent Radius (Å)',
    'valence_electrons': 'Valence Electrons',
    'ionization_energy': 'Ionization Energy (eV)',
    'electron_affinity': 'Electron Affinity (eV)',
    'atomic_volume': 'Atomic Volume (Å³)',
    'bond_length': 'Avg Bond Length (Å)',
    'bond_angle': 'Avg Bond Angle (°)'
}
display_names = [name_map[c] for c in feature_cols]

print("   - Plotting SHAP Summary (Beeswarm)...")
plt.figure(figsize=(12, 9), dpi=300)
shap.summary_plot(
    shap_values,
    X,
    feature_names=display_names,
    plot_type="dot",
    show=False,
    color=plt.cm.coolwarm,
    alpha=0.6
)
plt.title('SHAP Summary Plot: Feature Impact on Band Gap', fontsize=16, pad=20, fontweight='bold')
plt.savefig(f'{OUTPUT_DIR}/01_shap_summary_beeswarm.png', bbox_inches='tight')
plt.close()
print("   Saved: 01_shap_summary_beeswarm.png")

print("   - Plotting SHAP Importance Bar Chart...")
plt.figure(figsize=(10, 7), dpi=300)
shap.summary_plot(
    shap_values,
    X,
    feature_names=display_names,
    plot_type="bar",
    show=False,
    color='#2c7bb6'
)
plt.title('Mean |SHAP| Value: Feature Importance Ranking', fontsize=16, pad=20, fontweight='bold')
plt.savefig(f'{OUTPUT_DIR}/02_shap_importance_bar.png', bbox_inches='tight')
plt.close()
print("   Saved: 02_shap_importance_bar.png")

print("   - Plotting SHAP Heatmap (Sampling 2000 samples)...")
sample_size = min(2000, len(X))
np.random.seed(42)
sample_idx = np.random.choice(len(X), sample_size, replace=False)
X_sample = X.iloc[sample_idx]
shap_sample = shap_values[sample_idx]

plt.figure(figsize=(14, 10), dpi=300)
shap.heatmap(
    shap_sample,
    X_sample,
    show=False,
    cmap='vlag'
)
plt.title(f'SHAP Heatmap (Sampled {sample_size} Materials)', fontsize=16, pad=20, fontweight='bold')
plt.savefig(f'{OUTPUT_DIR}/03_shap_heatmap.png', bbox_inches='tight')
plt.close()
print("   Saved: 03_shap_heatmap.png")

print("   - Plotting Key Feature Dependence Plots...")
importance_means = np.abs(shap_values).mean(axis=0)
top_2_idx = np.argsort(importance_means)[-2:][::-1]

for idx in top_2_idx:
    feat_key = feature_cols[idx]
    feat_label = display_names[idx]

    plt.figure(figsize=(9, 7), dpi=300)
    shap.dependence_plot(
        idx,
        shap_values,
        X,
        feature_names=display_names,
        show=False,
        dot_size=4,
        alpha=0.5,
        cmap=plt.cm.coolwarm
    )
    ax = plt.gca()
    ax.set_title(f'SHAP Dependence: {feat_label}', fontsize=15, pad=15, fontweight='bold')
    ax.set_xlabel(feat_label, fontsize=13)
    ax.set_ylabel('SHAP Value (Impact on Band Gap / eV)', fontsize=13)
    ax.grid(True, linestyle='--', alpha=0.3)

    plt.savefig(f'{OUTPUT_DIR}/04_dependence_{feat_key}.png', bbox_inches='tight')
    plt.close()
    print(f"   Saved: 04_dependence_{feat_key}.png")

print("\nAll done!")
print(f"Images saved to: {os.path.abspath(OUTPUT_DIR)}")