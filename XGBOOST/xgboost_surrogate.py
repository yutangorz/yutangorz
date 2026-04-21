import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import joblib
import os
import warnings

warnings.filterwarnings("ignore")

INPUT_FILE = "material_features.csv"
OUTPUT_CSV = "ablation_study_results.csv"
MODEL_DIR = "ablation_models_saved"

if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

ALL_FEATURES = [
    "period",
    "electronegativity",
    "covalent_radius",
    "valence_electrons",
    "ionization_energy",
    "electron_affinity",
    "atomic_volume",
    "bond_length",
    "bond_angle"
]

TARGET_COL = "predicted_bandgap"

ablation_configs = []

ablation_configs.append({
    "name": "Full_Model",
    "features": ALL_FEATURES
})

for feat in ALL_FEATURES:
    remaining_feats = [f for f in ALL_FEATURES if f != feat]
    ablation_configs.append({
        "name": f"LOO_{feat}",
        "features": remaining_feats
    })

structural_feats = ["bond_length", "bond_angle"]
ablation_configs.append({
    "name": "No_Structural",
    "features": [f for f in ALL_FEATURES if f not in structural_feats]
})

atomic_feats = [f for f in ALL_FEATURES if f not in structural_feats]
ablation_configs.append({
    "name": "No_Atomic",
    "features": [f for f in ALL_FEATURES if f in structural_feats]
})

key_three = ["valence_electrons", "electronegativity", "ionization_energy"]
ablation_configs.append({
    "name": "No_Key_Three",
    "features": [f for f in ALL_FEATURES if f not in key_three]
})

print(f"Defined {len(ablation_configs)} ablation study models:")
for cfg in ablation_configs:
    print(f"   - {cfg['name']}: {len(cfg['features'])} features")

print("\nLoading data...")
if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(f"File not found: {INPUT_FILE}")

df = pd.read_csv(INPUT_FILE)

if TARGET_COL not in df.columns:
    candidates = [c for c in df.columns if 'gap' in c.lower()]
    if candidates:
        TARGET_COL = candidates[0]
        print(f"Auto-detected band gap column: '{TARGET_COL}'")
    else:
        raise ValueError(f"Band gap column not found.")

print("Checking and filling missing values...")
for col in ALL_FEATURES:
    if col in df.columns and df[col].isnull().any():
        df[col].fillna(df[col].mean(), inplace=True)

if df[TARGET_COL].isnull().any():
    df.dropna(subset=[TARGET_COL], inplace=True)
    print("Removed samples with missing labels.")
else:
    print("Data is perfect, no missing values!")

X_full = df[ALL_FEATURES]
y = df[TARGET_COL]

train_indices, test_indices = train_test_split(
    X_full.index, test_size=0.2, random_state=42
)

y_train = y.loc[train_indices]
y_test = y.loc[test_indices]

print(f"Total samples: {len(df)}")
print(f"Training set: {len(train_indices)}, Test set: {len(test_indices)}")

results_data = {
    "Sample_ID": test_indices,
    "True_Bandgap": y_test.values
}

metrics_summary = []

print("\nTraining 13 models...")

for i, config in enumerate(ablation_configs):
    name = config["name"]
    feats = config["features"]

    print(f"[{i + 1}/{len(ablation_configs)}] Training model: {name} ...")

    X_train_curr = df.loc[train_indices, feats]
    X_test_curr = df.loc[test_indices, feats]

    model = xgb.XGBRegressor(
        n_estimators=1000,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        reg_alpha=0.1,
        reg_lambda=1.0,
        objective='reg:squarederror',
        n_jobs=-1,
        random_state=42,
        verbosity=0
    )

    model.fit(X_train_curr, y_train)

    y_pred = model.predict(X_test_curr)

    results_data[f"Pred_{name}"] = y_pred

    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)

    metrics_summary.append({
        "Model_Name": name,
        "Num_Features": len(feats),
        "Features_Used": str(feats),
        "R2": r2,
        "RMSE": rmse,
        "MAE": mae
    })

    joblib.dump(model, os.path.join(MODEL_DIR, f"{name}.pkl"))

    print(f"      -> R²: {r2:.4f}, RMSE: {rmse:.4f}")

print("\nSaving detailed results...")

df_results = pd.DataFrame(results_data)
df_results.to_csv(OUTPUT_CSV, index=False)
print(f"Sample-level prediction results saved to: {OUTPUT_CSV}")

df_metrics = pd.DataFrame(metrics_summary)
metrics_file = "ablation_metrics_summary.csv"
df_metrics.to_csv(metrics_file, index=False)
print(f"Model performance metrics summary saved to: {metrics_file}")

print("\n" + "=" * 80)
print("Ablation Study Performance Summary")
print("=" * 80)
print(df_metrics[['Model_Name', 'Num_Features', 'R2', 'RMSE', 'MAE']].to_string(index=False))
print("=" * 80)

print("\nAll training and saving tasks completed! Please use plot_fig5.py for plotting.")