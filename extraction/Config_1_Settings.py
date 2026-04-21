import os
from typing import List, Dict

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "results")
PLOT_DATA_DIR = os.path.join(OUTPUT_DIR, "plot_data")

for d in [DATA_DIR, OUTPUT_DIR, PLOT_DATA_DIR]:
    os.makedirs(d, exist_ok=True)

CLEAN_FEATURES_CSV = os.path.join(DATA_DIR, "material_features.csv")
MODEL_PATH = os.path.join(BASE_DIR, "model", "best_model.pt")

TARGET_ATOMIC_FEATURES = [
    "period", "electronegativity", "covalent_radius", "valence_electrons",
    "ionization_energy", "electron_affinity", "atomic_volume"
]

TARGET_STRUCT_FEATURES = [
    "bond_length", "bond_angle"
]

ALL_FEATURES = TARGET_ATOMIC_FEATURES + TARGET_STRUCT_FEATURES

PLOT_CONFIG = {
    "font": "Arial",
    "dpi": 600,
    "formats": ["png", "pdf"],
    "palette": {
        "primary": "#2E86AB",
        "secondary": "#A23B72",
        "success": "#06A77D",
        "warning": "#F18F01",
        "gray": "#C7C7C7"
    }
}

if __name__ == "__main__":
    print(">>> Configuration Check:")
    print(f"Data File: {CLEAN_FEATURES_CSV} (Exists: {os.path.exists(CLEAN_FEATURES_CSV)})")
    print(f"Feature List: {ALL_FEATURES}")