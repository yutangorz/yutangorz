import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from Config_1_Settings import CLEAN_FEATURES_CSV, ALL_FEATURES, TARGET_ATOMIC_FEATURES, TARGET_STRUCT_FEATURES


def load_clean_data():
    if not isinstance(CLEAN_FEATURES_CSV, str) or not __import__('os').path.exists(CLEAN_FEATURES_CSV):
        raise FileNotFoundError(f"❌ Data file not found: {CLEAN_FEATURES_CSV}")

    df = pd.read_csv(CLEAN_FEATURES_CSV)

    missing = [col for col in ALL_FEATURES + ['predicted_bandgap'] if col not in df.columns]
    if missing:
        raise ValueError(f"❌ Data missing columns: {missing}")

    initial_len = len(df)
    df = df.dropna(subset=ALL_FEATURES + ['predicted_bandgap'])
    final_len = len(df)
    print(f"📊 Data Loading: Removed {initial_len - final_len} rows with missing values, {final_len} samples remaining.")

    return df


def prepare_xy(df):
    X = df[ALL_FEATURES].values
    y = df['predicted_bandgap'].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X, X_scaled, y, scaler


class DataLoader:
    def __init__(self, config):
        self.config = config
        self.feature_stats = {}

    def load_feature_data(self, file_path):
        df = pd.read_csv(file_path)

        atomic_features = [
            'valence_electrons', 'covalent_radius', 'atomic_volume',
            'electron_affinity', 'electronegativity', 'ionization_energy', 'period'
        ]

        print("📊 Calculating feature statistics to eliminate dimensional effects...")
        for feat in atomic_features:
            if feat in df.columns:
                mean_val = df[feat].mean()
                std_val = df[feat].std()
                self.feature_stats[feat] = {'mean': mean_val, 'std': std_val}
                print(f"   {feat}: Mean={mean_val:.4f}, Std={std_val:.4f}")

        return df

    def get_feature_scaler(self, feature_name):
        if feature_name in self.feature_stats:
            std = self.feature_stats[feature_name]['std']
            return std if std > 1e-6 else 1.0
        return 1.0

if __name__ == "__main__":
    try:
        df = load_clean_data()
        print("✅ Data loading successful!")
        print(df.head())

        X, X_scaled, y, _ = prepare_xy(df)
        print(f"\nFeature matrix shape: {X.shape}")
        print(f"Label shape: {y.shape}")
        print(f"Bandgap range: {y.min():.2f} - {y.max():.2f} eV")
    except Exception as e:
        print(f"❌ Error: {e}")