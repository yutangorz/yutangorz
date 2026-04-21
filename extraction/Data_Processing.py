import os
import pandas as pd
import numpy as np
from pymatgen.core import Structure
from pymatgen.core.periodic_table import Element
from tqdm import tqdm
import re

VASP_DIR = "./dataset"
PREDICTION_CSV = "./model_predictions.csv"
ELEMENT_TABLE_CSV = "./atom_importance.csv"
OUTPUT_CSV = "./material_features.csv"

CSV_COLUMN_MAP = {
    "element_symbol": "Symbol",
    "electronegativity": "Electronegativity",
    "covalent_radius": "Covalent Radius",
    "period": "Period",
    "valence_electrons": "Valence Electrons",
    "ionization_energy": "First Ionization Energy",
    "electron_affinity": "Electron Affinity",
    "atomic_volume_element": "Atomic Volume"
}

def load_element_table(csv_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Error: Element table not found: {csv_path}")
    try:
        df = pd.read_csv(csv_path)
        print(f"Read element table columns: {list(df.columns)}")
        required_cols = list(CSV_COLUMN_MAP.values())
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            print(f"Warning: CSV is missing the following columns: {missing}")
            print("Please check if column names match exactly (including spaces and capitalization).")
        symbol_col = CSV_COLUMN_MAP["element_symbol"]
        element_dict = {}
        for _, row in df.iterrows():
            symbol = str(row[symbol_col]).strip()
            symbol = re.match(r"([A-Za-z]+)", symbol).group(1) if re.match(r"([A-Za-z]+)", symbol) else symbol
            props = {}
            for standard_name, csv_col in CSV_COLUMN_MAP.items():
                if standard_name == "element_symbol":
                    continue
                if csv_col in df.columns:
                    val = row[csv_col]
                    try:
                        props[standard_name] = float(val)
                    except (ValueError, TypeError):
                        props[standard_name] = np.nan
                else:
                    props[standard_name] = np.nan
            element_dict[symbol] = props
        print(f"Successfully loaded element table: {len(element_dict)} elements.")
        return element_dict
    except Exception as e:
        print(f"Failed to read element table: {e}")
        return {}

USER_ELEMENT_TABLE = load_element_table(ELEMENT_TABLE_CSV)

def get_element_property(symbol, prop_name):
    clean_symbol = str(symbol).split()[0]
    match = re.match(r"([A-Za-z]+)", clean_symbol)
    if match:
        clean_symbol = match.group(1)
    if clean_symbol in USER_ELEMENT_TABLE:
        val = USER_ELEMENT_TABLE[clean_symbol].get(prop_name, np.nan)
        if not np.isnan(val):
            return val
    try:
        el = Element(clean_symbol)
        if prop_name == "period":
            return el.row
        elif prop_name == "electronegativity":
            return el.X
        elif prop_name == "covalent_radius":
            return el.covalent_radius
        elif prop_name == "valence_electrons":
            return el.num_valence_electrons
        elif prop_name == "ionization_energy":
            return el.ionization_energy
        elif prop_name == "electron_affinity":
            return el.electron_affinity
        elif prop_name == "atomic_volume_element":
            return el.atomic_volume
        else:
            return np.nan
    except Exception:
        return np.nan

def calculate_structure_features(structure: Structure):
    features = {}
    total_atoms = len(structure)
    props_to_calc = ["period", "electronegativity", "covalent_radius", "valence_electrons", "ionization_energy", "electron_affinity"]
    prop_sums = {p: 0.0 for p in props_to_calc}
    valid_atom_count = 0
    for site in structure:
        symbol = site.species_string
        for prop in props_to_calc:
            val = get_element_property(symbol, prop)
            if not np.isnan(val):
                prop_sums[prop] += val
                valid_atom_count += 1
    if valid_atom_count == 0:
        for p in props_to_calc:
            features[f"avg_{p}"] = np.nan
    else:
        for p in props_to_calc:
            features[f"avg_{p}"] = prop_sums[p] / valid_atom_count
    features["atomic_volume"] = structure.volume / total_atoms
    bond_lengths = []
    bond_angles = []
    dist_matrix = structure.distance_matrix
    for i in range(total_atoms):
        neighbors = []
        r_i = get_element_property(structure[i].species_string, "covalent_radius")
        if np.isnan(r_i):
            r_i = 1.5
        for j in range(total_atoms):
            if i == j:
                continue
            d = dist_matrix[i][j]
            r_j = get_element_property(structure[j].species_string, "covalent_radius")
            if np.isnan(r_j):
                r_j = 1.5
            cutoff = (r_i + r_j) * 1.4
            if 0.5 < d < cutoff:
                neighbors.append((j, d))
        neighbors.sort(key=lambda x: x[1])
        neighbors = neighbors[:8]
        for n_idx, dist in neighbors:
            bond_lengths.append(dist)
        if len(neighbors) >= 2:
            center_cart = structure.cart_coords[i]
            for k in range(len(neighbors)):
                for m in range(k + 1, len(neighbors)):
                    n1_idx, _ = neighbors[k]
                    n2_idx, _ = neighbors[m]
                    vec1 = structure.cart_coords[n1_idx] - center_cart
                    vec2 = structure.cart_coords[n2_idx] - center_cart
                    norm1 = np.linalg.norm(vec1)
                    norm2 = np.linalg.norm(vec2)
                    if norm1 * norm2 == 0:
                        continue
                    cos_theta = np.dot(vec1, vec2) / (norm1 * norm2)
                    cos_theta = np.clip(cos_theta, -1.0, 1.0)
                    angle = np.degrees(np.arccos(cos_theta))
                    bond_angles.append(angle)
    features["bond_length"] = np.mean(bond_lengths) if bond_lengths else np.nan
    features["bond_angle"] = np.mean(bond_angles) if bond_angles else np.nan
    return features

def main():
    print(">>> Starting feature dataset construction (based on custom element table & VASP structures)...")
    if not os.path.exists(PREDICTION_CSV):
        print(f"Warning: Prediction file {PREDICTION_CSV} not found, will only calculate features, bandgap column will be left empty.")
        pred_df = pd.DataFrame()
        id_col = None
        gap_col = None
    else:
        pred_df = pd.read_csv(PREDICTION_CSV)
        id_candidates = [c for c in pred_df.columns if 'id' in c.lower() or 'name' in c.lower() or 'file' in c.lower() or 'material' in c.lower()]
        gap_candidates = [c for c in pred_df.columns if 'gap' in c.lower() or 'band' in c.lower() or 'target' in c.lower() or 'pred' in c.lower()]
        if id_candidates and gap_candidates:
            id_col = id_candidates[0]
            gap_col = gap_candidates[0]
            print(f"Identified ID column: '{id_col}', Bandgap column: '{gap_col}'")
        else:
            print("Unable to automatically identify prediction file columns, will skip bandgap merging and only output features.")
            id_col = None
            gap_col = None
    results = []
    if not os.path.exists(VASP_DIR):
        print(f"Error: VASP directory {VASP_DIR} not found")
        return
    files = [f for f in os.listdir(VASP_DIR) if f.startswith("POSCAR") or f.startswith("CONTCAR") or f.endswith(".vasp") or f.endswith(".cif")]
    print(f"Found {len(files)} structure files...")
    for filename in tqdm(files):
        filepath = os.path.join(VASP_DIR, filename)
        try:
            struct = Structure.from_file(filepath)
            mat_id = filename.split('.')[0]
            for prefix in ["POSCAR_", "CONTCAR_"]:
                if mat_id.startswith(prefix):
                    mat_id = mat_id[len(prefix):]
            predicted_gap = np.nan
            if id_col and not pred_df.empty:
                row = pred_df[pred_df[id_col].astype(str).str.contains(mat_id, na=False)]
                if row.empty:
                    row = pred_df[pred_df[id_col].astype(str) == mat_id]
                if not row.empty:
                    predicted_gap = row[gap_col].values[0]
            feats = calculate_structure_features(struct)
            data_row = {
                "material_id": mat_id,
                "formula": struct.formula,
                "predicted_bandgap": predicted_gap,
                "period": feats.get("avg_period", np.nan),
                "electronegativity": feats.get("avg_electronegativity", np.nan),
                "covalent_radius": feats.get("avg_covalent_radius", np.nan),
                "valence_electrons": feats.get("avg_valence_electrons", np.nan),
                "ionization_energy": feats.get("avg_ionization_energy", np.nan),
                "electron_affinity": feats.get("avg_electron_affinity", np.nan),
                "atomic_volume": feats.get("atomic_volume", np.nan),
                "bond_length": feats.get("bond_length", np.nan),
                "bond_angle": feats.get("bond_angle", np.nan)
            }
            results.append(data_row)
        except Exception as e:
            print(f"\nError processing {filename}: {e}")
            continue
    if results:
        out_df = pd.DataFrame(results)
        cols_order = [
            "material_id", "formula", "predicted_bandgap",
            "period", "electronegativity", "covalent_radius",
            "valence_electrons", "ionization_energy", "electron_affinity",
            "atomic_volume", "bond_length", "bond_angle"
        ]
        final_cols = [c for c in cols_order if c in out_df.columns]
        out_df = out_df[final_cols]
        out_df.to_csv(OUTPUT_CSV, index=False)
        print(f"\nSuccess! Generated file: {OUTPUT_CSV}")
        print(f"Valid sample count: {len(out_df)}")
        print("\nFirst 5 rows preview:")
        print(out_df[final_cols].head().to_string())
        print("\nTip: Please check if the 'predicted_bandgap' column has values. If all are NaN, check if the IDs in model_predictions.csv match the filenames.")
    else:
        print("\nNo data generated.")

if __name__ == "__main__":
    main()