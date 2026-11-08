"""Check labels by material ID, without loading any structure files."""

# Portable repository locations and local scientific packages.
from pathlib import Path as _RepositoryPath
import sys as _repository_sys
_repository_root = next((p for p in _RepositoryPath(__file__).resolve().parents if (p / "repository_paths.py").is_file()), None)
if _repository_root is None:
    raise RuntimeError("Repository root marker is missing")
_repository_sys.path.insert(0, str(_repository_root))
from repository_paths import repo_path as _repo_path
pass  # Optional local dependency directory is not included.
pass  # Optional local dependency directory is not included.
pass  # Optional local dependency directory is not included.
pass  # Optional local dependency directory is not included.
_repository_sys.path.insert(0, str(_repo_path('03_DESCRIPTOR_EXTRACTION/src/shared_features')))
_repository_sys.path.insert(0, str(_repo_path('03_DESCRIPTOR_EXTRACTION/src')))
_repository_sys.path.insert(0, str(_repo_path('05_CRISP_X/src')))
_repository_sys.path.insert(0, str(_repo_path('02_ALIGNN_TRAINING/src/training')))
import argparse
import csv
import json
from pathlib import Path
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument('--source', type=Path, required=True)
ap.add_argument('--reference', type=Path, required=True)
ap.add_argument('--out', type=Path, required=True)
a = ap.parse_args()
ref = pd.read_csv(a.reference)
truth = dict(zip(ref.structure_id.astype(str), ref.band_gap))
report = {}
for part in ['dataset', 'vasp_train', 'vasp_test', 'train', 'test']:
    path = a.source / part / 'id_prop.csv'
    rows = list(csv.reader(path.open(encoding='utf-8-sig')))
    checked, bad = [], []
    for row in rows:
        try:
            mid, value = Path(row[0]).stem, float(row[1])
        except (ValueError, IndexError):
            continue
        if mid in truth:
            error = abs(value - truth[mid])
            checked.append(error)
            if error > 1e-5:
                bad.append({'ID': mid, 'csv_label': value, 'reference': truth[mid]})
    report[part] = dict(checked_n=len(checked), mismatch_n=len(bad),
                        max_abs_error_eV=max(checked) if checked else None,
                        examples=bad[:5])
a.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(report, ensure_ascii=False, indent=2))
