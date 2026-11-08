"""Audit historical artifacts without training, changing weights, or reading structures.
Run with the supplied evidence directory and an empty output directory.
Python dependencies: numpy, pandas, torch (CPU is sufficient).
"""

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
import hashlib
import itertools
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def norm(x):
    return Path(str(x)).stem


def read_ids(p):
    rows = list(csv.reader(p.open(encoding='utf-8-sig')))
    if rows and rows[0][0] == 'material_id':
        rows = rows[1:]
    return [(norm(r[0]), float(r[1])) for r in rows if len(r) >= 2]


def load(p):
    return json.loads(p.read_text(encoding='utf-8-sig'))


def metrics(y, p):
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    assert len(y) == len(p) and len(y) > 0
    e = p - y
    return dict(n=len(y), mae_eV=float(abs(e).mean()),
                rmse_eV=float(np.sqrt((e * e).mean())),
                r2=float(1 - (e * e).sum() / ((y - y.mean()) ** 2).sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--evidence', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    src, out = args.evidence, args.out
    out.mkdir(parents=True, exist_ok=False)
    temp = src / 'temp'
    config = load(temp / 'config.json')
    splits = {k: [norm(x) for x in v] for k, v in load(temp / 'ids_train_val_test.json').items()}
    full = pd.read_csv(src / '2step2.csv')
    full['ID'] = full['structure_id'].astype(str).map(norm)
    assert len(full) == 7176 and full.ID.is_unique
    truth = dict(zip(full.ID, full.band_gap))
    formula = dict(zip(full.ID, full['name']))
    ordered = read_ids(src / 'vasp_train_id_prop.csv')
    external = read_ids(src / 'vasp_test_id_prop.csv')
    original = read_ids(src / 'dataset_id_prop.csv')
    n = len(ordered)
    nt, nv, ne = (int(n * config[k]) for k in ('train_ratio', 'val_ratio', 'test_ratio'))
    reconstructed = dict(id_train=[i for i, _ in ordered[:nt]],
                         id_val=[i for i, _ in ordered[-(nv + ne):-ne]],
                         id_test=[i for i, _ in ordered[-ne:]])
    exact = {k: splits[k] == reconstructed[k] for k in splits}
    assert all(exact.values())
    allocated = set(itertools.chain.from_iterable(splits.values()))
    outer = {i for i, _ in external}
    unused = {i for i, _ in ordered} - allocated
    assert not (outer & allocated) and not (outer & unused)
    assert allocated | outer | unused == set(truth)
    labels = {i: y for i, y in original}
    max_label_error = max(abs(truth[i] - labels[i]) for i in labels)
    indices = list(range(len(original)))
    random.Random(42).shuffle(indices)
    replay_outer = {original[j][0] for j in indices[:int(.1 * len(original))]}
    report = {'scope': 'historical result audit; no new training or forward inference',
              'config': config,
              'counts': {k: len(v) for k, v in splits.items()},
              'unique_counts': {k: len(set(v)) for k, v in splits.items()},
              'ordered_input_n': n, 'exact_ordered_split_reconstruction': exact,
              'external_holdout_n': len(outer), 'rounding_gap_ids': sorted(unused),
              'seed42_outer_membership_replay_matches': replay_outer == outer,
              'all_7176_accounted_for': True, 'input_label_max_abs_difference_eV': max_label_error,
              'id_overlap': {}, 'formula_overlap_exact_string': {}}
    for a, b in itertools.combinations(splits, 2):
        report['id_overlap'][a + '__' + b] = len(set(splits[a]) & set(splits[b]))
        report['formula_overlap_exact_string'][a + '__' + b] = len({formula[i] for i in splits[a]} & {formula[i] for i in splits[b]})
    status = {i: k for k, v in splits.items() for i in v}
    status.update({i: 'external_717' for i in outer})
    status.update({i: 'unallocated_floor_rounding' for i in unused})
    full[['ID', 'name', 'band_gap']].assign(split=full.ID.map(status)).to_csv(out / 'membership_7176.csv', index=False)
    report['results'] = {}
    for kind in ('Train', 'Val', 'Test'):
        rows = load(temp / (kind + '_results.json'))
        yy, pp = [], []
        batch_mae_sum, batch_mse_sum = 0., 0.
        for r in rows:
            y = np.asarray(r['target_out'], dtype=float).reshape(-1)
            p = np.asarray(r['pred_out'], dtype=float).reshape(-1)
            assert y.shape == p.shape
            yy.extend(y.tolist()); pp.extend(p.tolist())
            batch_mae_sum += abs(y - p).mean()
            batch_mse_sum += ((y - p) ** 2).mean()
        res = metrics(yy, pp)
        res.update(records=len(rows), sum_batch_mae=float(batch_mae_sum), sum_batch_mse=float(batch_mse_sum))
        if kind != 'Test':
            hist = np.asarray(load(temp / ('history_' + kind.lower() + '.json')))
            res.update(history_epochs=len(hist), history_best_epoch_1based=int(hist[:, 0].argmin()) + 1,
                       history_final=float(hist[-1, 0]),
                       mae_history_abs_difference=abs(float(hist[-1, 0]) - batch_mae_sum),
                       mse_history_abs_difference=abs(float(hist[-1, 0]) - batch_mse_sum))
            res['id_caution'] = 'Train has no per-sample IDs; Val stores one misleading ID per batch. Do not treat batch IDs as sample IDs.'
        else:
            ids = [norm(r['id']) for r in rows]
            assert ids == splits['id_test'] and len(set(ids)) == len(ids)
            res['target_max_abs_error_vs_7176'] = max(abs(truth[i] - y) for i, y in zip(ids, yy))
            pd.DataFrame(dict(ID=ids, target_eV=yy, prediction_eV=pp)).to_csv(out / 'test_predictions_1291.csv', index=False)
        report['results'][kind] = res
    states = {k: torch.load(temp / (k + '_model.pt'), map_location='cpu', weights_only=True)
              for k in ('best', 'last', 'current')}
    report['checkpoint_tensor_equal_to_best'] = {
        k: states[k].keys() == states['best'].keys() and all(torch.equal(states[k][j], states['best'][j]) for j in states['best'])
        for k in states}
    report['best_tensor_keys'] = len(states['best'])
    report['best_matches_project_file_sha256'] = sha(temp / 'best_model.pt') == sha(src / 'project_best_model.pt')
    if (src / 'bandgap_predictions.csv').exists():
        ext = pd.read_csv(src / 'bandgap_predictions.csv')
        ids = ext.material_id.astype(str).map(norm)
        report['external_prediction_file'] = {
            'n': len(ext), 'unique_ids': ids.nunique(), 'id_set_equals_outer_holdout': set(ids) == outer,
            'checkpoint_provenance': 'not established by the CSV; no forward replay in this audit'}
        if ids.is_unique and set(ids) <= set(truth):
            report['external_prediction_file']['metrics_against_reference'] = metrics([truth[i] for i in ids], ext.bandgap)
    report['source_runtime_caveats'] = [
        'Local vendor train.py is candidate source, not proven executed historical source.',
        'Candidate alignn_ branch overrides criterion with L1Loss, consistent with saved histories.',
        'Candidate creates scheduler before replacing optimizer and steps scheduler once per epoch; actual historical LR trajectory is unverified.',
        'Config warmup_steps and early-stop patience alone do not prove implementation or activation.',
        'No evidence establishes a systematic hyperparameter search or global optimality.']
    (out / 'audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, default=int), encoding='utf-8')
    manifest = {str(p.relative_to(src)): {'size': p.stat().st_size, 'sha256': sha(p)}
                for p in sorted(src.rglob('*')) if p.is_file()}
    (out / 'input_hashes.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: report[k] for k in ('counts', 'external_holdout_n', 'rounding_gap_ids', 'seed42_outer_membership_replay_matches', 'results', 'external_prediction_file')}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
