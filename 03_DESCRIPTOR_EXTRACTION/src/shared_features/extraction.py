
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
_repository_sys.path.insert(0, str(_repo_path('03_DESCRIPTOR_EXTRACTION/src')))
_repository_sys.path.insert(0, str(_repo_path('05_CRISP_X/src')))
_repository_sys.path.insert(0, str(_repo_path('03_DESCRIPTOR_EXTRACTION/src/shared_features')))
from pathlib import Path
import numpy as np
import pandas as pd

ATOMIC = ['period', 'electronegativity', 'covalent_radius', 'valence_electrons',
          'ionization_energy', 'electron_affinity', 'atomic_volume']
STRUCTURAL = ['bond_length', 'bond_angle']
FEATURES = ATOMIC + STRUCTURAL
SOURCE_COLUMNS = ['Period', 'Electronegativity', 'Covalent Radius', 'Valence Electrons',
                  'First Ionization Energy', 'Electron Affinity', 'Atomic Volume']


class ConceptExtractor:
    """CGCNN table values preserved; elemental median imputation is label-independent.

    Geometry: each center's periodic neighbors within 1.20 times its nearest
    distance. Include all equal-distance images. Average length over neighbors,
    angle over unordered neighbor pairs, then equally over eligible centers.
    These are operational first-shell geometric descriptors, not certified bonds
    or specifically B-X-B angles. Unknown elemental units are not invented.
    """
    def __init__(self, table, shell_ratio=1.20):
        self.table_path = str(table)
        self.frame = pd.read_csv(table).set_index('Symbol')
        if not self.frame.index.is_unique:
            raise ValueError('Duplicate elements in atomic table')
        self.numeric = self.frame[SOURCE_COLUMNS].astype(float)
        self.medians = self.numeric.median().to_numpy()
        if not np.isfinite(self.medians).all():
            raise ValueError('Empty atomic property column')
        self.shell_ratio = float(shell_ratio)

    def extract(self, structure):
        if not structure.is_ordered:
            raise ValueError('Disordered structures require a separate site policy')
        if len(structure) == 0 or structure.volume <= 0:
            raise ValueError('Invalid structure')
        symbols = [s.specie.symbol for s in structure]
        unknown = sorted(set(symbols) - set(self.frame.index))
        if unknown:
            raise ValueError('Elements absent from supplied table: ' + ','.join(unknown))
        raw = self.numeric.loc[symbols].to_numpy()
        coverage = np.isfinite(raw).mean(axis=0)
        atomic = np.where(np.isfinite(raw), raw, self.medians).mean(axis=0)
        # Expand the search only if a center lacks a complete first shell.
        cutoff = 4.0
        while True:
            ci, pj, offset, dist = structure.get_neighbor_list(cutoff, exclude_self=True)
            near = np.full(len(structure), np.inf)
            np.minimum.at(near, ci, dist)
            if np.isfinite(near).all() and (near * self.shell_ratio < cutoff - 1e-7).all():
                break
            cutoff *= 1.6
            if cutoff > 40:
                raise ValueError('First shell not found below 40 angstrom')
        lengths, angles, counts = [], [], []
        for center in range(len(structure)):
            idx = np.flatnonzero((ci == center) & (dist > 1e-7) &
                                 (dist <= near[center] * self.shell_ratio + 1e-7))
            if len(idx) == 0:
                raise ValueError('No positive-distance first neighbors')
            vectors = (structure.frac_coords[pj[idx]] + offset[idx] - structure.frac_coords[center]) @ structure.lattice.matrix
            unit = vectors / np.linalg.norm(vectors, axis=1)[:, None]
            lengths.append(float(dist[idx].mean()))
            counts.append(len(idx))
            if len(idx) >= 2:
                dots = (unit @ unit.T)[np.triu_indices(len(idx), 1)]
                angles.append(float(np.degrees(np.arccos(np.clip(dots, -1, 1))).mean()))
        angle = float(np.mean(angles)) if angles else np.nan
        values = np.r_[atomic, np.mean(lengths), angle]
        quality = {'missing_fraction_' + f: float(1 - coverage[j]) for j, f in enumerate(ATOMIC)}
        quality.update(angle_center_coverage=len(angles) / len(structure),
                       min_shell_distance=float(near.min()),
                       mean_coordination=float(np.mean(counts)), cell_volume=float(structure.volume),
                       nsites=len(structure))
        return values, quality

    def describe(self):
        return {'features': FEATURES, 'atomic': ATOMIC, 'structural': STRUCTURAL,
                'aggregation': 'site-count-weighted elemental table means; equal-site periodic first-shell geometric means',
                'atomic_units': 'unchanged supplied-table coordinates; covalent radius in source pm; other transformations/units require source verification',
                'structural_units': {'bond_length': 'angstrom', 'bond_angle': 'degree'},
                'atomic_volume_definition': 'mean of supplied elemental Atomic Volume column; never crystal V/N',
                'missing': 'fixed per-column median across supplied element table, including unused elements; record per-feature missing fraction',
                'medians': dict(zip(ATOMIC, self.medians.tolist())), 'shell_ratio': self.shell_ratio}
