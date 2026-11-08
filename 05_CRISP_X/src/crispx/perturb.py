
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
_repository_sys.path.insert(0, str(_repo_path('05_CRISP_X/src/crispx')))
import hashlib
import numpy as np
from pymatgen.core import Structure, Lattice
from shared_features.extraction import SOURCE_COLUMNS


def structure_key(s):
    payload = str([x.specie.symbol for x in s]) + str(np.round(s.lattice.matrix, 7).tolist()) + str(np.round(s.frac_coords % 1, 7).tolist())
    return hashlib.sha256(payload.encode()).hexdigest()


def generate_candidates(structure, extractor, seed, size=80, geometry_scale=1.):
    """Bounded geometry and whole-species substitutions; not relaxed/DFT stable.

    Reject close contacts, large volume/shortest-contact changes. Chemical
    candidates share block, differ by at most one valence electron and have
    similar tabulated radius. This permits probing valence instead of fixing it.
    No claim of charge neutrality, energy stability or synthesis feasibility.
    """
    rng = np.random.default_rng(seed)
    base, base_q = extractor.extract(structure)
    known = {structure_key(structure)}
    candidates, rows, rejected = [], [], []
    source_symbols = sorted(set(s.specie.symbol for s in structure))
    substitutions = []
    for old in source_symbols:
        src = extractor.frame.loc[old]
        for new, dst in extractor.frame.iterrows():
            if new == old or new in source_symbols or new in {'He','Ne','Ar','Kr','Xe','Rn','Og'} or not np.isfinite(extractor.numeric.loc[new].to_numpy()).all():
                continue
            if src.Block == dst.Block and abs(src['Valence Electrons'] - dst['Valence Electrons']) <= 1:
                ratio = dst['Covalent Radius'] / src['Covalent Radius']
                if .75 <= ratio <= 1.25 and abs(src.Period - dst.Period) <= 1:
                    substitutions.append((old, new))
    for attempt in range(size * 12):
        if len(candidates) >= size:
            break
        if substitutions and attempt % 2:
            old, new = substitutions[int(rng.integers(len(substitutions)))]
            species = [new if x.specie.symbol == old else x.specie.symbol for x in structure]
            s = Structure(structure.lattice, species, structure.frac_coords)
            # Apply a small geometric change as well; all resulting concepts are recomputed.
            strain = rng.uniform(-.025, .025) * geometry_scale
            s.scale_lattice(s.volume * (1 + strain) ** 3)
            # Within-chemical-path independent geometry variation, not only dilation.
            shear = rng.normal(0, .008 * geometry_scale, (3,3))
            shear = (shear + shear.T) / 2
            lattice = Lattice(s.lattice.matrix @ (np.eye(3) + shear))
            shift = rng.normal(0, .008 * geometry_scale, (len(s),3))
            shift -= shift.mean(axis=0)
            s = Structure(lattice, species, (s.frac_coords + shift @ np.linalg.inv(lattice.matrix)) % 1)
            event = dict(kind='bounded_same_block_species_substitution', old=old, new=new,
                         valence_delta=float(extractor.frame.loc[new,'Valence Electrons']-extractor.frame.loc[old,'Valence Electrons']),
                         strain=float(strain))
        else:
            strain = rng.normal(0, .016 * geometry_scale, (3, 3))
            strain = (strain + strain.T) / 2
            strain = np.clip(strain, -.035, .035)
            lattice = Lattice(structure.lattice.matrix @ (np.eye(3) + strain))
            # Zero-mean atomic displacement avoids a redundant translation.
            shift = rng.normal(0, .018 * geometry_scale, (len(structure), 3))
            shift -= shift.mean(axis=0)
            s = Structure(lattice, [x.specie.symbol for x in structure],
                          (structure.frac_coords + shift @ np.linalg.inv(lattice.matrix)) % 1)
            event = dict(kind='strain_and_internal_displacement', strain=strain.tolist(), displacement_A=shift.tolist())
        try:
            x, q = extractor.extract(s)
            if not np.isfinite(x).all():
                raise ValueError('Undefined concept')
            if q['min_shell_distance'] < max(.65, .75 * base_q['min_shell_distance']):
                raise ValueError('Short contact constraint')
            if not .85 < s.volume / structure.volume < 1.15:
                raise ValueError('Volume bound')
            key = structure_key(s)
            if key in known:
                continue
            known.add(key)
            event.update(candidate=len(candidates), structure_hash=key,
                         min_contact_A=q['min_shell_distance'], n_missing=sum(q['missing_fraction_' + f] > 0 for f in extractor.describe()['atomic']))
            candidates.append(s)
            rows.append((x, event))
        except ValueError as e:
            rejected.append(str(e))
    if len(candidates) < min(size, 32):
        raise ValueError('Too few valid candidates: ' + str(len(candidates)))
    return candidates, np.array([r[0] for r in rows]), [r[1] for r in rows], rejected
