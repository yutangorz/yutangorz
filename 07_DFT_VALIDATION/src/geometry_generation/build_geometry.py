"""Construct a bounded 1230247 structural sensitivity set from the frozen A cell.

This is a new fixed-geometry branch. It does not reuse the legacy 7750830
N--N or Yb--N--Yb generator and does not run VASP.
"""

# Portable repository locations and local scientific packages.
from pathlib import Path as _RepositoryPath
import sys as _repository_sys
_repository_root = next((p for p in _RepositoryPath(__file__).resolve().parents if (p / "repository_paths.py").is_file()), None)
if _repository_root is None:
    raise RuntimeError("Repository root marker is missing")
_repository_sys.path.insert(0, str(_repository_root))
from repository_paths import repo_path as _repo_path
_repository_sys.path.insert(0, str(_repo_path('07_DFT_VALIDATION/src/substitution_qualification')))
_repository_sys.path.insert(0, str(_repo_path('07_DFT_VALIDATION/src/soc_qualification')))
_repository_sys.path.insert(0, str(_repo_path('07_DFT_VALIDATION/src/input_guards')))
_repository_sys.path.insert(0, str(_repo_path('07_DFT_VALIDATION/src/validation')))
_repository_sys.path.insert(0, str(_repo_path('03_DESCRIPTOR_EXTRACTION/src/shared_features')))
_repository_sys.path.insert(0, str(_repo_path('03_DESCRIPTOR_EXTRACTION/src')))
_repository_sys.path.insert(0, str(_repo_path('05_CRISP_X/src')))
_repository_sys.path.insert(0, str(_repo_path('07_DFT_VALIDATION/src/geometry_generation')))
import csv
import hashlib
from pathlib import Path

import numpy as np
from pymatgen.core import Structure
from scipy.optimize import brentq

ROOT = _repo_path('.')
HERE = _repo_path('07_DFT_VALIDATION/workflows/geometry_generation')
SOURCE = _repo_path('07_DFT_VALIDATION/inputs/1230247/A/POSCAR')
OUT = _repo_path('07_DFT_VALIDATION/workflows/geometry_generation/frames')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def angle(w, o, ca):
    a, b = w - o, ca - o
    return float(np.degrees(np.arccos(np.clip(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)), -1, 1))))


def main():
    assert not OUT.exists(), "Frozen output already exists"
    s = Structure.from_file(SOURCE)
    assert len(s) == 20 and s.composition.reduced_formula == "Ba2CaWO6"
    assert [str(s[i].specie) for i in (4, 6, 8)] == ["Ca", "W", "O"]
    w = s[6].coords
    _, oi = s.lattice.get_distance_and_image(s[6].frac_coords, s[8].frac_coords)
    o = s[8].coords + np.asarray(oi) @ s.lattice.matrix
    _, ci = s.lattice.get_distance_and_image(s[8].frac_coords, s[4].frac_coords)
    ca = s[4].coords + (np.asarray(oi) + np.asarray(ci)) @ s.lattice.matrix
    radius = np.linalg.norm(o - w)
    radial = (o - w) / radius
    toward = ca - w - np.dot(ca - w, radial) * radial
    tangent = toward / np.linalg.norm(toward)
    base_angle = angle(w, o, ca)
    base_ca_o = np.linalg.norm(ca - o)
    OUT.mkdir(parents=True)
    rows = []

    def save(kind, level, requested, moved):
        new = Structure(s.lattice, s.species, s.frac_coords, coords_are_cartesian=False)
        new.replace(8, "O", coords=new.lattice.get_fractional_coords(moved) % 1, coords_are_cartesian=False)
        assert all(new[i].specie == s[i].specie for i in range(20))
        assert all(np.allclose(new[i].frac_coords, s[i].frac_coords, atol=1e-12) for i in range(20) if i != 8)
        assert np.allclose(new.lattice.matrix, s.lattice.matrix, atol=1e-12)
        mind = min(new.get_distance(i, j) for i in range(20) for j in range(i + 1, 20))
        displacement = float(np.linalg.norm(moved - o))
        assert mind > 1.40 and displacement < 0.16, (kind, level, mind, displacement)
        d_wo = float(np.linalg.norm(moved - w))
        a = angle(w, moved, ca)
        if kind == "LENGTH":
            assert abs((d_wo - radius) - requested) < 1e-9
        else:
            assert abs((a - base_angle) - requested) < 1e-7
            assert abs(d_wo - radius) < 1e-9
        name = f"1230247_{kind}_{level:02d}"
        folder = OUT / name
        folder.mkdir()
        new.to(fmt="poscar", filename=str(folder / "POSCAR"))
        rows.append(dict(task_id=name, kind=kind, level=level, requested_delta_angstrom=requested if kind == "LENGTH" else "", requested_delta_degree=requested if kind == "ANGLE" else "", W_site=6, O_site=8, Ca_site=4, baseline_W_O_angstrom=radius, actual_W_O_angstrom=d_wo, baseline_W_O_Ca_degree=base_angle, actual_W_O_Ca_degree=a, baseline_Ca_O_angstrom=base_ca_o, actual_Ca_O_angstrom=float(np.linalg.norm(ca - moved)), oxygen_displacement_angstrom=displacement, min_pair_distance_angstrom=mind, source_POSCAR_sha256=sha(SOURCE), frame_POSCAR_sha256=sha(folder / "POSCAR"), protocol="fixed20_PBE_NM_nonSOC_1230247_A_12206", status="GEOMETRY_ONLY_NOT_SUBMITTED"))

    for n, delta in enumerate([-0.018, -0.015, -0.012, -0.009, -0.006, -0.003, 0.003, 0.006, 0.009, 0.012, 0.015, 0.018], 1):
        save("LENGTH", n, delta, w + (radius + delta) * radial)
    for n, desired in enumerate([-1.0, -0.8, -0.6, -0.4, -0.2, 0.2, 0.4, 0.6, 0.8, 1.0], 1):
        f = lambda phi: angle(w, w + radius * (np.cos(phi) * radial + np.sin(phi) * tangent), ca) - base_angle - desired
        phi = brentq(f, -0.12, 0.12)
        save("ANGLE", n, desired, w + radius * (np.cos(phi) * radial + np.sin(phi) * tangent))
    assert len(rows) == 22 and len({r['frame_POSCAR_sha256'] for r in rows}) == 22
    with (_repo_path('07_DFT_VALIDATION/workflows/geometry_generation/GEOMETRY_MATRIX.csv')).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("frames", len(rows), "reference", sha(SOURCE), "W-O", round(radius, 6), "W-O-Ca", round(base_angle, 6))


if __name__ == "__main__":
    main()
