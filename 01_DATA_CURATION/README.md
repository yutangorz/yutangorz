# OQMD data selection

Positive band gaps, candidate classification, descriptor completeness, formation-energy/convex-hull checks and composition–space-group redundancy selection. The 61,912 initial-analysis branch is independent of the final 7,176-material branch.

Main source: [src/selection/select_materials.py](src/selection/select_materials.py).

Inputs/configuration: data/raw (packed OQMD source), data/checkpoints, data/final; configs/selection.json. Generated scientific products belong under the configured outputs/01_DATA_CURATION/ scientific protocol directory; no output is produced by opening or reading this README.

Prerequisites/limits: PVK classifier model redistribution rights are unresolved; its weight file is withheld. The independent initial-analysis historical entrypoint is incomplete.

To check the repository without executing science, run `python 09_REPRODUCTION_CHECKS/verify_repository.py` and `python 09_REPRODUCTION_CHECKS/verify_saved_metrics.py` from the repository root. Read the named source and adjacent configuration before starting its scientific calculation. No scientific command is described as already executed here. CLI-required workspaces/configuration or licensed upstream outputs must be supplied explicitly. Original source-hash/authorization guards may reject modified or incomplete inputs and are not weakened.

Source files:

- [src/selection/El2.0.py](src/selection/El2.0.py)
- [src/selection/F_xrd_2.0.py](src/selection/F_xrd_2.0.py)
- [src/selection/XRD_all.py](src/selection/XRD_all.py)
- [src/selection/change-cif-conventional.py](src/selection/change-cif-conventional.py)
- [src/selection/classify_candidates.py](src/selection/classify_candidates.py)
- [src/selection/dele_band0.py](src/selection/dele_band0.py)
- [src/selection/select_materials.py](src/selection/select_materials.py)
- [src/selection/step1_St_Bear_3.0.py](src/selection/step1_St_Bear_3.0.py)
- [src/selection/step1_stable.py](src/selection/step1_stable.py)
- [src/selection/step2.py](src/selection/step2.py)
- [src/selection/step2_redundant.py](src/selection/step2_redundant.py)
- [src/selection/step3.py](src/selection/step3.py)
