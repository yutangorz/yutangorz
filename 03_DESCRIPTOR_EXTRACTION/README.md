# Nine-descriptor representation

Seven stoichiometrically averaged elemental properties plus material-level mean bond length L and mean bond angle Θ. Atomic Volume is an elemental-table property. The periodic first shell uses 1.20 times nearest nonzero distance, followed by equal-centre averaging.

Main source: [src/shared_features/extraction.py](src/shared_features/extraction.py).

Inputs/configuration: data/atom_importance.csv; data/model_input_7176.csv; data/outer_fold_manifest.csv. Generated scientific products belong under the configured outputs/03_DESCRIPTOR_EXTRACTION/ scientific protocol directory; no output is produced by opening or reading this README.

Prerequisites/limits: Source crystal files must be obtained from the identified data sources; the frozen descriptor matrix supports model/statistical checks without re-extraction.

To check the repository without executing science, run `python 09_REPRODUCTION_CHECKS/verify_repository.py` and `python 09_REPRODUCTION_CHECKS/verify_saved_metrics.py` from the repository root. Read the named source and adjacent configuration before starting its scientific calculation. No scientific command is described as already executed here. CLI-required workspaces/configuration or licensed upstream outputs must be supplied explicitly. Original source-hash/authorization guards may reject modified or incomplete inputs and are not weakened.

Source files:

- [src/shared_features/__init__.py](src/shared_features/__init__.py)
- [src/shared_features/extraction.py](src/shared_features/extraction.py)
