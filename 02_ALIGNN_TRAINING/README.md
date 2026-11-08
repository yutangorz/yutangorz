# ALIGNN training

Actual recorded network training, structure conversion and fixed internal splits. Optional transfer/finetuning scaffolding is excluded. vendor/ retains the upstream package layout as reference, not as a complete installed ALIGNN.

Main source: [src/training/train.py](src/training/train.py).

Inputs/configuration: configs/training.json; data/splits/ids_train_val_test.json; data/training. Generated scientific products belong under the configured outputs/02_ALIGNN_TRAINING/ scientific protocol directory; no output is produced by opening or reading this README.

Prerequisites/limits: DGL is absent in the review environment. Actual loss/scheduler execution identity is partly unresolved; no training was rerun.

To check the repository without executing science, run `python 09_REPRODUCTION_CHECKS/verify_repository.py` and `python 09_REPRODUCTION_CHECKS/verify_saved_metrics.py` from the repository root. Read the named source and adjacent configuration before starting its scientific calculation. No scientific command is described as already executed here. CLI-required workspaces/configuration or licensed upstream outputs must be supplied explicitly. Original source-hash/authorization guards may reject modified or incomplete inputs and are not weakened.

Source files:

- [src/training/alignn_atomwise.py](src/training/alignn_atomwise.py)
- [src/training/check_conversion_targets.py](src/training/check_conversion_targets.py)
- [src/training/check_training_inputs.py](src/training/check_training_inputs.py)
- [src/training/convert_structures.py](src/training/convert_structures.py)
- [src/training/data.py](src/training/data.py)
- [src/training/split_dataset.py](src/training/split_dataset.py)
- [src/training/train.py](src/training/train.py)
- [vendor/alignn/graphs.py](vendor/alignn/graphs.py)
- [vendor/alignn/models/alignn_atomwise.py](vendor/alignn/models/alignn_atomwise.py)
- [vendor/alignn/models/utils.py](vendor/alignn/models/utils.py)
