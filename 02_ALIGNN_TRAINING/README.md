# ALIGNN training

ALIGNN training configuration, trained weights, training/validation loss histories, test-set predictions, structure conversion and fixed data splits. Optional transfer/finetuning scaffolding is excluded. vendor/ retains the upstream package layout as reference, not as a complete installed ALIGNN.

Main source: [src/training/train.py](src/training/train.py).

Inputs/configuration: configs/training.json; data/splits/ids_train_val_test.json; data/training; data/historical_training contains the training records. Generated scientific products belong under the configured outputs/02_ALIGNN_TRAINING/ scientific protocol directory; no output is produced by opening or reading this README.

Prerequisites/limits: A compatible DGL/PyTorch/ALIGNN environment is required. Scalar MAE/L1 and constant effective AdamW LR 5e-4 are supported by historical implementation reconstruction; exact installed version and stepwise LR logs remain unavailable. See [training implementation](TRAINING_IMPLEMENTATION.md).

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

Author-owned material is MIT-licensed; third-party terms remain applicable. See the root [licence and source notes](../DATA_AND_LICENSE_AUDIT.md).
