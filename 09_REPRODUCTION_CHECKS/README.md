# Integrity and saved-result checks

Low-cost integrity, syntax, structured-data, layout and saved-statistic checks. These do not import scientific entrypoints or run models.

Main source: [verify_repository.py](verify_repository.py).

Inputs/configuration: Release manifest and included fixed inputs/statistics. Generated scientific products belong under the configured outputs/09_REPRODUCTION_CHECKS/ scientific protocol directory; no output is produced by opening or reading this README.

Prerequisites/limits: These checks establish file/format/arithmetic consistency only, not complete historical numerical reproduction.

To check the repository without executing science, run `python 09_REPRODUCTION_CHECKS/verify_repository.py` and `python 09_REPRODUCTION_CHECKS/verify_saved_metrics.py` from the repository root. Read the named source and adjacent configuration before starting its scientific calculation. No scientific command is described as already executed here. CLI-required workspaces/configuration or licensed upstream outputs must be supplied explicitly. Original source-hash/authorization guards may reject modified or incomplete inputs and are not weakened.

Numerical input consistency: `python 09_REPRODUCTION_CHECKS/verify_input_identity.py` verifies ordered IDs/features, fixed ET SHA and standardization using absolute tolerance 1e-12. It loads no predictive model and performs no explanation fits. See [input preparation](INPUT_PREPARATION.md).

Source files:

- [prepare_data.py](prepare_data.py)
- [verify_repository.py](verify_repository.py)
- [verify_saved_metrics.py](verify_saved_metrics.py)

Author-owned material is MIT-licensed; third-party terms remain applicable. See the root [licence and source notes](../DATA_AND_LICENSE_AUDIT.md).
