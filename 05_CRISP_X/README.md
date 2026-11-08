# Structure-aware local response interpretation

Constrained geometric/chemical perturbations, descriptor re-extraction, fixed ET queries, local reconstruction and support. Shared packages common/, crispx/, crispx_closure/, lime_baseline/ and shap_baseline/ retain import structure.

Main source: [src/study_closure.py](src/study_closure.py).

Inputs/configuration: models/; configs/local_reconstruction.json; data/local_reconstruction; shared descriptors in module03. Generated scientific products belong under the configured outputs/05_CRISP_X/ scientific protocol directory; no output is produced by opening or reading this README.

Prerequisites/limits: Original development-pool/confirmation prerequisites and source receipts are not fully closed. Returning a response does not prove L/Θ separate slope identifiability.

To check the repository without executing science, run `python 09_REPRODUCTION_CHECKS/verify_repository.py` and `python 09_REPRODUCTION_CHECKS/verify_saved_metrics.py` from the repository root. Read the named source and adjacent configuration before starting its scientific calculation. No scientific command is described as already executed here. CLI-required workspaces/configuration or licensed upstream outputs must be supplied explicitly. Original source-hash/authorization guards may reject modified or incomplete inputs and are not weakened.

Source files:

- [src/audited_paths.py](src/audited_paths.py)
- [src/common/__init__.py](src/common/__init__.py)
- [src/common/data.py](src/common/data.py)
- [src/common/evaluation.py](src/common/evaluation.py)
- [src/common/oracle.py](src/common/oracle.py)
- [src/crispx/__init__.py](src/crispx/__init__.py)
- [src/crispx/core.py](src/crispx/core.py)
- [src/crispx/path_model.py](src/crispx/path_model.py)
- [src/crispx/perturb.py](src/crispx/perturb.py)
- [src/crispx/run.py](src/crispx/run.py)
- [src/crispx_closure/__init__.py](src/crispx_closure/__init__.py)
- [src/crispx_closure/paths.py](src/crispx_closure/paths.py)
- [src/lime_baseline/__init__.py](src/lime_baseline/__init__.py)
- [src/lime_baseline/compare.py](src/lime_baseline/compare.py)
- [src/refined.py](src/refined.py)
- [src/shap_baseline/__init__.py](src/shap_baseline/__init__.py)
- [src/shap_baseline/importance.py](src/shap_baseline/importance.py)
- [src/study.py](src/study.py)
- [src/study_closure.py](src/study_closure.py)
