# OQ24/J38 external response evaluation

JARVIS selection, local response evaluation and Gaussian-versus-structure common-Ridge sampling controls. Method-configuration comparisons are separate experiments.

Main source: [src/sampling_control/production_factory.py](src/sampling_control/production_factory.py).

Inputs/configuration: configs/; data/external_response/PILOT38.csv; data/common_ridge. Generated scientific products belong under the configured outputs/06_EXTERNAL_TRANSFER/ scientific protocol directory; no output is produced by opening or reading this README.

Prerequisites/limits: JARVIS external structures/derived arrays are withheld for rights review. Local reconstruction fidelity is against fixed ET, not independent DFT band gap accuracy. Original authorization/receipt requirements are retained.

To check the repository without executing science, run `python 09_REPRODUCTION_CHECKS/verify_repository.py` and `python 09_REPRODUCTION_CHECKS/verify_saved_metrics.py` from the repository root. Read the named source and adjacent configuration before starting its scientific calculation. No scientific command is described as already executed here. CLI-required workspaces/configuration or licensed upstream outputs must be supplied explicitly. Original source-hash/authorization guards may reject modified or incomplete inputs and are not weakened.

Source files:

- [src/common_ridge/execute_control.py](src/common_ridge/execute_control.py)
- [src/common_ridge/refined.py](src/common_ridge/refined.py)
- [src/common_ridge/statistics_locked.py](src/common_ridge/statistics_locked.py)
- [src/constrained_lime/pilot.py](src/constrained_lime/pilot.py)
- [src/external_response/audited_paths.py](src/external_response/audited_paths.py)
- [src/external_response/qualify.py](src/external_response/qualify.py)
- [src/external_response/refined.py](src/external_response/refined.py)
- [src/external_response/resume_secondary.py](src/external_response/resume_secondary.py)
- [src/external_response/study.py](src/external_response/study.py)
- [src/external_response/study_closure.py](src/external_response/study_closure.py)
- [src/external_response/test_closure.py](src/external_response/test_closure.py)
- [src/external_response/variant_f7ef64c6/prepare.py](src/external_response/variant_f7ef64c6/prepare.py)
- [src/gaussian_sampling/preflight.py](src/gaussian_sampling/preflight.py)
- [src/sampling_control/candidate_store.py](src/sampling_control/candidate_store.py)
- [src/sampling_control/full_driver.py](src/sampling_control/full_driver.py)
- [src/sampling_control/production_factory.py](src/sampling_control/production_factory.py)
- [src/selection/build_manifest.py](src/selection/build_manifest.py)
- [src/selection/build_raw_candidates.py](src/selection/build_raw_candidates.py)
- [src/selection/deduplicate_external.py](src/selection/deduplicate_external.py)
- [src/selection/domain_distance.py](src/selection/domain_distance.py)
- [src/selection/extract_external_features.py](src/selection/extract_external_features.py)
- [src/selection/filter_inorganic.py](src/selection/filter_inorganic.py)
- [src/selection/select_master.py](src/selection/select_master.py)
- [src/selection/select_splits.py](src/selection/select_splits.py)
- [src/selection/validate_delivery.py](src/selection/validate_delivery.py)
