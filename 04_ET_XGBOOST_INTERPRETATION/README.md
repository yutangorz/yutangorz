# Independent models and attribution

Independent ET/XGBoost regression on database band gaps, grouped CV, SHAP, Gaussian LIME and refitted single/group deletion. Grouped deletion nonadditivity is predictive, not causal.

Main source: [src/grouped_cv/ablation.py](src/grouped_cv/ablation.py).

Inputs/configuration: configs/ and data/ organised by grouped_cv, native_lime, grouped_ablation and xgboost_ablation; shared descriptors in module03. Generated scientific products belong under the configured outputs/04_ET_XGBOOST_INTERPRETATION/ scientific protocol directory; no output is produced by opening or reading this README.

Prerequisites/limits: Some analyses require upstream fitted models, OOF products, local attribution rows and valid input identities. Prior receipt/hash checks remain; do not bypass them.

To check the repository without executing science, run `python 09_REPRODUCTION_CHECKS/verify_repository.py` and `python 09_REPRODUCTION_CHECKS/verify_saved_metrics.py` from the repository root. Read the named source and adjacent configuration before starting its scientific calculation. No scientific command is described as already executed here. CLI-required workspaces/configuration or licensed upstream outputs must be supplied explicitly. Original source-hash/authorization guards may reject modified or incomplete inputs and are not weakened.

Source files:

- [src/conditioned_shap/analyze.py](src/conditioned_shap/analyze.py)
- [src/grouped_ablation/ablation_core.py](src/grouped_ablation/ablation_core.py)
- [src/grouped_ablation/compare_shap.py](src/grouped_ablation/compare_shap.py)
- [src/grouped_ablation/explain_tree.py](src/grouped_ablation/explain_tree.py)
- [src/grouped_ablation/grouped_ablation.py](src/grouped_ablation/grouped_ablation.py)
- [src/grouped_ablation/original_ablation.py](src/grouped_ablation/original_ablation.py)
- [src/grouped_cv/ablation.py](src/grouped_cv/ablation.py)
- [src/native_lime/assemble_analysis.py](src/native_lime/assemble_analysis.py)
- [src/native_lime/compute_lime.py](src/native_lime/compute_lime.py)
- [src/xgboost_ablation/ablation_core.py](src/xgboost_ablation/ablation_core.py)
- [src/xgboost_ablation/finalize_ablation.py](src/xgboost_ablation/finalize_ablation.py)
- [src/xgboost_ablation/study_ablation.py](src/xgboost_ablation/study_ablation.py)
- [src/xgboost_ablation/test_ablation.py](src/xgboost_ablation/test_ablation.py)
