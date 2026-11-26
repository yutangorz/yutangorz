# Environments and dependencies

There is no certified universal environment or complete historical ALIGNN environment. Module requirements and environments/SAVED_ENVIRONMENT_VERSIONS.csv preserve actual saved version evidence. Install compatible dependencies in an isolated environment for the scientific module concerned; installing latest packages is not a claim of historical equivalence.

File integrity and documentation checks use the Python standard library. Numerical input identity and array/heldout-index checks additionally use NumPy without loading any predictive model. Scientific analysis uses numpy, pandas, scipy, scikit-learn, xgboost, joblib, pymatgen, matminer, shap and lime; JARVIS scripts use jarvis-tools (import name jarvis), with tqdm and psutil where imported. ALIGNN additionally requires compatible PyTorch, DGL, jarvis-tools and ALIGNN; see the module implementation notes before choosing versions. Obtain these packages through their official distributions; a package listing alone does not certify dependency resolution or historical equivalence.

VASP, PAW and LOBSTER are obtained separately. The optional/licensed DFT/COHP prerequisites are described in the SI.

Use [input preparation](09_REPRODUCTION_CHECKS/INPUT_PREPARATION.md). Pure checks can be run from any current directory using absolute script paths. Some scientific entrypoints execute at import: do not import them to inspect syntax.
