# Reproducibility scope

The repository provides research code, fixed models, input data, saved results and source identities. File integrity, input consistency and saved-result checks are distinct from end-to-end scientific execution and physical validation.

- Candidate selection uses the 47-feature PVK classifier and strict probability threshold p>0.5. The classifier is distinct from the nine-descriptor band-gap regressors.
- ALIGNN training records contain configuration, weights, loss histories and predictions. Scalar MAE/L1 and constant effective AdamW learning rate 5e-4 follow from compatible official implementations and saved records. The exact installed version and stepwise LR logs were not retained; see [training implementation](02_ALIGNN_TRAINING/TRAINING_IMPLEMENTATION.md).
- Conditioned ET attribution compares parent-to-perturbed allocation changes: 2304 points, 24 parents and 48 pools. Geometry has 1152 points; the non-abstain chemical scope has 1144. Eight chemical abstentions remain in the complete table. These are model allocations, not independent causal or DFT effects.
- The 24 OQMD parent structures, 38 JARVIS structures, 76 physical pools, arrays, anchors and Gaussian candidates have fixed identities. Scientific source tables, ordering and historical hash records remain authoritative.
- Numerical input consistency is verified using rtol=0 and an absolute tolerance of 1e-12, with matching nonfinite-value masks, to account for floating-point representation and serialization differences. ID and row ordering, feature ordering, model SHA and standardization inputs are checked separately. Numerical equivalence does not prove all LIME explanations or local-fidelity metrics identical.
- Source, qualification and authorization gates remain active where required. An input-workspace manifest certifies the files in that workspace; it does not replace a historical execution approval. Runtime scientific workflows and a universal compatible environment have not been independently certified by the integrity checks.
- DFT and COHP remain protocol-specific, with their limitations described in the modules and SI. Proprietary software and the SI-described external prerequisites are obtained separately.

Checks: `python 09_REPRODUCTION_CHECKS/verify_repository.py` and `python 09_REPRODUCTION_CHECKS/verify_input_identity.py`. See [input preparation](09_REPRODUCTION_CHECKS/INPUT_PREPARATION.md) and [licence/source information](DATA_AND_LICENSE_AUDIT.md).
