# Input preparation and checks

## Input identity

`python 09_REPRODUCTION_CHECKS/verify_input_identity.py` checks the fixed ET SHA, 7176 unique material IDs, the ordered 1462-material held-out set, nine ordered descriptors and standardization inputs without loading the model. Numerical consistency uses rtol=0, absolute tolerance 1e-12 and matching nonfinite masks. Use `--output <new-file>` to save the result; existing receipts are never overwritten. A successful check is input consistency, not a LIME/SHAP fidelity or explanation-equivalence test.

## Workspaces

Set JCTC_WORKSPACE_ROOT to an empty external directory, then run `python 09_REPRODUCTION_CHECKS/prepare_workspaces.py --stage all` to copy minimum configs, code and inputs for et/xgboost/crispx/lime/external. Existing stage directories are refused. INPUT_CODE_SHA256 and FROZEN_INPUT_CODE describe that workspace's actual files. Archived configs/*source_identity* records retain their own identities. Native LIME and external drivers require their declared source/qualification gates; workspace preparation alone does not satisfy these gates or authorize scientific execution.

The native LIME driver expects an IDENTITY_AUDIT.json under the configured outputs/04_ET_XGBOOST_INTERPRETATION/native_lime directory. Generate that receipt with verify_input_identity.py --output <new-receipt-path> after preparing its parent location. Its passed flag reflects the checks above. Further source-hash requirements remain independent. Use a new output location rather than overwriting historical records.

## Structures

The 24 OQMD parent POSCARs are in 03_DESCRIPTOR_EXTRACTION/data/structures; all 7176 CIF records remain in the compressed source table. All 38 selected JARVIS CIFs are in 06_EXTERNAL_TRANSFER/data/external_response/cif. `recover_structures.py --source oq24|j38 --output <external-directory>` copies exact files and verifies frozen SHA. Optional `--source oqmd-raw-cif --ids <comma-separated-IDs>` exports raw database text, without asserting byte identity to POSCAR serialization. Database recovery must use the fixed source version, IDs and hashes in the selection workflow; never silently replace an original identity after reserialization.

## Tables and perturbation inputs

Grouped CV/OOF, native LIME, local response and conditioned attribution tables are in their module data directories. The 5714-row training matrix, standard deviations, anchors, 76 physical pools and 38 Gaussian candidates are fixed inputs. INPUT_PATH_BINDINGS.json maps archived reads while preserving table bytes, sample order and feature order. CASE_INPUTS76 and scientific provenance manifests retain their original identities. Candidate loading verifies manifest SHA, parent IDs, shape/dtype, file bytes and anchor consistency. PUBLICATION_BINDING.json records the location-specific candidate manifest identity; it is not a scientific execution approval.

Use repo_path for new workspaces/outputs and input_path only for saved-input reads. JCTC_INPUT_ROOT supplies separately stored inputs, JCTC_OUTPUT_ROOT redirects generated outputs. Never direct generated outputs into fixed data directories. Regenerating OOF, perturbation pools or explanations is a scientific operation and requires its own declared configuration and valid gates.
