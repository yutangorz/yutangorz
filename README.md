# Interpretable band gap modelling of perovskite candidates

Scientific code and necessary input records for the JCTC manuscript *Physical Insights of Band Gap in Perovskites Revealed by Interpretable ALIGNN*. Use the published paper citation/DOI when assigned. This repository records computational methods and necessary input sources; it does not promise item-by-item reproduction of historical numerical results.

## Scientific workflow

OQMD selection → ALIGNN prediction → material-level nine-descriptor representation → independent ET/XGBoost prediction → global/local interpretation → external local-response evaluation → independent DFT response comparison → COHP electronic-structure analysis.

The 72,747-material OQMD source and independent 61,912-material initial-analysis branch must not be described as one linear filtering chain. The final modelling set has 7,176 perovskite candidates. ALIGNN motivates information types; the descriptors do not reproduce its full graph or hidden representation. ET and XGBoost regress database band gaps independently and are not ALIGNN surrogate models.

| Module | Scientific role |
|---|---|
| [01_DATA_CURATION](01_DATA_CURATION/README.md) | Candidate selection, completeness, energetic screening and redundancy reduction |
| [02_ALIGNN_TRAINING](02_ALIGNN_TRAINING/README.md) | Actual training/configuration and fixed splits |
| [03_DESCRIPTOR_EXTRACTION](03_DESCRIPTOR_EXTRACTION/README.md) | Seven elemental properties and material mean L/Θ |
| [04_ET_XGBOOST_INTERPRETATION](04_ET_XGBOOST_INTERPRETATION/README.md) | Independent models, SHAP/LIME, correlation and refitted ablation |
| [05_CRISP_X](05_CRISP_X/README.md) | Structure perturbations, fixed ET response reconstruction and support |
| [06_EXTERNAL_TRANSFER](06_EXTERNAL_TRANSFER/README.md) | OQ24/J38 evaluation and common-Ridge sampling controls |
| [07_DFT_VALIDATION](07_DFT_VALIDATION/README.md) | Protocol-resolved DFT, association and direction/amplitude statistics |
| [08_COHP_ANALYSIS](08_COHP_ANALYSIS/README.md) | Separate original/matched series, occupied ICOHP and energy-window analysis |
| [09_REPRODUCTION_CHECKS](09_REPRODUCTION_CHECKS/README.md) | Integrity and deterministic saved-statistic verification |

## Inputs, environment and locations

Use each module's src/, configs/, data/, models/ or inputs/ according to its README. Material IDs, node IDs, model versions and calculation protocols are retained. DFT inputs remain separated by material/protocol/state; matched COHP SCF and projection inputs remain distinct. [Manuscript mapping](MANUSCRIPT_CODE_TRACEABILITY.csv) identifies entrypoints, inputs, expected outputs and limitations.

The seven elemental properties are period, electronegativity, covalent radius, valence electrons, ionization energy, electron affinity and atomic volume, averaged by atom fraction. L and Θ are material-level means over periodic local environments, not selected local bonds/angles. Local DFT perturbations can change both means.

See [environment notes](DEPENDENCIES_AND_ENVIRONMENT.md) and [saved versions](environments/SAVED_ENVIRONMENT_VERSIONS.csv). No single universal environment is certified. repository_paths.py resolves repository-relative locations; JCTC_INPUT_ROOT can supply separately obtained/decompressed inputs and JCTC_OUTPUT_ROOT can redirect output storage. No private absolute path is required by the location helper.

## Checked operations

From the repository root, these standard-library checks execute no scientific stages:

```text
python 09_REPRODUCTION_CHECKS/verify_repository.py
python 09_REPRODUCTION_CHECKS/verify_saved_metrics.py
python 09_REPRODUCTION_CHECKS/prepare_data.py --output ../jctc_inputs_new --dry-run
```

To decompress included large source data, remove --dry-run and choose a nonexistent directory outside this repository, then set JCTC_INPUT_ROOT to that directory. This only copies/decompresses inputs. Scientific stages require their own declared input schemas, configurations and generated upstream tables. Original source/receipt/authorization checks remain in force. Missing prerequisites are listed in [reproducibility notes](REPRODUCIBILITY_QA.md), not bypassed. Do not import entry scripts merely to inspect them; some have top-level computation. Do not load untrusted pickle/joblib files.

## Interpretation and independent evidence

SHAP/LIME describe model behaviour. CRISP-X reconstructs fixed-ET responses to explicit structural changes and qualifies local support. Common-Ridge sampling controls differ from comparisons of differently configured explanation methods. J38 evaluation concerns ET response reconstruction, not external DFT/experimental band gap accuracy.

DFT evidence is protocol- and sample-specific. All 371 formal nodes/17 parents and the 124 in-range/247 out-of-range split remain. Overall CRISP-X direction agreement is 59.030%, below the 70.350% majority baseline; response MAE is 4.003 meV and R² is −4.139611. Limited favourable subsets do not establish general DFT-response prediction. COHP supplies conditional pair/orbital/energy-region evidence, not a unique universal causal mechanism.

## Data, rights and citation

OQMD data require attribution under CC BY 4.0. CGCNN elemental/upstream material uses the retained MIT notice; ALIGNN/JARVIS software notices are in Third_Party_Notices/. VASP, PAW and LOBSTER software are obtained separately. JARVIS structure/array and PVK classifier redistribution is withheld pending rights review; no replacement model was trained. The authors' code/model/data license grant awaits author choice. See [data and licence notes](DATA_AND_LICENSE_AUDIT.md). Cite the paper and the relevant OQMD, CGCNN, ALIGNN, ExtraTrees, XGBoost, SHAP, LIME, JARVIS, VASP and LOBSTER primary sources using the verified manuscript bibliography.
