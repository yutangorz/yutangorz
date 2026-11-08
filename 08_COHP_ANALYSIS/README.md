# Pair/orbital-resolved electronic response

Original spectra and matched-nine-state quantitative analysis are separate scientific series. Occupied ICOHP and moving CBM ±0.1 eV integrals are different estimands.

Main source: [src/response_matrix/matrix_analysis_v2.py](src/response_matrix/matrix_analysis_v2.py).

Inputs/configuration: inputs/original_six and inputs/matched_nine; data and src by analysis purpose; final small tables in module07/data/saved_statistics. Generated scientific products belong under the configured outputs/08_COHP_ANALYSIS/ scientific protocol directory; no output is produced by opening or reading this README.

Prerequisites/limits: LOBSTER and preceding DFT outputs are required separately. Original-series complete input/layer identity remains partial. Source-specific projection limits and sign conventions are not removed.

To check the repository without executing science, run `python 09_REPRODUCTION_CHECKS/verify_repository.py` and `python 09_REPRODUCTION_CHECKS/verify_saved_metrics.py` from the repository root. Read the named source and adjacent configuration before starting its scientific calculation. No scientific command is described as already executed here. CLI-required workspaces/configuration or licensed upstream outputs must be supplied explicitly. Original source-hash/authorization guards may reject modified or incomplete inputs and are not weakened.

Source files:

- [src/matched_nine/prepare_a06_v2.py](src/matched_nine/prepare_a06_v2.py)
- [src/matched_nine/run_lobster_guarded.py](src/matched_nine/run_lobster_guarded.py)
- [src/orbital_windows/analyze_a09.py](src/orbital_windows/analyze_a09.py)
- [src/original_spectra/reuse_core.py](src/original_spectra/reuse_core.py)
- [src/pair_windows/remote_analysis.py](src/pair_windows/remote_analysis.py)
- [src/pair_windows/report.py](src/pair_windows/report.py)
- [src/projection_checks/independent_output_audit_v2.py](src/projection_checks/independent_output_audit_v2.py)
- [src/response_matrix/matrix_analysis_v2.py](src/response_matrix/matrix_analysis_v2.py)
- [src/response_matrix/summarize_projections.py](src/response_matrix/summarize_projections.py)
