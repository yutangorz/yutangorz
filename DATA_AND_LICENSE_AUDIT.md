# Licence and data sources

The [MIT License](LICENSE), Copyright (c) 2026 Geming Fu, applies to the original code, processing and descriptor implementations, trained models, author-generated research data, predictions, interpretation results, DFT/COHP analysis results and project documentation, to the extent the authors hold the necessary rights. No separate data licence is assigned to these original materials.

Third-party materials remain under their respective licences and terms. The project MIT grant does not relicense source databases, upstream software or proprietary packages.

| Material | Source and terms | Repository scope |
|---|---|---|
| OQMD | https://oqmd.org/ ; CC BY 4.0 | Source and selected data retain OQMD attribution; processing is described in 01_DATA_CURATION. |
| JARVIS data | Fixed jdft_3d-9-24-2025.json; DOI 10.6084/m9.figshare.6815699.v11; CC BY 4.0 as recorded in selection/build_manifest.py | 38 parent structures, source identifiers, descriptor data and evaluation cases retain database provenance. Original analysis code/results are MIT; source-derived content retains applicable database terms. Cite Choudhary et al., npj Computational Materials 6, 173 (2020), DOI 10.1038/s41524-020-00440-1. Mixed structural source lineage is recorded by the selection workflow and is not relabelled as wholly author-created data. |
| CGCNN | Upstream MIT | Elemental table and adapted code retain [CGCNN notice](Third_Party_Notices/CGCNN_LICENSE.txt). |
| ALIGNN | NIST Terms of Use | Official source retains [ALIGNN notice](Third_Party_Notices/ALIGNN_NIST_TERMS.rst). Author-trained weights and original results are MIT; upstream source is not relabelled. |
| JARVIS software | NIST Terms of Use | [JARVIS notice](Third_Party_Notices/JARVIS_NIST_TERMS.rst) covers the software; the database terms above apply separately. |
| PVK, ET and XGBoost weights | Author-trained models; MIT | Model IDs, feature definitions and SHA256 records identify the fixed weights. Third-party training/inference libraries retain their own licences. |
| VASP, PAW and LOBSTER | Proprietary software/potential terms | Obtain required software and potentials separately. The repository's original parameter files and analysis results do not grant a software or PAW licence. |

Preserve upstream attribution and notices when redistributing or modifying source-derived material. The saved database/source metadata support the scope stated here; no blanket grant is made for unrelated third-party content.
