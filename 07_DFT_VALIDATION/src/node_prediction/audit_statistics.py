
# Portable repository locations and local scientific packages.
from pathlib import Path as _RepositoryPath
import sys as _repository_sys
_repository_root = next((p for p in _RepositoryPath(__file__).resolve().parents if (p / "repository_paths.py").is_file()), None)
if _repository_root is None:
    raise RuntimeError("Repository root marker is missing")
_repository_sys.path.insert(0, str(_repository_root))
from repository_paths import repo_path as _repo_path
_repository_sys.path.insert(0, str(_repo_path('07_DFT_VALIDATION/src/substitution_qualification')))
_repository_sys.path.insert(0, str(_repo_path('07_DFT_VALIDATION/src/soc_qualification')))
_repository_sys.path.insert(0, str(_repo_path('07_DFT_VALIDATION/src/input_guards')))
_repository_sys.path.insert(0, str(_repo_path('07_DFT_VALIDATION/src/validation')))
_repository_sys.path.insert(0, str(_repo_path('03_DESCRIPTOR_EXTRACTION/src/shared_features')))
_repository_sys.path.insert(0, str(_repo_path('03_DESCRIPTOR_EXTRACTION/src')))
_repository_sys.path.insert(0, str(_repo_path('05_CRISP_X/src')))
_repository_sys.path.insert(0, str(_repo_path('07_DFT_VALIDATION/src/node_prediction')))
from pathlib import Path
import json,hashlib
import numpy as np,pandas as pd
from scipy.stats import spearmanr
R=_repo_path('07_DFT_VALIDATION/workflows/node_prediction');P=_repo_path('07_DFT_VALIDATION/workflows/response_pairing');O=_repo_path('outputs/07_DFT_VALIDATION/response_pairing')
b=pd.read_csv(_repo_path('outputs/07_DFT_VALIDATION/response_pairing/FIG6B_ELEMENT_PAIRED_LONG.csv'));w=pd.read_csv(_repo_path('outputs/07_DFT_VALIDATION/response_pairing/FIG6B_ELEMENT_PAIRED_WIDE.csv'))
assert len(b)==140 and b.material_id.nunique()==20 and len(w)==20
assert np.max(abs(b.sub_mean-b.parent_mean-b.delta))<1e-12
assert np.max(abs(b.Eg_sub_eV-b.Eg_parent_eV-b.delta_Eg_eV))<1e-12
st=pd.read_csv(_repo_path('outputs/07_DFT_VALIDATION/response_pairing/FIG6B_PROTOCOL_STRATIFIED_BOOTSTRAP.csv'));dr=pd.read_csv(_repo_path('outputs/07_DFT_VALIDATION/response_pairing/FIG6B_PROTOCOL_BOOTSTRAP_DRAWS.csv'))
maxdiff=0
for r in st.itertuples():
 g=w[(w.analysis_scope=='FORMAL_FIXED_13')&(w.protocol_class==r.protocol_class)]
 x=g[r.descriptor].to_numpy();y=g.delta_Eg_eV.to_numpy()
 expected=float(spearmanr(x,y).statistic) if len(g)>2 and np.ptp(x)>0 else np.nan
 assert np.isfinite(expected)==np.isfinite(r.rho)
 if np.isfinite(expected):maxdiff=max(maxdiff,abs(expected-r.rho))
 boot=dr[(dr.protocol_class==r.protocol_class)&(dr.descriptor==r.descriptor)].rho.dropna()
 assert len(boot)==r.valid_draws
 if np.isfinite(r.display_ci_low):
  assert len(boot)>=1600
  assert abs(r.display_ci_low-boot.quantile(.025))<1e-12 and abs(r.display_ci_high-boot.quantile(.975))<1e-12
c=pd.read_csv(_repo_path('outputs/07_DFT_VALIDATION/response_pairing/FIG6C_GEOMETRY_PAIRED_PROTOCOL_ANNOTATED.csv'));assert len(c)==440 and c.point_id.nunique()==440
main=c[c.primary_distribution];assert len(main)==371 and main.material_id.nunique()==17
assert (abs(main.delta_L_A)>1e-10).all() and (abs(main.delta_Theta_degree)>1e-10).all()
assert (c.protocol_status=='KEY_INPUTS_EXACT_MATCH').all()
assert not c.LSORBIT.any()
df=pd.read_csv(_repo_path('outputs/07_DFT_VALIDATION/response_pairing/FIG6D_PRIMARY_DISPLAY_QUALIFICATION.csv'));assert len(df)==45
assert df[df.method=='DFT'].primary_figure_score.isna().all() and df[df.method=='DFT'].primary_figure_rank.isna().all()
sources=pd.read_csv(_repo_path('07_DFT_VALIDATION/workflows/response_pairing/INPUT_SOURCE_MAPPING.csv'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert all(sha(r.source_path)==sha(r.snapshot_path)==r.sha256 for r in sources.itertuples())
qa={'status':'PASS','element_pairs':20,'element_long_rows':140,'within_protocol_rho_maxdifference':maxdiff,
 'bootstrap_validity_and_display_gate_pass':True,'structure_rows':440,'primary_points':371,'primary_parents':17,
 'all_primary_global_LTheta_cochange':True,'DFT9_scores_not_fabricated':True,'source_sha_pass':True}
(_repo_path('07_DFT_VALIDATION/workflows/node_prediction/STATISTICAL_INDEPENDENT_QA.json')).write_text(json.dumps(qa,indent=2),encoding='utf-8');print(json.dumps(qa,indent=2))
