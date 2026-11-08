"""Frozen design audit only: no VASP inputs, model query, network or submission."""

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
_repository_sys.path.insert(0, str(_repo_path('07_DFT_VALIDATION/src/substitution_design')))
from pathlib import Path
import csv,hashlib,json,math,platform,shutil,sys
import numpy as np
import pandas as pd
from pymatgen.core import Structure
from pymatgen.core.periodic_table import Element
RUN=_repo_path('unavailable_inputs/source_d4c9faa0838c/input_bundle');ST=_repo_path('unavailable_inputs/source_6a438450d638/06_cluster_delivery');ROOT=_repo_path('.')
PRIMARY={
 '1229318':('Sb','Sn','valence_changing_candidate: formal oxidation/carrier review mandatory'),
 '2005394':('Bi','In','valence_changing_candidate: formal oxidation/carrier review mandatory'),
 '2602856':('Zr','Hf','same_periodic_group_candidate; Eu magnetic/f-electron gate'),
 '6235374':('Rb','K','same_group_A_site_candidate; avoid historical O-to-halogen substitution'),
 '6313886':('Rh','Ir','same_group_transition_metal_candidate; magnetic/SOC gate'),
 '6314177':('Cu','Ag','same_group_transition_metal_candidate; oxidation/magnetic gate'),
 '6353933':('Br','Cl','same_group_halogen_candidate'),
 '6796121':('Sc','Y','previously_authorized_endmember; accepted reference available conditionally'),
 '7750830':('W','Mo','same_group_metal_candidate; Yb magnetic/f-electron gate; avoid historical N-to-O'),
 '7907750':('I','Br','same_group_halogen_candidate'),
 '8158832':('Ga','Al','same_group_candidate; formal oxidation/electron count review still required'),
 '8269167':('Tl','Pb','valence_changing_candidate: formal oxidation/carrier review mandatory'),
 '8282736':('I','Br','same_group_halogen_candidate; Sm magnetic/f-electron gate'),
 '8303092':('Tl','In','same_group_candidate; Tb magnetic/f-electron gate'),
 '8366012':('Pb','Sn','previously_authorized_endmember; accepted reference available conditionally'),
 '8550885':('Ta','Nb','same_group_metal_candidate'),
 '8638126':('F','Cl','same_group_halogen_candidate; fixed-geometry contact/force review'),
 '8997093':('Sn','Ge','same_group_candidate; Sm magnetic/f-electron gate'),
 '9136060':('Te','Se','same_group_candidate; Eu magnetic/f-electron gate'),
 '9836170':('Br','Cl','same_group_halogen_candidate; 100-site cell requires workload review')}
COLS=['Period','Electronegativity','Covalent Radius','Valence Electrons','First Ionization Energy','Electron Affinity','Atomic Volume']
FEATURES=['period','electronegativity','covalent_radius','valence_electrons','ionization_energy','electron_affinity','atomic_volume']
SOURCE=_repo_path('07_DFT_VALIDATION/data/material_selection/structures')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):
    with p.open('x',encoding='utf-8') as f:json.dump(v,f,ensure_ascii=False,indent=2)
def table(name,rows):
    with (_repo_path('unavailable_inputs/source_72d8b75d2b30/results')/name).open('x',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
    for d in ['inputs','inputs/reference_geometries','results','reports']:(RUN/d).mkdir(parents=True,exist_ok=True)
    for src,name in [
       (_repo_path('unavailable_inputs/source_20f6d76b6f4f/historical_final20.csv'),'HISTORICAL_FINAL20.csv'),
       (_repo_path('unavailable_inputs/source_549bc7506931/atom_importance.csv'),'atom_importance.csv'),
       (_repo_path('unavailable_inputs/source_2d953d19079b/TABLE_SUBSTITUTIONS_MISSINGNESS.csv'),'HISTORICAL_SUBSTITUTIONS.csv'),
       (_repo_path('outputs/07_DFT_VALIDATION/spectrum_qualification/REFERENCE_20_HEALTH.csv'),'HISTORICAL_REFERENCE_HEALTH.csv'),
       (_repo_path('unavailable_inputs/source_2699bfeee0f8/FULL_38_CALL_STATUS.csv'),'SUPERSEDED_TWO_HOST_PLAN_STATUS.csv'),
       (_repo_path('unavailable_inputs/source_3eee3abcddb7/FEATURE_DEFINITION.json'),'FEATURE_DEFINITION.json')]:
        assert not (_repo_path('unavailable_inputs/source_e2da7f3b1259/inputs')/name).exists();shutil.copy2(src,_repo_path('unavailable_inputs/source_e2da7f3b1259/inputs')/name)
    cohort=list(csv.DictReader((_repo_path('unavailable_inputs/source_f5b04a57abc1/HISTORICAL_FINAL20.csv')).open(encoding='utf-8-sig')))
    assert len(cohort)==20 and {r['material_id'] for r in cohort}==set(PRIMARY)
    frame=pd.read_csv(_repo_path('unavailable_inputs/source_68e0418c8f11/atom_importance.csv')).set_index('Symbol')[COLS].astype(float)
    medians=frame.median().to_numpy();rows=[];matrix=[];same=[];checks=[];source_rows=[]
    for mid in sorted(PRIMARY):
        old,new,rationale=PRIMARY[mid]
        # Use accepted current geometry only for the two already tested hosts.
        src=(_repo_path('unavailable_inputs/source_7471e5ebb16c/inputs')/(mid+'_reference.vasp')) if mid in ['8366012','6796121'] else SOURCE/(mid+'.vasp')
        dst=_repo_path('unavailable_inputs/source_437cb6fc2952/reference_geometries')/(mid+'.vasp');shutil.copy2(src,dst)
        s=Structure.from_file(dst);symbols=[x.specie.symbol for x in s]
        assert old in symbols and new in frame.index
        b=s.copy();b.replace_species({old:new})
        assert len(s)==len(b)
        assert np.array_equal(s.frac_coords,b.frac_coords) and np.array_equal(s.lattice.matrix,b.lattice.matrix)
        raw0=frame.loc[symbols].to_numpy();raw1=frame.loc[[x.specie.symbol for x in b]].to_numpy()
        x0=np.where(np.isfinite(raw0),raw0,medians).mean(axis=0);x1=np.where(np.isfinite(raw1),raw1,medians).mean(axis=0)
        delta=x1-x0;fraction=symbols.count(old)/len(s)
        expected=fraction*(frame.loc[new].fillna(frame.median()).to_numpy()-frame.loc[old].fillna(frame.median()).to_numpy())
        assert np.allclose(delta,expected,rtol=0,atol=1e-12)
        same_group=Element(old).group==Element(new).group
        missing=int(frame.loc[[old,new]].isna().sum().sum())
        assert missing==0,'Primary source/replacement columns must be observed'
        matrix.append(delta)
        if same_group:same.append(delta)
        magnetic=sorted(set(symbols)&{'Eu','Sm','Tb','Yb','Rh','Cu'})
        rows.append({'material_id':mid,'parent_formula':s.composition.reduced_formula,'candidate_formula':b.composition.reduced_formula,
          'original_element':old,'replacement_element':new,'nsites':len(s),'replaced_sites':symbols.count(old),'site_fraction':fraction,
          'scope':'all_sites_of_original_element; neutral-cell composition change; fixed positions and lattice',
          'same_periodic_group_not_proven_isovalent':same_group,'valence_descriptor_changes':bool(abs(delta[3])>1e-12),
          'magnetic_or_f_electron_screen':';'.join(magnetic),'chemical_rationale_or_hold':rationale,
          'ready_for_submission':False,'status':'proposed_design_only_pending_chemistry_PAW_spin_numeric_review',
          'reference_reuse_job':'12015' if mid=='8366012' else '12029' if mid=='6796121' else '',
          'geometry_status':'accepted_scalar_reference_SOC_geometry' if mid in ['8366012','6796121'] else 'frozen_database_geometry_not_certified_equilibrium',
          'source_path':str(src),'source_sha256':sha(src),'source_new_element_missing_count':missing,
          **{'delta_'+f:float(v) for f,v in zip(FEATURES,delta)}})
        source_rows.append({'material_id':mid,'source':str(src),'sha256':sha(src),'bytes':src.stat().st_size})
        checks.append({'material_id':mid,'all_passed':True,'checks':['cohort_identity','site_order_count','exact_fixed_geometry','full_element_replacement','table_mean_delta','observed_seven_element_columns']})
    X=np.array(matrix);scale=X.std(axis=0);assert (scale>1e-12).all()
    Z=X/scale;sv=np.linalg.svd(Z,compute_uv=False);centered=Z-Z.mean(axis=0)
    loo=[{'omitted_material':mid,'rank':int(np.linalg.matrix_rank(np.delete(Z,i,axis=0))),
          'condition_number':float(np.linalg.cond(np.delete(Z,i,axis=0)))} for i,mid in enumerate(sorted(PRIMARY))]
    assert np.linalg.matrix_rank(Z)==7 and all(x['rank']==7 for x in loo)
    same=np.array(same);assert np.max(np.abs(same[:,3]))<1e-12
    table('PROPOSED_20_PRIMARY_SUBSTITUTIONS.csv',rows)
    table('SOURCE_GEOMETRY_PROVENANCE.csv',source_rows)
    table('LEAVE_ONE_HOST_DESIGN_RANK.csv',loo)
    table('STANDARDIZED_CHANGE_MATRIX.csv',[{'material_id':mid,**dict(zip(FEATURES,[float(v) for v in z]))} for mid,z in zip(sorted(PRIMARY),Z)])
    table('SINGULAR_VALUES.csv',[{'index1':i+1,'singular_value':float(v)} for i,v in enumerate(sv)])
    save(_repo_path('unavailable_inputs/source_8f6d4bb0811f/DESIGN_AUDIT.json'),{'candidate_hosts':20,'descriptor_columns':FEATURES,'rank':7,
       'centered_rank_with_intercept':int(np.linalg.matrix_rank(centered)),'condition_number':float(sv[0]/sv[-1]),
       'scaling':'divide each delta column by population SD over20 proposed contrasts; no target used',
       'scales':dict(zip(FEATURES,[float(x) for x in scale])),'singular_values':[float(x) for x in sv],
       'same_group_events':len(same),'same_group_valence_column_all_zero':True,
       'valence_changing_candidates':[r['material_id'] for r in rows if r['valence_descriptor_changes']],
       'all_leave_one_host_ranks7':True,'chemistry_and_spin_validated':False,'DFT_performed':False,
       'interpretation':'global design coverage only, not seven causal effects, local rank, predictive fidelity or ground-state proof',
       'QC_survivor_rule':'recompute rank/conditioning after validity classification; do not delete valid metals or counterexamples to improve rank or signs'})
    save(_repo_path('unavailable_inputs/source_1fda93345acf/GEOMETRY_AND_FEATURE_DESIGN_CHECKS.json'),checks)
    # Keep all historical candidates as audit context, not pre-approved choices.
    hist=list(csv.DictReader((_repo_path('unavailable_inputs/source_63c1edc88520/HISTORICAL_SUBSTITUTIONS.csv')).open(encoding='utf-8-sig')))
    alt=[]
    for r in hist:
        if r['is_reference'].lower()=='true':continue
        mid=r['material_id'];old=r['orig_elem'];new=r['replace_elem']
        alt.append({'material_id':mid,'old':old,'new':new,'source':'historical candidate inventory; outcome not used for new selection',
                    'proposed_primary':PRIMARY[mid][:2]==(old,new),'ready_for_submission':False,
                    'hold':'full-element geometry/oxidation/spin/PAW/numeric audit; old optimized gaps not interchangeable with new fixed-geometry SOC pairs'})
    assert len(alt)==55;table('HISTORICAL_55_CANDIDATES_AUDIT_ONLY.csv',alt)
    jobs=[]
    for row in rows:
        mid=row['material_id']
        jobs.append({'plan_id':'SUB20_v1_'+mid+'_A_at_RA','material_id':mid,'state':'reference_fixed_geometry_SOC',
             'new_call_needed_if_gates_pass':not bool(row['reference_reuse_job']),'reuse_job':row['reference_reuse_job'],
             'ready':False,'dependency':'matched pair settings and verified reference identity'})
        jobs.append({'plan_id':'SUB20_v1_'+mid+'_B_at_RA','material_id':mid,'state':'substitution_fixed_geometry_SOC',
             'new_call_needed_if_gates_pass':True,'reuse_job':'','ready':False,
             'dependency':'all8 run035 terminal; audit; chemistry/PAW/spin/numerical gates; ask per-node counts'})
    table('SUB20_V1_LOGICAL_40_STATE_PLAN.csv',jobs)
    save(_repo_path('unavailable_inputs/source_b9a2f7a25817/BUDGET_SCENARIOS.json'),{'main_plan':'20 matched A@RA/B@RA pairs; no default relaxation or band-path chain',
       'logical_static_states':40,'existing_reference_reuse_conditional':2,'baseline_new_static_calls':38,
       'base_4h_node_hours':152,'three_nodes_whole_call_ideal_hours':52,
       'not_a_final_compute_cap':'new PAW or pair settings can invalidate reuse; magnetic/numeric checks and invalid-case repairs cost extra',
       'illustrative_extra_calls_not_authorized':[{'extra_calls':k,'total_calls':38+k,'ideal_3node_hours_at4h':math.ceil((38+k)/3)*4} for k in [8,16]],
       'old_380_call_extrapolation':'superseded, never an approved batch',
       'old_unsubmitted_geometry_calls':'deferred, not deleted or automatically replaced with new target labels'})
    save(_repo_path('unavailable_inputs/source_a24e30f5a0bf/EXECUTION_GATES.json'),{'current_batch_must_finish_first':['12032','12033','12034','12035','12036','12037','12038','12039'],
       'new_submissions':0,'new_production_inputs_generated':False,'automatic_submission':False,
       'per_future_batch':'ask user new node-specific quantities; prior3/2/3 authorization consumed by run035',
       'order':'after all8 terminal and audited, start qualified substitutions; do not require remaining old geometry plan to complete first',
       'stop_or_reclassify':'technical failure, unresolved spin/spectrum or lack of chemical review blocks affected item, not fabrication of gap0',
       'priority':'two already accepted hosts first for classifier/PAW pilot, then qualified20-host pairs; affected magnetic hosts may remain blocked'})
    save(_repo_path('unavailable_inputs/source_d38f52d86d26/ENVIRONMENT.json'),{'python':sys.version,'platform':platform.platform(),'numpy':np.__version__,'pandas':pd.__version__,
       'random_seed':'not applicable: deterministic proposed design; no DFT target-driven selection or model fitting',
       'input_definition':'same seven specified CGCNN table columns and fixed imputation policy; Atomic Volume never V/N'})
    save(_repo_path('unavailable_inputs/source_2630c1d6eb79/INPUT_MANIFEST.json'),[{'path':str(p.relative_to(RUN)).replace('\\','/'),'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted((_repo_path('unavailable_inputs/source_e2da7f3b1259/inputs')).rglob('*')) if p.is_file()])
    print(json.dumps({'hosts':20,'design_rank':7,'condition_number':float(sv[0]/sv[-1]),'valence_changing_candidates':3,
                      'proposed_main_new_calls':38,'submitted':0,'production_inputs_generated':False},ensure_ascii=False))
if __name__=='__main__':main()
