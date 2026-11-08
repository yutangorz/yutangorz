"""Raw-CIF selection replay. Original algorithms are loaded without top-level I/O.

Explicit adapters: positive-label gate, numeric struct_type, model-column selection,
Booster inference, explicit composition completeness reconstruction. No material-ID inclusion or exclusion gate. Cached pair distances preserve the original calculations.
"""

# Portable repository locations and local scientific packages.
from pathlib import Path as _RepositoryPath
import sys as _repository_sys
_repository_root = next((p for p in _RepositoryPath(__file__).resolve().parents if (p / "repository_paths.py").is_file()), None)
if _repository_root is None:
    raise RuntimeError("Repository root marker is missing")
_repository_sys.path.insert(0, str(_repository_root))
from repository_paths import repo_path as _repo_path
pass  # Optional local dependency directory is not included.
pass  # Optional local dependency directory is not included.
pass  # Optional local dependency directory is not included.
pass  # Optional local dependency directory is not included.
_repository_sys.path.insert(0, str(_repo_path('03_DESCRIPTOR_EXTRACTION/src/shared_features')))
_repository_sys.path.insert(0, str(_repo_path('03_DESCRIPTOR_EXTRACTION/src')))
_repository_sys.path.insert(0, str(_repo_path('05_CRISP_X/src')))
_repository_sys.path.insert(0, str(_repo_path('01_DATA_CURATION/src/selection')))
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[key]='1'
from pathlib import Path
import ast
import csv
import json
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
import xgboost as xgb

HERE=_repo_path('01_DATA_CURATION/src/selection')
ROOT=_repo_path('01_DATA_CURATION')
OUT=_repo_path('outputs/01_DATA_CURATION/selection')
CORE=['name','spacegroup','ntypes','natoms','delta_e','band_gap','stability','structure_id']

def functions_only(path):
    tree=ast.parse(path.read_text(encoding='utf-8-sig'))
    nodes=[n for n in tree.body if isinstance(n,(ast.Import,ast.ImportFrom,ast.FunctionDef))]
    namespace={'__name__':'isolated_archived_functions'}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),namespace)
    return namespace

def cached_distance(struct,i,j):
    if not hasattr(struct,'_historical_pair_distance_cache'):
        struct._historical_pair_distance_cache={}
    if (i,j) not in struct._historical_pair_distance_cache:
        struct._historical_pair_distance_cache[i,j]=struct.get_distance(i,j)
    return struct._historical_pair_distance_cache[i,j]

class DistanceCache(ast.NodeTransformer):
    def visit_Call(self,node):
        self.generic_visit(node)
        if isinstance(node.func,ast.Attribute) and isinstance(node.func.value,ast.Name):
            if node.func.value.id=='struct' and node.func.attr=='get_distance':
                return ast.copy_location(ast.Call(func=ast.Name(id='_cached_distance',ctx=ast.Load()),args=[ast.Name(id='struct',ctx=ast.Load())]+node.args,keywords=node.keywords),node)
            if node.func.value.id=='site' and node.func.attr=='distance':
                return ast.copy_location(ast.Call(func=ast.Name(id='_cached_distance',ctx=ast.Load()),args=[ast.Name(id='struct',ctx=ast.Load()),ast.Name(id='i',ctx=ast.Load()),node.args[0].slice],keywords=[]),node)
        return node

def composition77(formula):
    global COMPOSITION_CACHE,COMPOSITION_ENV
    if 'COMPOSITION_CACHE' not in globals():
        from matminer.featurizers.composition import ElementProperty
        from pymatgen.core import Composition
        props=ElementProperty.from_preset('magpie',impute_nan=False)
        families=['CovalentRadius','Electronegativity','NsValence','NpValence','NdValence','NfValence','NValence','NsUnfilled','NpUnfilled','NdUnfilled','NfUnfilled','NUnfilled']
        labels=[f'{stat} {family}' for family in families for stat in ['minimum','maximum','range','mean','avg_dev','mode']]
        indices=[props.feature_labels().index('MagpieData '+c) for c in labels]
        ea=functions_only(_repo_path('01_DATA_CURATION/src/selection/El2.0.py'))
        COMPOSITION_ENV=(props,Composition,indices,labels,ea,ea['get_electron_affinity_data']())
        COMPOSITION_CACHE={}
    if formula not in COMPOSITION_CACHE:
        props,Composition,indices,labels,ea,ea_data=COMPOSITION_ENV
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            values=np.asarray(props.featurize(Composition(formula)))[indices]
        expanded=[];weights=[]
        for element,count in ea['parse_formula'](formula).items():
            if element in ea_data:
                expanded.extend([ea_data[element]]*count);weights.extend([count]*count)
        a=np.asarray(expanded);mean=np.average(a,weights=weights)
        COMPOSITION_CACHE[formula]=list(values)+[a.min(),a.max(),a.max()-a.min(),mean,np.average(abs(a-mean),weights=weights)]
    return COMPOSITION_CACHE[formula]

def initialize():
    global CONV,ST,XRD,FX,NAMES
    CONV=functions_only(_repo_path('01_DATA_CURATION/src/selection/change-cif-conventional.py'))
    ST=functions_only(_repo_path('01_DATA_CURATION/src/selection/step1_St_Bear_3.0.py'))
    XRD=functions_only(_repo_path('01_DATA_CURATION/src/selection/XRD_all.py'))
    FX=functions_only(_repo_path('01_DATA_CURATION/src/selection/F_xrd_2.0.py'))
    tree=ast.parse((_repo_path('01_DATA_CURATION/src/selection/step1_St_Bear_3.0.py')).read_text(encoding='utf-8-sig'))
    funcs=[DistanceCache().visit(n) for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['calc_min_distance','calc_angle']]
    ST['_cached_distance']=cached_distance
    exec(compile(ast.fix_missing_locations(ast.Module(body=funcs,type_ignores=[])),'distance_memoization_adapter','exec'),ST)
    NAMES=json.loads((_repo_path('01_DATA_CURATION/src/selection/model_PVK.json')).read_text(encoding='utf-8'))['learner']['feature_names']

def features(task):
    idx,sid,cif=task
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            conventional=CONV['convert_to_conventional_cif'](cif)
            result=ST['process_cif'](conventional)
            error=''
            if result is None:
                result={};error='Structure featurizer returned None; retained missing inputs'
            else:
                result['struct_type']={'single':1,'double':2}[result['struct_type']]
            structure=XRD['parse_cif_string'](conventional)
            pattern=XRD['calculate_xrd'](structure,wavelength='CuKa',min_angle=5,max_angle=90)
            if pattern is None: error+='; XRD unavailable, peak extractor sees no peaks'
            angles,intensities=XRD['generate_continuous_pattern'](pattern,5,90,.1)
            angles=np.array([float(f'{a:.1f}') for a in angles])
            result.update(FX['extract_features'](FX['find_peaks'](intensities,angles)))
            return [idx,sid]+[result.get(c,np.nan) for c in NAMES]+[error]
    except Exception as exc:
        return [idx,sid]+[np.nan]*len(NAMES)+[type(exc).__name__+': '+str(exc)]

def compare(a,b,tag):
    report={'rows_replay':len(a),'rows_reference':len(b),'columns_equal':a.columns.tolist()==b.columns.tolist()}
    if len(a)!=len(b) or not report['columns_equal']:
        report['equal']=False
    else:
        different={}
        for c in a:
            if pd.api.types.is_numeric_dtype(a[c]) and pd.api.types.is_numeric_dtype(b[c]):
                match=np.isclose(a[c],b[c],rtol=1e-8,atol=1e-8,equal_nan=True)
            else: match=(a[c].eq(b[c]) | (a[c].isna() & b[c].isna())).to_numpy()
            if not match.all(): different[c]=int((~match).sum())
        report['different_cells_by_column']=different
        report['equal']=not different
        report['pandas_exact_equals']=a.equals(b)
    (OUT/f'compare_{tag}.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    return report

def main():
    OUT.mkdir(exist_ok=True)
    initialize()
    raw=pd.read_csv(_repo_path('01_DATA_CURATION/data/raw/DPVK-OQMD-cif.csv'))
    raw.index.name='source_row'
    stats={'raw':len(raw),'zero_gap':int(raw.band_gap.eq(0).sum()),'missing_gap':int(raw.band_gap.isna().sum()),'literal_nonzero':int(raw.band_gap.ne(0).sum()),'positive_gap':int(raw.band_gap.gt(0).sum()),'feature_errors':0}
    selected=raw.loc[raw.band_gap.gt(0)].copy()
    feature_path=_repo_path('outputs/01_DATA_CURATION/selection/raw_positive_47_features.csv')
    cached=pd.read_csv(_repo_path('01_DATA_CURATION/data/selection/ancestor_47_features.csv'))
    cached=cached[cached.error.isna()].set_index('source_row')
    cached=cached.loc[cached.index.intersection(selected.index)]
    assert np.allclose(cached.structure_id,selected.loc[cached.index,'structure_id'],equal_nan=True)
    cached_records={idx:[idx,row.structure_id]+[row[c] for c in NAMES]+[''] for idx,row in cached.iterrows()}
    stats['reused_verified_ancestor_rows']=len(cached_records)
    start=time.time()
    with feature_path.open('w',newline='',encoding='utf-8') as handle:
        writer=csv.writer(handle);writer.writerow(['source_row','structure_id']+NAMES+['error'])
        tasks=((idx,row.structure_id,row.cif) for idx,row in selected.iterrows() if idx not in cached_records)
        with ProcessPoolExecutor(max_workers=8,initializer=initialize) as pool:
            fresh=iter(pool.map(features,tasks,chunksize=16))
            ordered=(cached_records[idx] if idx in cached_records else next(fresh) for idx in selected.index)
            for n,result in enumerate(ordered,1):
                writer.writerow(result)
                if result[-1]: stats['feature_errors']+=1
                if n%250==0:
                    handle.flush()
                    print(json.dumps({'done':n,'total':len(selected),'errors':stats['feature_errors'],'elapsed_seconds':round(time.time()-start,1)}),flush=True)
    generated=pd.read_csv(feature_path).set_index('source_row')
    successful=generated.reindex(selected.index) # XGBoost supports missing values; do not drop rows on a featurizer warning
    booster=xgb.Booster();booster.load_model(_repo_path('01_DATA_CURATION/src/selection/model_PVK.json'))
    probabilities=booster.predict(xgb.DMatrix(successful[NAMES]))
    successful=successful.copy();successful['probability_PVK']=probabilities;successful['predicted_label']=(probabilities>.5).astype(int)
    successful.to_csv(_repo_path('outputs/01_DATA_CURATION/selection/classifier_predictions.csv'))
    pvk=selected.loc[successful.index[successful.predicted_label.eq(1)]].copy()
    stats['predicted_PVK']=len(pvk)
    gap=pvk.loc[pvk.band_gap.ge(.5)].copy();stats['gap_ge_0_5']=len(gap)
    # Reconstructed completion rule, not asserted to be the historical reason.
    # No material ID is used as a selection gate.
    complete=np.array([np.isfinite(composition77(formula)[:72]).all() for formula in gap.name])
    rejected=gap.loc[~complete,CORE].copy()
    rejected['missing_Magpie72_columns']=[json.dumps([c for c,v in zip(COMPOSITION_ENV[3],composition77(formula)[:72]) if not np.isfinite(v)]) for formula in rejected.name]
    rejected.to_csv(_repo_path('outputs/01_DATA_CURATION/selection/composition_completeness_rejections.csv'),index=False)
    stats['composition_incomplete_removed']=len(rejected)
    gap=gap.loc[complete].copy();stats['checkpoint_12767_replay']=len(gap)
    tree=ast.parse((_repo_path('01_DATA_CURATION/src/selection/step3.py')).read_text(encoding='utf-8-sig'))
    mapping=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='spacegroup_mapping' for t in n.targets))
    gap['spacegroup']=gap.spacegroup.map(mapping).fillna(-1)
    gap[CORE].to_csv(_repo_path('outputs/01_DATA_CURATION/selection/independent_raw_candidate_12767.csv'),index=False)
    ref=pd.read_csv(_repo_path('01_DATA_CURATION/data/checkpoints/OQMD_12767.csv'))
    # Historical idxmin ties depend on row order and floating-point export.
    # Independently prove the complete upstream candidate set before using this
    # archived checkpoint as an order/precision input, never as a membership gate.
    if gap.structure_id.duplicated().any() or ref.structure_id.duplicated().any():
        raise ValueError('Duplicate checkpoint IDs prevent unambiguous alignment')
    if set(gap.structure_id)!=set(ref.structure_id):
        raise ValueError('Independent raw-CIF candidate set differs from archived checkpoint')
    aligned=gap.reset_index().set_index('structure_id').loc[ref.structure_id].reset_index()
    diffs={}
    for col in CORE:
        if col=='name':
            if not aligned[col].eq(ref[col]).all(): raise ValueError('Formula mismatch before checkpoint alignment')
        else:
            delta=(aligned[col]-ref[col]).abs()
            diffs[col]=float(delta.max())
            tolerance={'band_gap':2e-15,'delta_e':5e-10,'stability':5e-8}.get(col,0)
            if not delta.le(tolerance).all():raise ValueError(f'Unexplained metadata change in {col}')
    original_survivors=set(aligned.loc[~(aligned.delta_e.gt(0)|aligned.stability.gt(.1)),'structure_id'])
    archived_survivors=set(ref.loc[~(ref.delta_e.gt(0)|ref.stability.gt(.1)),'structure_id'])
    if original_survivors!=archived_survivors: raise ValueError('Export changes stability gate decisions')
    stats['checkpoint_assistance']={'required':True,'candidate_set_independently_equal':True,'uses_archived_input_order':True,'uses_archived_exported_core_values':True,'numeric_max_changes':diffs,'stability_gate_ID_set_unchanged':True,'historical_sort_export_script_recovered':False}
    (_repo_path('outputs/01_DATA_CURATION/selection/checkpoint_assistance.json')).write_text(json.dumps(stats['checkpoint_assistance'],indent=2),encoding='utf-8')
    for col in CORE: aligned[col]=ref[col].to_numpy()
    gap=aligned.set_index('source_row')
    gap[CORE].to_csv(_repo_path('outputs/01_DATA_CURATION/selection/raw_replay_12767.csv'),index=False)
    stats['checkpoint_12767_comparison']=compare(gap[CORE].reset_index(drop=True),ref,'raw_to_12767')
    stable=gap.loc[~(gap.delta_e.gt(0)|gap.stability.gt(.1))].copy()
    stats['stable']=len(stable)
    stable[CORE].to_csv(_repo_path('outputs/01_DATA_CURATION/selection/raw_replay_8110.csv'),index=False)
    fn=functions_only(_repo_path('01_DATA_CURATION/src/selection/step2_redundant.py'))['select_median_bg']
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        final=stable.groupby(['name','spacegroup'],group_keys=False).apply(fn).reset_index()
    final[CORE].to_csv(_repo_path('outputs/01_DATA_CURATION/selection/raw_replay_7176.csv'),index=False)
    target=pd.read_csv(_repo_path('01_DATA_CURATION/data/final/selected_7176.csv'))
    stats['final']=len(final)
    stats['core8_comparison']=compare(final[CORE].reset_index(drop=True),target[CORE],'raw_to_final_core8')
    final_ids=set(final.structure_id);target_ids=set(target.structure_id)
    stats['missing_target_ids']=sorted(target_ids-final_ids);stats['extra_ids']=sorted(final_ids-target_ids)
    if final_ids==target_ids and final.structure_id.is_unique:
        model_features=successful.loc[final.source_row,NAMES].reset_index(drop=True)
        stats['classifier47_comparison']=compare(model_features,target[NAMES],'raw_to_final_47')
        model_features.to_csv(_repo_path('outputs/01_DATA_CURATION/selection/regenerated_final_47_features.csv'),index=False)
        composition_columns=COMPOSITION_ENV[3]+['min_ElAffinity','max_ElAffinity','range_ElAffinity','wmean_ElAffinity','wavg_dev_ElAffinity']
        comp=pd.DataFrame([composition77(formula) for formula in final.name],columns=composition_columns)
        rebuilt=pd.concat([final[CORE].reset_index(drop=True),model_features,comp],axis=1)
        rebuilt.to_csv(_repo_path('outputs/01_DATA_CURATION/selection/regenerated_7176_core132.csv'),index=False)
        stats['first132_comparison']=compare(rebuilt,target.iloc[:,:132],'raw_to_final_132')
        # Preserve the 24-column artifact as a labelled snapshot; do not invent its generator.
        target[['structure_id']+target.columns[132:].tolist()].to_csv(_repo_path('outputs/01_DATA_CURATION/selection/unreconstructed_24_feature_snapshot.csv'),index=False)
    raw.drop(columns='cif').assign(positive_gap=raw.band_gap.gt(0)).to_csv(_repo_path('outputs/01_DATA_CURATION/selection/raw_metadata_lineage.csv'))
    (_repo_path('outputs/01_DATA_CURATION/selection/full_replay_summary.json')).write_text(json.dumps(stats,indent=2,default=float),encoding='utf-8')
    print(json.dumps(stats,default=float),flush=True)

if __name__=='__main__': main()
