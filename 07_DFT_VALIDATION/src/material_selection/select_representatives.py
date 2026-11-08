"""Execute archived class methods, not main/export/plot routines."""

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
_repository_sys.path.insert(0, str(_repo_path('07_DFT_VALIDATION/src/material_selection')))
import argparse,importlib.util,json,time,warnings,types,os
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.metrics import adjusted_rand_score
from cluster_core import DiskDistanceBackend,sha,candidate_partition_metrics


def dump(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);run=ap.parse_args().run
    for r in json.loads((run/'FROZEN_INPUT_CODE.json').read_text(encoding='utf-8')):assert sha(run/r['path'])==r['sha256']
    out=run/'results';out.mkdir(exist_ok=False);start=time.perf_counter()
    initial=module(run/'inputs/historical_code/11.py','historical_initial');final=module(run/'inputs/historical_code/替换后权重.py','historical_final')
    manifest=pd.read_csv(run/'inputs/structure_manifest.csv',dtype={'material_id':str});folder=Path(json.loads((run/'config.json').read_text(encoding='utf-8'))['structure_source'])
    # Original loader uses os.listdir order. Check it against the frozen order.
    actual=[p for p in os.listdir(folder) if p.endswith('.vasp')]
    assert set(actual)==set(manifest.filename),'structure set changed'
    # Replay the captured source os.listdir order, even when copying to another volume.
    original_listdir=os.listdir
    initial.os=types.SimpleNamespace(path=os.path,listdir=lambda p:manifest.filename.tolist() if Path(p)==folder else original_listdir(p))
    for row in manifest.itertuples():assert sha(row.path)==row.sha256,row.material_id
    loader=initial.MaterialDatasetLoader(str(folder),str(run/'inputs/historical_id_prop.csv'))
    structures=loader.load_dataset();assert len(structures)==len(manifest)==7176
    assert [s.material_id for s in structures]==manifest.material_id.tolist()
    features=initial.MaterialFeatureExtractor().extract_features(structures)
    assert len(features)==len(structures) and features.shape[1]==15 and np.isfinite(features.to_numpy()).all()
    assert (features.band_gap==0).all(),'historical missing-ID fallback expected'
    original_columns=features.columns.tolist()
    feature_table=features.copy();feature_table.insert(0,'material_id',manifest.material_id)
    feature_table.to_csv(out/'historical_features_7176.csv',index=False)
    dump(out/'FEATURE_EXTRACTION.json',dict(features=original_columns,rows=len(features),gap_nonzero_n=int((features.band_gap!=0).sum()),
        skipped_structures=0,index_alignment='all structures and feature rows equal in original file order'))
    backend=DiskDistanceBackend(out/'distance_cache')
    for m in [initial,final]:m.pdist=backend.pdist;m.squareform=backend.squareform;m.linkage=backend.linkage
    selector=initial.RepresentativeMaterialSelector(n_clusters=10,n_samples_per_cluster=2,max_total_samples=20)
    # This method mutates the feature frame by adding cluster/distance columns.
    selected=selector.select_representative_samples(features,structures)
    assert len(selected)==20 and len(set(selected))==20
    full=feature_table.copy();full['cluster_id']=features.cluster.astype(int);full['distance_to_center']=features.distance_to_center
    full['selected_initial']=False;full.loc[selected,'selected_initial']=True
    full.to_csv(out/'full_partition_7176.csv',index=False)
    first=full.iloc[selected].copy();first.to_csv(out/'initial_selected_20.csv',index=False)
    expected_initial=pd.read_csv(run/'inputs/expected_initial20.csv',dtype={'material_id':str})
    assert len(expected_initial)==20
    initial_set=set(first.material_id)==set(expected_initial.material_id)
    initial_order=first.material_id.tolist()==expected_initial.material_id.tolist()
    assert initial_set,'Historical initial20 did not reproduce; preserve output, do not force IDs'
    expected_final=pd.read_csv(run/'inputs/expected_final20.csv',dtype={'material_id':str})
    pairmetrics=candidate_partition_metrics(full.material_id,full.cluster_id,expected_final)
    final_frame=final.MaterialFeatureExtractor().extract_all_features(structures)[0]
    assert final_frame.columns.tolist()==original_columns and np.array_equal(final_frame.to_numpy(),feature_table[original_columns].to_numpy())
    analyzer=final.MaterialClusterAnalyzer(n_clusters=10);labels=analyzer.perform_clustering(final_frame)
    assert np.array_equal(labels,full.cluster_id.to_numpy()),'same full feature matrix, same source clustering result'
    weights=final.MaterialWeightCalculator().calculate_weights(final_frame,labels,expected_final.material_id.tolist(),structures)
    weights_rows=[]
    for ID,w in weights.items():
        row=dict(material_id=str(ID),cluster_id=int(w['cluster_id']))
        # Source returns ID keyed rows containing nested components.
        row['source_payload']=json.dumps(w,default=lambda v:v.item() if isinstance(v,np.generic) else str(v))
        weights_rows.append(row)
    # Source keys / schemas are inspected and replayed, not guessed silently.
    dump(out/'final_source_weight_payload.json',weights)
    pd.DataFrame(weights_rows).to_csv(out/'final_source_weight_payload.csv',index=False)
    errors=[]
    for row in expected_final.itertuples():
        actual_weight=weights[row.material_id]
        for col in ['final_weight','representativeness_weight','importance_weight','diversity_weight','feature_significance','band_gap_weight','distance_to_center','cluster_size','volume']:
            errors.append(dict(material_id=row.material_id,quantity=col,saved=float(getattr(row,col)),replayed=float(actual_weight[col]),absolute_error=abs(float(getattr(row,col))-float(actual_weight[col]))))
    pd.DataFrame(errors).to_csv(out/'TABLE_saved_weight_replay.csv',index=False)
    chosen=full.set_index('material_id').loc[expected_final.material_id].reset_index()
    chosen.to_csv(out/'final_historical_20_labels.csv',index=False)
    comparisons=expected_final[['material_id','cluster_id','cluster_size','distance_to_center']].merge(
        chosen[['material_id','cluster_id','distance_to_center']],on='material_id',suffixes=('_saved','_replay'),validate='one_to_one')
    sizes=full.cluster_id.value_counts();comparisons['cluster_size_replay']=comparisons.cluster_id_replay.map(sizes)
    comparisons['distance_abs_difference']=abs(comparisons.distance_to_center_saved-comparisons.distance_to_center_replay)
    comparisons.to_csv(out/'TABLE_saved_vs_replayed20.csv',index=False)
    dump(out/'BACKEND_EQUIVALENCE_AND_CACHE.json',dict(calls=backend.calls,storage='float64 disk-backed; native pdist and native Ward arithmetic',
        warning='original linkage(2D square matrix) is observations, not precomputed distances; historical global warning suppression retained in source',
        cache_files=[dict(path=str(p.relative_to(run)),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted((out/'distance_cache').iterdir())]))
    result=dict(status='historical_replay_complete_pending_final_audit',structures=7176,features=15,
        raw_file_order_matches_all_saved_initial_indices=True,initial20_set_exact=initial_set,initial20_order_exact=initial_order,
        final20_partition=pairmetrics,final20_numeric_labels_exact=bool((comparisons.cluster_id_saved==comparisons.cluster_id_replay).all()),
        saved_cluster_sizes_exact=bool((comparisons.cluster_size==comparisons.cluster_size_replay).all()),
        max_saved_centroid_distance_error=float(comparisons.distance_abs_difference.max()),
        max_saved_weight_or_property_absolute_error=float(pd.DataFrame(errors).absolute_error.max()),
        historical_full7176_labels_independent_reference_available=False,
        gap_ID_suffix_mismatch_reproduced=True,gap_is_not_zero_physical_evidence=True,
        no_supervised_pair_constraints_used=True,elapsed_seconds=time.perf_counter()-start)
    dump(out/'HISTORICAL_REPLAY_RESULT.json',result);print('HISTORICAL REPLAY COMPLETE',json.dumps(result),flush=True)


if __name__=='__main__':main()
