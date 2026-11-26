"""Verify fixed ET/LIME input identity without loading or querying a model."""
from pathlib import Path
import argparse,csv,hashlib,json
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
FEATURES=['period','electronegativity','covalent_radius','valence_electrons','ionization_energy','electron_affinity','atomic_volume','bond_length','bond_angle']
ATOL=1e-12
MODEL_SHA='1e3c7c3370b5fcc41b25db1eb20ff483c877c190d23d3c83b2035e3fd21784c6'

def rows(p):
    with p.open(encoding='utf-8-sig',newline='') as stream:return list(csv.DictReader(stream))

def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def equivalent(a,b):
    return bool(a.shape==b.shape and np.array_equal(np.isnan(a),np.isnan(b))
        and np.array_equal(np.isposinf(a),np.isposinf(b)) and np.array_equal(np.isneginf(a),np.isneginf(b))
        and np.allclose(a,b,rtol=0,atol=ATOL,equal_nan=True))

def check():
    data=rows(ROOT/'03_DESCRIPTOR_EXTRACTION/data/model_input_7176.csv')
    folds=rows(ROOT/'03_DESCRIPTOR_EXTRACTION/data/outer_fold_manifest.csv')
    refs=rows(ROOT/'05_CRISP_X/data/final_response/shap_baseline/per_material_nine_attributions.csv')
    historical=json.loads((ROOT/'04_ET_XGBOOST_INTERPRETATION/data/native_lime/HISTORICAL_IDENTITY_AUDIT.json').read_text(encoding='utf-8'))
    by={r['ID']:r for r in data};foldby={r['ID']:r['outer_fold'] for r in folds}
    unique=len(data)==len(by)==len(folds)==len(foldby)==7176 and len(refs)==len({r['ID'] for r in refs})==1462
    ids_ok=set(by)==set(foldby) and {r['ID'] for r in refs}=={i for i in by if foldby.get(i)=='0'}
    order_ok=[r['ID'] for r in refs]==sorted(i for i in by if foldby.get(i)=='0')
    features_ok=FEATURES==historical['features_computational_order'] and all([k for k in table[0] if k in FEATURES]==FEATURES for table in [data,refs])
    if not (unique and ids_ok and order_ok and features_ok):
        return dict(passed=False,status='FAIL',IDs_exact=ids_ok,row_order_exact=order_ok,features_computational_order=FEATURES,feature_order_exact=features_ok,reason='ID, ordering or schema mismatch')
    current=np.array([[float(by[r['ID']][f]) for f in FEATURES] for r in refs])
    reference=np.array([[float(r[f]) for f in FEATURES] for r in refs])
    exact=bool(np.array_equal(current,reference,equal_nan=True));numeric=equivalent(current,reference)
    float32=bool(np.array_equal(current.astype(np.float32),reference.astype(np.float32),equal_nan=True))
    # This is the same training row order selected by the LIME driver.
    training=np.array([[float(r[f]) for f in FEATURES] for r in data if foldby[r['ID']]!='0'])
    folder=ROOT/'06_EXTERNAL_TRANSFER/data/common_ridge'
    saved_training=np.load(folder/'training5714.npy',allow_pickle=False)
    saved_sd=np.load(folder/'training_SD.npy',allow_pickle=False)
    train_ok=equivalent(training,saved_training)
    mean_ok=equivalent(training.mean(axis=0),saved_training.mean(axis=0))
    scale_ok=equivalent(training.std(axis=0),saved_sd)
    model_sha=digest(ROOT/'05_CRISP_X/models/property_model.joblib')
    passed=unique and ids_ok and order_ok and features_ok and numeric and float32 and train_ok and mean_ok and scale_ok and model_sha==MODEL_SHA
    finite=np.isfinite(current)&np.isfinite(reference)
    difference=float(np.max(np.abs(current[finite]-reference[finite]))) if finite.any() else 0.0
    return dict(passed=passed,status=('PASS_WITH_NUMERICAL_TOLERANCE' if passed and not exact else 'PASS' if passed else 'FAIL'),scope='fixed inputs and standardization only; not an explanation-equivalence or fidelity test',EXACT_EQUALITY='PASS' if exact else 'FAIL',NUMERICAL_EQUIVALENCE='PASS' if numeric else 'FAIL',rtol=0,atol=ATOL,equal_nan=True,IDs_exact=ids_ok,row_order_exact=order_ok,n=len(refs),features_computational_order=FEATURES,feature_order_exact=features_ok,nonfinite_masks_equal=bool(np.array_equal(np.isfinite(current),np.isfinite(reference))),float32_features_exact=float32,training_array_consistent=train_ok,training_mean_consistent=mean_ok,training_population_SD_consistent=scale_ok,max_abs_feature_difference=difference,model_sha256=model_sha,new_model_queries=0,new_model_fits=0,new_LIME_fits=0)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path);args=parser.parse_args();result=check()
    if args.output:
        if args.output.exists():raise SystemExit('Refusing to overwrite an existing identity receipt')
        args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2));raise SystemExit(0 if result['passed'] else 1)

if __name__=='__main__':main()
