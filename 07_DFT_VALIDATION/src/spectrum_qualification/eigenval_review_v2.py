
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
_repository_sys.path.insert(0, str(_repo_path('07_DFT_VALIDATION/src/spectrum_qualification')))
from pathlib import Path
import json,gzip,hashlib,math
ROOT=_repo_path('07_DFT_VALIDATION/inputs/FGM/DFT_val')
OUT=_repo_path('07_DFT_VALIDATION/src/spectrum_qualification')
def parse(path):
    before=path.stat();h=hashlib.sha256();points=[];largest=0.;weighted=0.
    with path.open('rb') as stream:
        def line():
            b=next(stream);h.update(b);return b.decode('ascii','replace')
        head=[line() for _ in range(6)];ne,nk,nb=map(int,head[-1].split())
        for k in range(nk):
            a=line()
            while not a.strip():a=line()
            kp=list(map(float,a.split()));bands=[]
            for b in range(nb):
                v=list(map(float,line().split()))
                if len(v)!=3 or int(v[0])!=b+1:raise ValueError('requires ordered scalar ISPIN=1 EIGENVAL')
                bands.append(v[1:]);largest=max(largest,v[2]);weighted+=kp[3]*v[2]
            points.append((kp,bands))
        for b in stream:h.update(b)
    full=1. if largest<=1.001 else 2.
    if largest>2.001:raise ValueError('occupation range unsupported')
    edges=[];vbm=-math.inf;cbm=math.inf;fixedv=-math.inf;fixedc=math.inf;fractional=0
    for k,(kp,bands) in enumerate(points,1):
        filled=[e for e,o in bands if o>full*.5];empty=[e for e,o in bands if o<=full*.5]
        v=max(filled) if filled else None;c=min(empty) if empty else None
        fractional+=sum(.001<o/full<.999 for e,o in bands)
        if v is not None:vbm=max(vbm,v)
        if c is not None:cbm=min(cbm,c)
        if ne%2==0 and 0<ne//2<nb:
            fixedv=max(fixedv,bands[ne//2-1][0]);fixedc=min(fixedc,bands[ne//2][0])
        edges.append([k]+kp+[v,c])
    valid=math.isfinite(vbm) and math.isfinite(cbm)
    return {'nelect':ne,'nkpoints':nk,'nbands':nb,'maximum_occupation':largest,'inferred_full_occupation':full,
        'raw_gap_occupation_threshold_eV':cbm-vbm if valid else None,'gap_occupation_threshold_eV':max(0.,cbm-vbm) if valid else None,
        'raw_gap_fixed_filling_eV':fixedc-fixedv if math.isfinite(fixedv+fixedc) else None,'fractional_entries_normalized':fractional,
        'weighted_occupation_sum':weighted,'edges':edges,'edges_columns':['index1','kx','ky','kz','weight','vbm','cbm'],
        'sha256':h.hexdigest(),'bytes':before.st_size,'unchanged_during_read':before.st_size==path.stat().st_size and before.st_mtime_ns==path.stat().st_mtime_ns}
if __name__=='__main__':
    with gzip.open(_repo_path('07_DFT_VALIDATION/src/spectrum_qualification/stage_audits.jsonl.gz'),'rt') as f:stages=list(map(json.loads,f))
    dest=_repo_path('07_DFT_VALIDATION/src/spectrum_qualification/eigenval_crosscheck_v2.jsonl.gz')
    with gzip.open(dest,'xt',encoding='utf-8') as f:
        for s in stages:
            if s['phase'] not in ['2_scf','3_band']:continue
            try:
                assert s['actual_parameters_XML']['ISPIN']=='1' and s['actual_parameters_XML']['LSORBIT']=='F'
                row={'stage':s['relative_stage'],**parse(ROOT/s['relative_stage']/'EIGENVAL')}
            except Exception as exc:row={'stage':s['relative_stage'],'error':str(exc)}
            f.write(json.dumps(row)+'\n')
    print(json.dumps({'file':dest.name,'bytes':dest.stat().st_size,'sha256':hashlib.sha256(dest.read_bytes()).hexdigest(),'source_writes':0}))
