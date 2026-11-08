"""Authorized stage2 ONLY. Exact frozen sampler method, no model/fitter/statistics load."""

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
_repository_sys.path.insert(0, str(_repo_path('06_EXTERNAL_TRANSFER/src/gaussian_sampling')))
from pathlib import Path
import ast,json,hashlib,csv,importlib.metadata as meta,time,sys,traceback
import numpy as np,pandas as pd
R=_repo_path('.');P=_repo_path('06_EXTERNAL_TRANSFER/workflows/common_ridge');B=_repo_path('06_EXTERNAL_TRANSFER/workflows/execution_binding');O=_repo_path('06_EXTERNAL_TRANSFER/workflows/gaussian_sampling')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def table(name,rows,cols=None):pd.DataFrame(rows,columns=cols).to_csv(O/name,index=False,encoding='utf-8-sig')
def text(name,s):(O/name).write_text(s,encoding='utf8')
assert not (_repo_path('06_EXTERNAL_TRANSFER/workflows/gaussian_sampling/ATTEMPTS.jsonl')).exists(),'No retry/resume; use author-reviewed new version'
rows=[];failures=[];hashes=[];calls=0;started=time.time();blocked=[];state='HOLD';checks=[]
def event(x):
 with (_repo_path('06_EXTERNAL_TRANSFER/workflows/gaussian_sampling/ATTEMPTS.jsonl')).open('a',encoding='utf8') as f:f.write(json.dumps(x,ensure_ascii=False)+'\n');f.flush();__import__('os').fsync(f.fileno())
event(dict(action='START',authority='User explicitly approved stage2 real Gaussian standalone preflight; not stage3',formal_execution_authorized=False,time=time.time()))
try:
 report=_repo_path('06_EXTERNAL_TRANSFER/workflows/source_identity/SHA_RECHECK_REPORT.md')
 assert sha(report)=='aa2e33c5a9e833d9b5dd64dd7750023f714c958ed221965dadcbc2971d2bf0e3'
 binding=json.loads((_repo_path('06_EXTERNAL_TRANSFER/workflows/execution_binding/EXECUTION_APPROVAL_BINDINGS.json')).read_text(encoding='utf8'))
 for k,v in binding['bindings'].items():
  assert sha(v['path'])==v['sha256'],k;checks.append(dict(path=v['path'],sha256=v['sha256']))
 manifest=pd.read_csv(binding['bindings']['data_manifest_sha256']['path'])
 assert len(manifest)==241
 for x in manifest.to_dict('records'):assert sha(x['path'])==x['sha256'],x['path']
 assert all(meta.version(k)==v for k,v in binding['environment'].items())
 cfg=json.loads((_repo_path('06_EXTERNAL_TRANSFER/data/common_ridge/EXPERIMENT_CONFIG_LOCKED.json')).read_text(encoding='utf8'));F=cfg['features']
 assert sha(cfg['ET_model_path'])==cfg['ET_model_sha256']
 # Resolve and verify loaded official source path before invoking its sampler.
 import lime.lime_tabular as lt
 from lime.lime_tabular import LimeTabularExplainer
 installed=Path(lt.__file__).resolve();expect=manifest[manifest.path.map(lambda s:Path(s).resolve()==installed)]
 assert len(expect)==1 and sha(installed)==expect.iloc[0].sha256
 # Hard fail if an accidental real prediction/model load/Ridge fit is attempted.
 import joblib
 from sklearn.linear_model import Ridge
 from sklearn.ensemble._forest import ForestRegressor
 def forbidden(*a,**k):blocked.append('forbidden scientific call');raise RuntimeError('ET/model load/Ridge fit prohibited in stage2')
 joblib.load=forbidden;Ridge.fit=forbidden;ForestRegressor.predict=forbidden
 train=np.load(_repo_path('06_EXTERNAL_TRANSFER/workflows/common_ridge/training5714.npy'),allow_pickle=False);scale=np.load(_repo_path('06_EXTERNAL_TRANSFER/workflows/common_ridge/training_SD.npy'),allow_pickle=False)
 assert train.shape==(5714,9) and np.all(scale>0)
 with np.load(_repo_path('06_EXTERNAL_TRANSFER/workflows/common_ridge/EXACT_ANCHORS38.npz'),allow_pickle=False) as z:anchors={str(ID):v.copy() for ID,v in zip(z['IDs'],z['x0'])}
 assert len(anchors)==38
 cases=json.loads((_repo_path('06_EXTERNAL_TRANSFER/data/common_ridge/CASE_INPUTS76.json')).read_text(encoding='utf8'));plans_sha=sha(_repo_path('06_EXTERNAL_TRANSFER/data/common_ridge/SELECTED_FIT_INDICES760.csv'))
 factory=Path(binding['bindings']['factory_sha256']['path']);tree=ast.parse(factory.read_text(encoding='utf8'))
 methods=[n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='sample'];assert len(methods)==1
 # Compile only exact sample method; never execute factory() (which would load ET).
 node=methods[0];module=ast.Module(body=[node],type_ignores=[]);ast.fix_missing_locations(module)
 namespace=dict(LimeTabularExplainer=LimeTabularExplainer,train=train,F=F,hashlib=hashlib)
 exec(compile(module,str(factory),'exec'),namespace)
 sampler=type('FrozenSamplerOnly',(),dict(sample=namespace['sample']))()
 (_repo_path('06_EXTERNAL_TRANSFER/workflows/gaussian_sampling/candidates')).mkdir(exist_ok=False)
 dump(_repo_path('06_EXTERNAL_TRANSFER/workflows/gaussian_sampling/RUNTIME_IDENTITY.json'),dict(bound_files=checks,actual_sampler_source=str(installed),actual_sampler_sha256=sha(installed),factory_ast_sample_only=True,actual_factory_sha256=sha(factory),model_deserialized=False,environment=binding['environment'],feature_order=F,selected_indices_sha256=plans_sha,protocol_unchanged=True))
 for ID in sorted(anchors):
  seed=int(hashlib.sha256(('run022|'+ID).encode()).hexdigest()[:8],16)%(2**31-1)
  row=dict(ID=ID,seed=seed,planned_shape='512x9',status='STARTED',finite=False,anchor_exact=False,anchor_bytes_equal=False,exact_collisions=0,tolerance_collisions=0)
  event(dict(action='GENERATION_RESERVED',ID=ID,seed=seed,attempt=1,rows=512))
  try:
   calls+=1;gx=sampler.sample(ID,anchors[ID])
   dest=_repo_path('06_EXTERNAL_TRANSFER/workflows/gaussian_sampling/candidates')/f'{ID}.npy';np.save(dest,gx,allow_pickle=False) # Preserve original output BEFORE validity tests.
   hashes.append(dict(ID=ID,path=str(dest),sha256=sha(dest),dtype=gx.dtype.str,shape=str(gx.shape),index_definition='original rows0..511; no reorder; anchor0'))
   row.update(dtype=gx.dtype.str,rows=gx.shape[0],columns=gx.shape[1] if gx.ndim==2 else None,candidate_sha256=sha(dest))
   if gx.shape!=(512,9):raise RuntimeError('SHAPE_MISMATCH')
   invalid=np.argwhere(~np.isfinite(gx));row['finite']=len(invalid)==0
   if len(invalid):
    for i,j in invalid:failures.append(dict(ID=ID,type='NONFINITE',candidate=int(i),feature=int(j),test_seed=None,test_candidate=None))
    raise RuntimeError('NONFINITE')
   row['anchor_exact']=bool(np.array_equal(gx[0],anchors[ID]));row['anchor_bytes_equal']=gx[0].dtype==anchors[ID].dtype and gx[0].tobytes()==anchors[ID].tobytes()
   if not row['anchor_exact'] or not row['anchor_bytes_equal']:
    failures.append(dict(ID=ID,type='ANCHOR_MISMATCH',candidate=0,feature=None,test_seed=None,test_candidate=None));raise RuntimeError('ANCHOR_MISMATCH')
   for c in [c for c in cases if c['ID']==ID]:
    pool=pd.read_csv(c['pool_path']);xx=pool[F].to_numpy()
    for idx in c['test']:
     dist=np.max(np.abs(gx-xx[idx]),axis=1);exact=np.flatnonzero(np.all(gx==xx[idx],axis=1));tol=np.flatnonzero(dist<1e-12)
     row['exact_collisions']+=len(exact);row['tolerance_collisions']+=len(tol)
     for typ,indices in [('EXACT_TEST_COLLISION',exact),('LT_1e12_TEST_COLLISION',tol)]:
      for j in indices:failures.append(dict(ID=ID,type=typ,candidate=int(j),feature=None,test_seed=c['seed'],test_candidate=int(idx)))
   if row['tolerance_collisions'] or row['exact_collisions']:raise RuntimeError('TEST_COLLISION')
   row['status']='PASS';event(dict(action='PREFLIGHT_COMPLETE',ID=ID,status='PASS',sha256=sha(dest)))
  except BaseException as e:
   row['status']='FAIL';row['error']=repr(e)
   if not any(f['ID']==ID for f in failures):failures.append(dict(ID=ID,type=repr(e),candidate=None,feature=None,test_seed=None,test_candidate=None))
   event(dict(action='STOP_NO_RETRY',ID=ID,error=repr(e)));rows.append(row);break
  rows.append(row)
  table('GAUSSIAN_PREFLIGHT_BY_MATERIAL.csv',rows);table('GAUSSIAN_CANDIDATE_SHA256.csv',hashes)
 assert sha(_repo_path('06_EXTERNAL_TRANSFER/data/common_ridge/SELECTED_FIT_INDICES760.csv'))==plans_sha
 for v in binding['bindings'].values():assert sha(v['path'])==v['sha256'],'Postrun bound file change'
 assert not blocked
 state='PASS' if len(rows)==38 and all(x['status']=='PASS' for x in rows) else 'HOLD_FAILED_PREFLIGHT'
except BaseException as e:
 failures.append(dict(ID='GLOBAL',type=repr(e),candidate=None,feature=None,test_seed=None,test_candidate=None));event(dict(action='GLOBAL_STOP',error=repr(e)));state='HOLD'
 text('ERROR_TRACE.txt',traceback.format_exc())
finally:
 done={x['ID'] for x in rows}
 if 'anchors' in globals():
  rows.extend(dict(ID=ID,status='NOT_ATTEMPTED_AFTER_STOP',planned_shape='512x9') for ID in sorted(anchors) if ID not in done)
 table('GAUSSIAN_PREFLIGHT_BY_MATERIAL.csv',rows)
 table('GAUSSIAN_CANDIDATE_SHA256.csv',hashes,['ID','path','sha256','dtype','shape','index_definition'])
 table('PREFLIGHT_FAILURES.csv',failures,['ID','type','candidate','feature','test_seed','test_candidate'])
 n=sum(x['status']=='PASS' for x in rows)
 text('GAUSSIAN_PREFLIGHT_REPORT.md',f'''# 真实Gaussian独立预检
状态：{state}；38计划母体，{calls}次生成尝试，{n}个PASS，{len(failures)}条异常记录。
第一阶段报告SHA核对，七项绑定和241输入及冻结环境在本轮再验；实际载入LIME源码路径/SHA与清单一致。仅从绑定factory提取原sample方法AST，未调用factory或加载ET；运行保护阻断ET/model load/Ridge.fit。
全部生成候选在检查前以原始.npy保存，dtype与索引见逐材料表。有限性检查覆盖全部512×9；首行np.array_equal并补查字节一致；碰撞检查覆盖**全部512行（含anchor）**对两个结构seed的全部48个固定test原始九维向量，精确==和原规定max(abs差)<1e-12分别登记。未删除、过滤或重采样。
PREFLIGHT_FAILURES.csv记录{len(failures)}项（0时为仅表头空表）；失败即停且保留结果，不继续未尝试母体，不自动重试。
真实Gaussian行尝试预算{calls*512}/19456。ET预测0、模型加载0、Ridge0、bootstrap0、新结构/GPU/DFT/训练0。formal execution_authorized=false，scientific_closeout_approved=false。此PASS如成立仅为第二阶段技术预检，不是正式实验验收。
后续必须复用这些候选字节，不重新生成；已有生产driver仍会调用sample，正式使用前需在新版本中改为读取本候选清单并绑定其SHA，且单独获得第三阶段授权。未因本预检通过自动放行ET。
''')
 text('CHECKPOINT.md',f'第二阶段独立预检完成：{state}，PASS {n}/38；失败记录{len(failures)}。候选与日志保留，不重采样。正式ET/Ridge/bootstrap执行授权false，当前停止等待作者审核。\n')
 text('NEXT_STEP_AND_AUTHOR_APPROVAL.md','请审核本真实预检报告和逐材料/候选SHA清单。第三阶段需独立明确批准，预算仍为14411响应请求、A380/B380拟合；不能重用当前预检授权。正式driver须改为加载已保存候选并重新冻结绑定，禁止再次采样；任何失败需作者决定，不自动重试。\n')
 table('SHA256.csv',[dict(path=str(p),sha256=sha(p)) for p in sorted(O.rglob('*')) if p.is_file() and p.name not in ['SHA256.csv','console.log']])
 print(state,'PASS',n,'/38','failures',len(failures),'ET/Ridge/bootstrap=0')
 if state!='PASS':raise SystemExit(1)
