"""Independent streaming acceptance evidence; reads only seven A10 directories.
Never launches LOBSTER, VASP or scheduler actions; no remote writes.
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
_repository_sys.path.insert(0, str(_repo_path('08_COHP_ANALYSIS/src/projection_checks')))
from pathlib import Path
import json,hashlib,math,re,time,base64,zlib
R=_repo_path('07_DFT_VALIDATION/inputs/FGM/nature_perovskite_revision/batches/cohp_A10_run105_20261030')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def geom(p):
 s=p.read_text().splitlines();scale=float(s[1]);l=[[float(v)*scale for v in x.split()] for x in s[2:5]];els=s[5].split();counts=list(map(int,s[6].split()));start=8 if not s[7].lower().startswith('s') else 9;coords=[[float(v) for v in x.split()[:3]] for x in s[start:start+sum(counts)]]
 if not s[start-1].lower().startswith('d'):raise ValueError('Coordinates not direct; explicit conversion required')
 return l,coords,[e for e,n in zip(els,counts) for _ in range(n)]
results=[]
for spec in json.loads((_repo_path('07_DFT_VALIDATION/inputs/FGM/nature_perovskite_revision/batches/cohp_A10_run105_20261030/BATCH.json')).read_text()):
 state=spec['state'];
 if state not in {'6313886_LENGTH_07','6313886_ANGLE_05','6313886_ANGLE_01','6313886_ANGLE_10'}:continue
 d=R/state;o=d/'OUTPUT';a=dict(state=state,status='HOLD')
 try:
  t=json.loads((d/'TASK.json').read_text());q=json.loads((d/'TECHNICAL_SCIENTIFIC_QC.json').read_text());term=json.loads((d/'TERMINAL.json').read_text());assert term['exit_code']==0 and not term['timeout'];assert q['technical_qc_status']=='TECHNICAL_PASS'
  # Wavefunction, CHGCAR and XML were already hashed by the terminal worker.
  # Reuse those identity checks plus unchanged-before-QC metadata, without rescanning 5GB/state.
  large={n:dict(worker_SHA_match=q['input_checks'].get(n),expected_SHA=t['source_files'][n]['sha256'],size=(o/n).stat().st_size,mtime_not_after_QC=(o/n).stat().st_mtime_ns <= (d/'TECHNICAL_SCIENTIFIC_QC.json').stat().st_mtime_ns) for n in ['WAVECAR','CHGCAR','vasprun.xml']}
  assert all(v['worker_SHA_match'] and v['mtime_not_after_QC'] and v['size']>0 for v in large.values())
  hashes={n:sha(o/n) for n in ['POSCAR','CONTCAR','INCAR','KPOINTS','POTCAR','OUTCAR','lobster-5.1.1','lobsterin','COHPCAR.lobster','ICOHPLIST.lobster','DOSCAR.lobster','lobsterout','bandOverlaps.lobster']}
  smallmatch={n:hashes[n]==v['sha256'] for n,v in t['source_files'].items() if n in hashes};assert all(smallmatch.values())
  assert hashes['lobster-5.1.1']==t['binary_sha256'] and hashes['lobsterin']==t['lobsterin_sha256']
  assert all(hashes[n]==v for n,v in q['output_SHA'].items())
  with (o/'COHPCAR.lobster').open() as f:
   title=next(f);h=next(f).split();nc,n=int(h[0]),int(h[2]);ef=float(h[5]);labels=[next(f).strip() for _ in range(nc)];pairs=[i for i,x in enumerate(labels) if '->' in x and '[' not in x];assert len(pairs)==2
   subs={i:[j for j,x in enumerate(labels) if x.startswith(labels[i].split(':')[0]+':') and '[' in x] for i in pairs};rows=0;last=None;errors={i:0.0 for i in pairs};ie={i:0.0 for i in pairs};near=None;ener=[]
   for line in f:
    if not line.strip():continue
    v=list(map(float,line.split()));assert len(v)==1+2*nc and all(math.isfinite(x) for x in v);assert last is None or v[0]>last;last=v[0];rows+=1
    for i in pairs:
     errors[i]=max(errors[i],abs(v[1+2*i]-sum(v[1+2*j] for j in subs[i])));ie[i]=max(ie[i],abs(v[2+2*i]-sum(v[2+2*j] for j in subs[i])))
    if near is None or abs(v[0])<abs(near[0]):near=v
   assert rows==n==12801 and nc==75
  l,c,el=geom(o/'POSCAR');lc,cc,ec=geom(o/'CONTCAR');assert l==lc and el==ec
  max_coord_delta=max(abs((b-a)-round(b-a)) for x,y in zip(c,cc) for a,b in zip(x,y))
  ico=[x.split() for x in (o/'ICOHPLIST.lobster').read_text().splitlines()];pair_results=[]
  for i in pairs:
   m=re.search(r'No\.(\d+):Rh(\d+)->Cl(\d+)\(([^)]+)\)',labels[i]);assert m
   num,r,k=int(m[1]),int(m[2]),int(m[3]);hits=[x for x in ico if len(x)==8 and x[0]==str(num) and x[1]=='Rh'+str(r) and x[2]=='Cl'+str(k)];assert len(hits)==1;v=hits[0];im=list(map(int,v[4:7]));dr=[c[k-1][j]+im[j]-c[r-1][j] for j in range(3)];cart=[sum(dr[j]*l[j][a] for j in range(3)) for a in range(3)];dist=math.sqrt(sum(x*x for x in cart));assert el[r-1]=='Rh' and el[k-1]=='Cl'
   pair_results.append(dict(label=labels[i],Rh_1based=r,Cl_1based=k,image=im,distance_A=dist,header_distance_A=float(m[4]),ICOHPLIST_distance_A=float(v[3]),ICOHP_eV=float(v[7]),COHP_nearest_EF_ICOHP=near[2+2*i],nearest_EF_energy=near[0],orbital_channels=len(subs[i]),orbital_sum_max_abs=errors[i],orbital_ICOHP_sum_max_abs=ie[i]))
  with (o/'DOSCAR.lobster').open() as f:
   h=[next(f) for _ in range(6)];nat=int(h[0].split()[0]);nd=int(h[5].split()[2]);assert nat==40 and nd==12801;blocks=0
   for ib in range(nat+1):
    if ib:header=next(f);assert int(header.split()[2])==nd
    last=None;cols=None
    for j in range(nd):
     v=list(map(float,next(f).split()));assert len(v)>=2 and all(math.isfinite(x) for x in v);assert last is None or v[0]>last;last=v[0]
     if cols is None:cols=len(v)
     assert len(v)==cols
    blocks+=1
   assert not f.read().strip()
  out=(o/'OUTCAR').read_text();protocol={k:re.search(v,out)[1] for k,v in dict(ENCUT=r'ENCUT\s*=\s*([\d.]+)',NBANDS=r'NBANDS=\s*(\d+)',NELECT=r'NELECT\s*=\s*([\d.]+)',ISPIN=r'ISPIN\s*=\s*(\d+)',SOC=r'LSORBIT\s*=\s*(\S+)',NKPTS=r'NKPTS\s*=\s*(\d+)').items()}
  log=(o/'lobsterout').read_text();basis={}
  for line in log.splitlines():
   if '(pbevaspfit2015)' in line:
    bits=line.split();basis[bits[0]]=bits[2:]
  basis_count=sum(len(basis[e]) for e in el);assert basis_count==192 and protocol['NBANDS']=='360'
  samples=[json.loads(x) for x in (d/'RESOURCE_SAMPLES.jsonl').read_text().splitlines() if x.strip()];full=[x for x in samples if len(x.get('thread_affinity',[]))==t['threads']];binding=bool(full) and all(all(len(a['cpus'])==1 for a in x['thread_affinity']) and set(a['cpus'][0] for a in x['thread_affinity'])==set(t['cpus']) for x in full);a['binding_diagnostic']=dict(total_samples=len(samples),full_OMP_samples=len(full),nonfull_counts=[len(x.get('thread_affinity',[])) for x in samples if len(x.get('thread_affinity',[]))!=t['threads']],full_OMP_samples_bound_correctly=binding)
  assert binding and samples
  stderr=(o/'terminal.stderr').read_text(errors='replace');fatal=[x for x in stderr.splitlines() if re.search(r'out of memory|oom.kill|segmentation|fatal error|time limit',x,re.I)]
  assert not fatal
  a.update(status='INDEPENDENT_TECHNICAL_PASS',source_large_identity_reuse=large,small_input_SHA_match=smallmatch,output_SHA=hashes,COHP_rows=rows,COHP_columns=1+2*nc,DOS_atoms=nat,DOS_blocks=blocks,DOS_rows_per_block=nd,POSCAR_CONTCAR_fractional_max_delta=max_coord_delta,pairs=pair_results,protocol=protocol,basis=basis,basis_orbitals=basis_count,thread_binding_all_recorded_samples=binding,resource_samples=len(samples),fatal_stderr=fatal,scientific_status='SCIENTIFIC_QC_HOLD',scientific_reason='orthonormalization warnings; occupied/near-degenerate band-edge and target-orbital impact not independently closed; low spilling alone is insufficient')
 except Exception as e:a.update(status='HOLD_EVIDENCE_OR_TECHNICAL_FAILURE',error=repr(e),traceback=__import__('traceback').format_exc())
 results.append(a)
 print('QC106_PROGRESS '+state+' '+a['status'],flush=True)
payload=dict(time=time.strftime('%F %T %Z'),results=results,new_DFT=0,new_LOBSTER=0,scheduler_actions=0)
b=base64.b64encode(zlib.compress(json.dumps(payload,separators=(',',':')).encode())).decode();print('QC106_AUDIT_BEGIN')
for i in range(0,len(b),64):print(b[i:i+64])
print('QC106_AUDIT_END',flush=True)
