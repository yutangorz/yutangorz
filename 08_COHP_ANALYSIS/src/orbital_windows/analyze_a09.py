
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
_repository_sys.path.insert(0, str(_repo_path('08_COHP_ANALYSIS/src/orbital_windows')))
from pathlib import Path
import numpy as np, re, json, csv, hashlib
R=_repo_path('08_COHP_ANALYSIS/workflows/orbital_windows'); P=_repo_path('08_COHP_ANALYSIS/workflows/orbital_windows/EVIDENCE')
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def write(name, rows):
    if not rows:return
    with (R/name).open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
q=json.loads((_repo_path('unavailable_inputs/source_6f1f73e7ab77/QC.json')).read_text())
checks=[]
extra={'PROCAR':'18f82c5bd20ac4bcb4cf29e87889345cf6d94bdb1d644022fb8d24e9097ffd25','EIGENVAL':'766f5d24fbafd6bb71d46d730704ec81f3559a91a7b18b1542247ec9693f9c77','DOSCAR':'64faf3c0b1774e56bd8aa2a33891fb5adedd6e91997c226134058624809dcbc7'}
extra['lobsterin']=q['task']['lobsterin_sha256']
for p in sorted(P.iterdir()):
    expected=extra.get(p.name,q['outputs'].get(p.name,q['input_check'].get(p.name,{})).get('sha256',''))
    checks.append(dict(file=p.name,sha256=sha(p),expected=expected,status='MATCH' if expected==sha(p) else 'UNVERIFIED'))
assert all(x['status']=='MATCH' for x in checks)
write('SOURCE_SHA256.csv',checks)
e=(_repo_path('08_COHP_ANALYSIS/workflows/orbital_windows/EVIDENCE/EIGENVAL')).read_text().splitlines();ne,nk,nb=map(int,e[5].split()); ks=[];ens=[];occs=[];i=6
while len(ks)<nk:
    if not e[i].strip():i+=1;continue
    ks.append(list(map(float,e[i].split())));i+=1
    a=np.array([list(map(float,x.split())) for x in e[i:i+nb]]);ens.append(a[:,1]);occs.append(a[:,2]);i+=nb
ks=np.array(ks);ens=np.array(ens);occs=np.array(occs)
assert nk==64 and nb==360 and np.all(occs[:,:144]>.99) and np.all(occs[:,144:]<.01)
vbm=ens[:,:144].max();cbm=ens[:,144:].min();ef=float(re.findall(r'E-fermi\s*:\s*([-\d.]+)',(_repo_path('08_COHP_ANALYSIS/workflows/orbital_windows/EVIDENCE/OUTCAR')).read_text())[-1])
def key(k):return tuple(np.mod(np.round(k,6),1))
lookup={key(k[:3]):j for j,k in enumerate(ks)}
lines=(_repo_path('08_COHP_ANALYSIS/workflows/orbital_windows/EVIDENCE/bandOverlaps.lobster')).read_text().splitlines();mat={};i=0
while i<len(lines):
    if 'at k-point' not in lines[i]:i+=1;continue
    k=list(map(float,lines[i].split('at k-point')[1].split()));reported=float(lines[i+1].split(':')[-1]);i+=2
    a=[]
    while i<len(lines):
        try:row=list(map(float,lines[i].split()))
        except ValueError:break
        if len(row)!=192:break
        a.append(row);i+=1
        if len(a)==192:break
    a=np.array(a);assert a.shape==(192,192);mat[lookup[key(k)]]=(a,reported)
assert len(mat)==48
rows=[];targets=set()
for j,k in enumerate(ks):
    row=dict(k_index_1based=j+1,kx=k[0],ky=k[1],kz=k[2],weight=k[3],matrix_status='RECORDED_WARNING' if j in mat else 'NOT_RECORDED_NOT_ZERO',software_maxDeviation='UNKNOWN',global_max='UNKNOWN',occupied_max='UNKNOWN',occ_unocc_max='UNKNOWN',unocc_max='UNKNOWN',band144_row='UNKNOWN',band145_row='UNKNOWN',max_row='UNKNOWN',max_col='UNKNOWN',max_row_energy_eV='UNKNOWN',max_col_energy_eV='UNKNOWN',occupied_max_row='UNKNOWN',occupied_max_col='UNKNOWN',occupied_row_energy='UNKNOWN',occupied_col_energy='UNKNOWN',band144_argmax_col='UNKNOWN',band145_argmax_col='UNKNOWN',band144_energy=ens[j,143],band145_energy=ens[j,144],orbital_identity='REQUIRES_PROCAR_MAPPING_NOT_ORBITAL_MATRIX_INDEX')
    if j in mat:
        a,rep=mat[j];d=abs(a-np.eye(192));r,c=np.unravel_index(d.argmax(),d.shape);r1,c1=np.unravel_index(d[:144,:144].argmax(),(144,144));b144=int(d[143].argmax());b145=int(d[144].argmax())
        row.update(software_maxDeviation=rep,global_max=d.max(),occupied_max=d[:144,:144].max(),occ_unocc_max=d[:144,144:].max(),unocc_max=d[144:,144:].max(),band144_row=d[143].max(),band145_row=d[144].max(),max_row=r+1,max_col=c+1,max_row_energy_eV=ens[j,r],max_col_energy_eV=ens[j,c],occupied_max_row=r1+1,occupied_max_col=c1+1,occupied_row_energy=ens[j,r1],occupied_col_energy=ens[j,c1],band144_argmax_col=b144+1,band145_argmax_col=b145+1)
        for b in [r,c,r1,c1,143,144,b144,b145]:targets.add((j,b))
    rows.append(row)
write('BANDOVERLAPS_FULL_DIAGNOSTIC.csv',rows)
# All sampled near-edge states; 0.10 eV is a reporting window, not acceptance tolerance.
for j in range(nk):
    for b in np.where(((ens[j]>=vbm-.10)&(ens[j]<=vbm+1e-6))|((ens[j]>=cbm-1e-6)&(ens[j]<=cbm+.10)))[0]:targets.add((j,int(b)))
for b in range(136,148):targets.add((0,b))
els=['Cs']*8+['Li']*4+['Rh']*4+['Cl']*24
pro={};j=-1;b=-1
with (_repo_path('08_COHP_ANALYSIS/workflows/orbital_windows/EVIDENCE/PROCAR')).open() as f:
    for line in f:
        if line.lstrip().startswith('k-point'):j=int(line.split()[1])-1
        elif line.startswith('band '):
            m=re.match(r'band\s+(\d+) # energy\s+([-\d.]+)',line);b=int(m[1])-1
            if (j,b) in targets:assert abs(float(m[2])-ens[j,b])<2e-6
        elif line.lstrip().startswith('ion') and (j,b) in targets:
            dat=np.array([list(map(float,next(f).split()[1:])) for _ in range(40)])
            pro[j,b]=dat
orb=['s','py','pz','px','dxy','dyz','dz2','dxz','x2-y2'];edge=[]
for (j,b),a in sorted(pro.items()):
    total=float(a[:,-1].sum());pairs=a[[12,17,18],:-1].sum(axis=0);top=np.argsort(a[:,:-1].sum(axis=0))[-3:][::-1]
    edge.append(dict(k_index=j+1,kpoint=' '.join(map(str,ks[j,:3])),band=b+1,energy_eV=ens[j,b],occupation=occs[j,b],relative_VBM=ens[j,b]-vbm,relative_CBM=ens[j,b]-cbm,near_edge_window_0p10=bool(vbm-.1<=ens[j,b]<=vbm+1e-6 or cbm-1e-6<=ens[j,b]<=cbm+.1),PROCAR_sphere_total=total,Cs_weight=a[:8,-1].sum(),Li_weight=a[8:12,-1].sum(),Rh_weight=a[12:16,-1].sum(),Cl_weight=a[16:,-1].sum(),target_Rh13_weight=a[12,-1],target_Cl18_weight=a[17,-1],target_Cl19_weight=a[18,-1],target_three_ions_fraction_of_sphere_sum=float(a[[12,17,18],-1].sum()/total) if total else 'UNKNOWN',top_lm=';'.join(orb[t] for t in top),target_Rh_d=a[12,4:9].sum(),target_Cl18_p=a[17,1:4].sum(),target_Cl19_p=a[18,1:4].sum(),warning_matrix=j in mat,qualification='PAW_SPHERE_WEIGHTS_NOT_PAIR_COHP_OR_LCAO_AMPLITUDES'))
write('BAND_EDGE_ORBITAL_AUDIT.csv',edge)
mapping=[]
for (j,b),a in sorted(pro.items()):
    d=abs(mat[j][0]-np.eye(192)) if j in mat else None
    flat=np.argsort(a[:,:-1].ravel())[-5:][::-1]
    mapping.append(dict(k_index=j+1,band=b+1,energy=ens[j,b],row_max=d[b].max() if d is not None else 'UNKNOWN',row_max_col=int(d[b].argmax()+1) if d is not None else 'UNKNOWN',occupied_row_max=d[b,:144].max() if d is not None else 'UNKNOWN',unoccupied_row_max=d[b,144:].max() if d is not None else 'UNKNOWN',top_ion_orbitals=';'.join(str(int(t//9+1))+':'+els[t//9]+':'+orb[t%9]+':'+str(a[t//9,t%9]) for t in flat)))
write('BAND_STATE_PROJECTION_MAP.csv',mapping)
# COHP columns: average + pair total and orbital resolved; values are COHP (negative bonding), not -COHP.
f=(_repo_path('08_COHP_ANALYSIS/workflows/orbital_windows/EVIDENCE/COHPCAR.lobster')).read_text().splitlines();nh=int(f[1].split()[0]);npts=int(f[1].split()[2]);ef_exact=float(f[1].split()[5]);labels=f[2:2+nh];a=np.array([list(map(float,x.split())) for x in f[2+nh:]])
assert a.shape==(npts,1+2*nh)
tot=[i for i,s in enumerate(labels) if '->' in s and '[' not in s];assert len(tot)==2
co=[]
for ti in tot:
    label=labels[ti];pair=label.split(':')[0];subs=[i for i,s in enumerate(labels) if s.startswith(pair+':') and '[' in s]
    err=float(abs(a[:,1+2*ti]-a[:,[1+2*i for i in subs]].sum(axis=1)).max());ierr=float(abs(a[:,2+2*ti]-a[:,[2+2*i for i in subs]].sum(axis=1)).max())
    for name,lo,hi in [('VBM_minus_0p1',vbm-ef_exact-.1,vbm-ef_exact+.1),('CBM_plusminus_0p1',cbm-ef_exact-.1,cbm-ef_exact+.1)]:
        m=(a[:,0]>=lo)&(a[:,0]<=hi);idx=int(np.argmin(abs(a[:,0]-(vbm if name.startswith('VBM') else cbm)+ef_exact)))
        co.append(dict(pair=label,window=name,energy_reference='E-EF, EF='+str(ef_exact),lo=lo,hi=hi,COHP_at_nearest_edge=a[idx,1+2*ti],window_COHP_integral_trapezoid=np.trapz(a[m,1+2*ti],a[m,0]),orbital_sum_max_abs_residual=err,integrated_orbital_sum_max_abs_residual=ierr,orbital_channels=len(subs),scientific_status='ALGEBRA_CHECK_ONLY_NOT_PROJECTION_VALIDATION'))
write('RH_CL_COHP_NUMERICAL_CHECK.csv',co)
# LOBSTER pDOS, integrate matched windows and retain signs; no forced normalization equivalence to VASP.
f=(_repo_path('08_COHP_ANALYSIS/workflows/orbital_windows/EVIDENCE/DOSCAR.lobster')).read_text().splitlines();nd=int(f[5].split()[2]);offset=6+nd;pd=[]
for atom in range(40):
    header=f[offset];names=header.split(';')[-1].split();offset+=1
    dat=np.array([list(map(float,x.split())) for x in f[offset:offset+nd]]);offset+=nd
    for name,energy in [('VBM',vbm-ef_exact),('CBM',cbm-ef_exact)]:
        m=abs(dat[:,0]-energy)<=.1
        for l,c in zip(names,dat[:,1:].T):
            pd.append(dict(atom=atom+1,element=els[atom],orbital=l,edge=name,window_eV=0.1,integral=np.trapz(c[m],dat[m,0]),negative_samples=int((c[m]<0).sum()),qualification='K_INTEGRATED_DOS_NOT_K_BAND_RESOLVED'))
write('LOBSTER_PDOS_EDGE_WINDOWS.csv',pd)
summary=dict(matrices=len(mat),nk=nk,occupied_deviation_gt_software_precision=sum(r['occupied_max']>1e-5 for r in rows if r['matrix_status']=='RECORDED_WARNING'),max_occupied=max(r['occupied_max'] for r in rows if r['matrix_status']=='RECORDED_WARNING'),VBM=vbm,CBM=cbm,gap=cbm-vbm,EF=ef_exact,actual_VBM_states=[(int(j+1),int(b+1)) for j,b in zip(*np.where(abs(ens-vbm)<1e-6))],actual_CBM_states=[(int(j+1),int(b+1)) for j,b in zip(*np.where(abs(ens-cbm)<1e-6))],near_edge_states=sum(x['near_edge_window_0p10'] for x in edge),cohp=co,limitations=['16 nonwarning matrices not serialized; do not invent zeros','band overlaps indices identify projected bands, not atomic orbitals','PROCAR rounded PAW-sphere weights and LCAO projections not quantitatively identical','no k-resolved COHP or projection coefficient matrix in supplied outputs'])
summary['edge_overlap_windows']=[]
for half in [.05,.1,.2]:
    for edge_name,level,occupied in [('VBM',vbm,True),('CBM',cbm,False)]:
        pairs=[(j,b) for j in range(nk) for b in range(192) if abs(ens[j,b]-level)<=half and ((b<144)==occupied)]
        vals=[float(abs(mat[j][0]-np.eye(192))[b].max()) for j,b in pairs if j in mat]
        summary['edge_overlap_windows'].append(dict(edge=edge_name,half_window=half,states=len(pairs),states_with_warning_matrix=len(vals),max_row_deviation=max(vals) if vals else None))
(_repo_path('08_COHP_ANALYSIS/workflows/orbital_windows/NUMERICAL_SUMMARY.json')).write_text(json.dumps(summary,ensure_ascii=False,indent=2,default=lambda x:x.item()),encoding='utf-8')
print(json.dumps(summary,ensure_ascii=False,indent=2,default=lambda x:x.item()))
