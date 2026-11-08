
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
_repository_sys.path.insert(0, str(_repo_path('08_COHP_ANALYSIS/src/matched_nine')))
import csv,json,hashlib,shutil,importlib.util
from pathlib import Path
import numpy as np
from pymatgen.io.vasp import Poscar
from pymatgen.io.lobster import Lobsterin
R=_repo_path('.');S=_repo_path('STRUCTURE_PERTURBATION_EXTENSION');O=_repo_path('08_COHP_ANALYSIS/workflows/matched_nine')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return list(csv.DictReader(p.open(encoding='utf-8-sig')))
def write(n,a):
 with (O/n).open('w',newline='',encoding='utf-8-sig') as f:
  w=csv.DictWriter(f,fieldnames=list(a[0]));w.writeheader();w.writerows(a)
mf={x['task_id']:x for x in read(_repo_path('07_DFT_VALIDATION/data/geometry_design/geometry_node_manifest.csv')) if x['material_id']=='6313886'};qc={x['point_id']:x for x in read(_repo_path('unavailable_inputs/source_0abfdfb082fe/SCREENING_POINT_DATA.csv')) if x['material_id']=='6313886'}
cur=next(x for x in read(_repo_path('07_DFT_VALIDATION/data/geometry_design/CURRENT_20_MATERIALS.csv')) if x['material_id']=='6313886');base=Path(cur['baseline_structure']);bs=Poscar.from_file(base).structure;assert sha(base)==cur['baseline_SHA256']
spec=importlib.util.spec_from_file_location('run019',_repo_path('03_DESCRIPTOR_EXTRACTION/src/shared_features/extraction.py'));m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);ex=m.ConceptExtractor(_repo_path('03_DESCRIPTOR_EXTRACTION/data/atom_importance.csv'))
def edges(s):
 i,j,img,d=s.get_neighbor_list(6,exclude_self=True);near=np.full(len(s),np.inf);np.minimum.at(near,i,d)
 return {(int(a),int(b),tuple(map(int,v))) for a,b,v,l in zip(i,j,img,d) if l>1e-7 and l<=near[a]*1.2+1e-7}
be=edges(bs);bv,bq=ex.extract(bs)
paths={}
for kind,key in [('LENGTH','actual_delta_bond_A'),('ANGLE','actual_delta_angle_degree')]:
 a=[x for x in mf.values() if x['kind']==kind];neg=[x for x in a if float(x[key])< -1e-8];pos=[x for x in a if float(x[key])>1e-8]
 paths[kind]=[('PARENT','PARENT'),('NEG_MID',min(neg,key=lambda x:abs(float(x[key])))['task_id']),('NEG_END',max(neg,key=lambda x:abs(float(x[key])))['task_id']),('POS_MID',min(pos,key=lambda x:abs(float(x[key])))['task_id']),('POS_END',max(pos,key=lambda x:abs(float(x[key])))['task_id'])]
 assert abs(abs(float(mf[paths[kind][1][1]][key]))-abs(float(mf[paths[kind][3][1]][key])))<1e-8
selected=list(dict.fromkeys(pid for a in paths.values() for role,pid in a));assert len(selected)==9
basis=Lobsterin.get_basis(bs,['Cs_sv','Li_sv','Rh','Cl']);num=sum(sum(2*'spdf'.index(t[-1])+1 for t in line.split()[1:])*int(bs.composition[line.split()[0]]) for line in basis);assert num==192 and num<=360
audit=[];pairs=[];inputs=[]
for pid in selected:
 g=mf['6313886_LENGTH_06' if pid=='PARENT' else pid];src=base if pid=='PARENT' else Path(g['structure_path']);s=Poscar.from_file(src).structure;v,q=ex.extract(s);assert len(s)==40 and [str(z.specie) for z in s]==[str(z.specie) for z in bs];assert np.max(abs(s.lattice.matrix-bs.lattice.matrix))<1e-9
 if pid!='PARENT':assert sha(src)==g['perturbed_structure_hash'] and abs(float(v[-2])-float(g['global_L_A']))<1e-9 and abs(float(v[-1])-float(g['global_Theta_degree']))<1e-9 and edges(s)==be
 mv,c,fix=json.loads(g['selected_site_indices']);im=json.loads(g['periodic_image_move']);fi=json.loads(g['periodic_image_fixed']);assert (mv,c,fix)==(18,12,17) and im==fi==[0,0,0]
 def vec(j,img):return s.lattice.get_cartesian_coords(s[j].frac_coords+np.array(img)-s[c].frac_coords)
 u,w=vec(mv,im),vec(fix,fi);l=float(np.linalg.norm(u));angle=float(np.degrees(np.arccos(np.clip(u@w/(np.linalg.norm(u)*np.linalg.norm(w)),-1,1))))
 if pid!='PARENT':assert abs(l-float(g['actual_local_bond_length_A']))<1e-8 and abs(angle-float(g['actual_local_bond_angle_degree']))<1e-7
 x=qc.get(pid);audit.append(dict(state=pid,structure_path=str(src),structure_sha=sha(src),baseline_sha=sha(base),atom_count=40,local_length_A=l,local_angle_deg=angle,global_L=float(v[-2]),global_Theta=float(v[-1]),delta_L=float(v[-2])-float(bv[-2]),delta_Theta=float(v[-1])-float(bv[-1]),coverage=q['angle_center_coverage'],shell_changed=edges(s)!=be,job_id=x['job_id'] if x else '12851',Eg_eV=x['Eg_eV'] if x else '1.108907',delta_Eg_eV=x['delta_Eg_eV'] if x else 0,QC_source=x['QC_source'] if x else 'run071 remote12851',electronic_state_audit='PENDING_REMOTE_SNAPSHOT',mechanism='NUMERICAL_ERROR_AND_PROJECTION_QA_PENDING'))
 for j,img in ((mv,im),(fix,fi)):pairs.append(dict(state=pid,center_index0=c,ligand_index0=j,LOBSTER_center1=c+1,LOBSTER_ligand1=j+1,center_element='Rh',ligand_element='Cl',image=json.dumps(img),length_A=float(np.linalg.norm(vec(j,img))),orbitals='ALL_RECOMMENDED_BASIS_CHANNELS_NO_ASSUMED_DOMINANT_ORBITAL'))
 # Small frozen scientific inputs are cloned without changing structure/DFT parameters.
 proto=_repo_path('unavailable_inputs/source_e45ef344e5a6/PARENT');d=_repo_path('08_COHP_ANALYSIS/workflows/matched_nine/STATE_INPUTS_V2')/pid;d.mkdir(parents=True,exist_ok=False);shutil.copyfile(src,d/'POSCAR');shutil.copyfile(proto/'INCAR.candidate',d/'INCAR.reference');shutil.copyfile(proto/'KPOINTS',d/'KPOINTS');shutil.copyfile(proto/'POTCAR_SOURCE.json',d/'POTCAR_SOURCE.json')
 # Window is explicitly provisional until evaluated-spectrum coverage is checked.
 lobster=Lobsterin({'basisSet':'pbeVaspFit2015','basisfunctions':basis,'COHPstartEnergy':-10.0,'COHPendEnergy':5.0,'COHPSteps':3000,'gaussianSmearingWidth':0.05,'cohpbetween':['atom 13 atom 19 orbitalWise','atom 13 atom 18 orbitalWise']});lobster.write_lobsterin(d/'lobsterin.candidate')
 parsed=Lobsterin.from_file(d/'lobsterin.candidate');assert [z.strip() for z in parsed['basisfunctions']]==[z.strip() for z in basis] and parsed['cohpbetween']==['atom 13 atom 19 orbitalWise','atom 13 atom 18 orbitalWise']
 (d/'BASIS_AUDIT.json').write_text(json.dumps(dict(PAW=['Cs_sv','Li_sv','Rh','Cl'],recommended_basis=basis,n_orbitals=num,actual_NBANDS=360,band_count_gate=True,source='pymatgen BASIS_PBE_54_standard recommendation plus LOBSTER5.1.1 manual pp9-10; runtime embedded basis support pending',manual_sha=sha(_repo_path('external_inputs/desktop/小论文/lobster-5.1.1/lobster-5.1.1/Lobster_Users_Guide_5.1.1.pdf')),window_status='PENDING_ACTUAL_SPECTRAL_COVERAGE'),indent=2))
 inputs.append(dict(task_id='COHP_A06_'+pid,state=pid,type='LOBSTER_IF_ELECTRONIC_STATE_REUSABLE',basis_count=192,NBANDS=360,syntax_check='PYMATGEN_PARSE_AND_MANUAL_STATIC_PASS_NOT_ENGINE_TEST',window='PROVISIONAL_MINUS10_PLUS5_RELATIVE_EF_NOT_FULL_BANDWIDTH',status='HOLD',reason='runtime basis/binary/energy coverage/resources/author review pending',state_input=str(d)))
write('NINE_STATE_STRUCTURE_AUDIT.csv',audit);write('TARGET_PAIR_MAP.csv',pairs);write('COHP_INPUT_MANIFEST.csv',inputs)
for kind,a in paths.items():
 write(kind+'_FIVE_POINT_SELECTION.csv',[dict(path=kind,role=role,**next(x for x in audit if x['state']==pid)) for role,pid in a])
write('PROVISIONAL_NODE_ASSIGNMENT.csv',[dict(task_id=x['task_id'],state=x['state'],node=('mgr','node02','node03')[i%3],status='HOLD_NOT_READY_NOT_AUTHORIZED',assignment_basis='unique-task balanced draft; no equal-performance assumption; refresh resources before approval') for i,x in enumerate(inputs)])
with (_repo_path('08_COHP_ANALYSIS/workflows/matched_nine/READY_NODE_ASSIGNMENT.csv')).open('w',newline='',encoding='utf-8-sig') as f:csv.writer(f).writerow(['task_id','state','node','status'])
print('selected9',selected,'basis192<=360; static parse9 pass')
