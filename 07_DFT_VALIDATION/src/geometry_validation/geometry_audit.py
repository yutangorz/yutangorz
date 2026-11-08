"""Independent motif audit for the two coverage-held SOC materials; no DFT."""

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
_repository_sys.path.insert(0, str(_repo_path('07_DFT_VALIDATION/src/geometry_validation')))
from pathlib import Path
import csv,hashlib,importlib.util,json,math
import numpy as np
from pymatgen.io.vasp import Poscar
S=_repo_path('STRUCTURE_PERTURBATION_EXTENSION');R=_repo_path('07_DFT_VALIDATION/workflows/geometry_validation')
rows={x['material_id']:x for x in csv.DictReader((_repo_path('07_DFT_VALIDATION/data/geometry_design/CURRENT_20_MATERIALS.csv')).open(encoding='utf-8-sig',newline=''))}
ext=_repo_path('03_DESCRIPTOR_EXTRACTION/src/shared_features/extraction.py')
table=_repo_path('03_DESCRIPTOR_EXTRACTION/data/atom_importance.csv')
spec=importlib.util.spec_from_file_location('run019',ext);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
extract=m.ConceptExtractor(table)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def vec(s,c,n,img):return (s.frac_coords[n]+np.asarray(img)-s.frac_coords[c])@s.lattice.matrix
def angle(a,b):return float(np.degrees(np.arccos(np.clip(np.dot(a,b)/np.linalg.norm(a)/np.linalg.norm(b),-1,1))))
def shell(s):
 ci,pj,off,dist=s.get_neighbor_list(6.0,exclude_self=True);near=np.full(len(s),np.inf);np.minimum.at(near,ci,dist)
 return frozenset((int(i),int(j),tuple(int(v) for v in o)) for i,j,o,d in zip(ci,pj,off,dist) if d>1e-7 and d<=near[i]*1.2+1e-7)
out=[]
for mid,bsym,anion in [('8158832','Ga','F'),('6235374','I','O')]:
 s=Poscar.from_file(rows[mid]['baseline_structure']).structure
 assert sha(Path(rows[mid]['baseline_structure']))==rows[mid]['baseline_SHA256']
 centers=[i for i,site in enumerate(s) if str(site.specie)==bsym];assert len(centers)==2
 c=centers[0];ns=[]
 for n in s.get_neighbors(s[c],4.8):
  if str(n.specie)==anion:ns.append((int(n.index),tuple(int(v) for v in n.image),float(n.nn_distance)))
 nearest=min(x[2] for x in ns);ns=[x for x in ns if x[2]<=nearest*1.2+1e-7]
 assert len(ns)==3
 pairs=[]
 for j,ij,d in ns:
  for k,ik,dk in ns:
   if j>=k:continue
   a=angle(vec(s,c,j,ij),vec(s,c,k,ik));pairs.append((abs(a-90),j,ij,k,ik,a))
 _,j,ij,k,ik,a=min(pairs)
 v=vec(s,c,j,ij);d=np.linalg.norm(v);u=v/d;w=vec(s,c,k,ik);w=w/np.linalg.norm(w)
 theta=math.acos(float(np.clip(np.dot(u,w),-1,1)));perp=(u-math.cos(theta)*w)/math.sin(theta)
 target=theta+math.radians(-1.0);shift=d*(math.cos(target)*w+math.sin(target)*perp-u)
 t=s.copy();t.translate_sites([j],shift,frac_coords=False,to_unit_cell=False)
 folder=R/mid;folder.mkdir(exist_ok=True)
 path=folder/'CANDIDATE_ANGLE_MINUS1_POSCAR';Poscar(t).write_file(str(path));t=Poscar.from_file(str(path)).structure
 local=angle(vec(t,c,j,ij),vec(t,c,k,ik));baseval,bq=extract.extract(s);val,q=extract.extract(t)
 dm=t.distance_matrix.copy();np.fill_diagonal(dm,np.inf)
 record={'material_id':mid,'baseline_sha256':rows[mid]['baseline_SHA256'],'candidate_sha256':sha(path),'center':c,'move':j,'move_image':ij,'fixed':k,'fixed_image':ik,'species':[anion,bsym,anion],'nominal_delta_angle_degree':-1.0,'actual_delta_angle_degree':local-a,'actual_delta_bond_A':np.linalg.norm(vec(t,c,j,ij))-d,'baseline_angle_center_coverage':bq['angle_center_coverage'],'candidate_angle_center_coverage':q['angle_center_coverage'],'baseline_global_L_A':baseval[-2],'candidate_global_L_A':val[-2],'baseline_global_Theta_degree':baseval[-1],'candidate_global_Theta_degree':val[-1],'shell_changed':shell(t)!=shell(s),'minimum_distance_A':float(dm.min()),'status':'HOLD_GLOBAL_COVERAGE_UNCHANGED'}
 (folder/'GEOMETRY_AUDIT.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 out.append(record)
(_repo_path('07_DFT_VALIDATION/workflows/geometry_validation/GEOMETRY_AUDIT_SUMMARY.json')).write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(out,ensure_ascii=False))
