"""Scalar ISPIN1 specialization; same SCF/occupancy gates, no metallic zero fill."""

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
_repository_sys.path.insert(0, str(_repo_path('07_DFT_VALIDATION/inputs/1230247/B')))
from pathlib import Path
import json,re,math
from xml_tools_frozen import xml_root,matrix
from scf_rounding import xml_energy_gate
from occupation_rounding import normalized_electron_gate,half_unit
def need(v,msg):
 if not v:raise ValueError(msg)
def spectral_summary(channels,ne):
 need(len(channels)==1 and bool(channels[0]),'One ISPIN1 channel required')
 full=1. if max(r[1] for t in channels[0] for r in t)<=1.001 else 2.
 occ=[];empty=[];counts=[]
 for t in channels[0]:
  need(all(len(r)==2 and all(math.isfinite(x) for x in r) and -1e-6<=r[1]/full<=1.000001 for r in t),'Invalid NM eigenvalues')
  need(all(t[i][0]<=t[i+1][0]+1e-7 for i in range(len(t)-1)),'Band ordering')
  n=sum(r[1]/full>.5 for r in t);counts.append(n);need(0<n<len(t) and t[-1][1]/full<.001,'Missing occupied/empty manifold')
  need(all(r[1]/full>.5 for r in t[:n]) and all(r[1]/full<=.5 for r in t[n:]),'Noncontiguous manifold')
  occ.extend([e,o/full] for e,o in t[:n]);empty.extend([e,o/full] for e,o in t[n:])
 stable=len(set(counts))==1 and 2*counts[0]==ne;v=max(r[0] for r in occ);c=min(r[0] for r in empty);nominal=c-v if stable else None;omin=min(r[1] for r in occ);emax=max(r[1] for r in empty)
 resolved=stable and nominal>1e-5 and omin>.999 and emax<.001
 return dict(gap_eV=nominal if resolved else None,nominal_band_separation_eV=nominal,VBM_eV=v,CBM_eV=c,occupied_min_normalized=omin,empty_max_normalized=emax,occupation_full=full,occupied_counts=counts,manifold_k_constant=stable,metal_verified=False,NA_not_zero=True)
def check(folder,c):
 p=Path(folder);root=xml_root(p/'vasprun.xml');a=c['actual_parameters'];need(c['mode']=='pbe_nm_lowcost_v1' and a['ISPIN']==1 and a['LSORBIT'] is False and a['LNONCOLLINEAR'] is False and a['EDIFF']==1e-6,'Wrong NM branch')
 def val(k):
  z=root.findall(".//parameters//i[@name='%s']"%k) or root.findall("incar/i[@name='%s']"%k);need(bool(z),'Missing '+k);return z[-1].text.strip()
 for k,x in a.items():
  y=val(k)
  if isinstance(x,bool):need(y.upper() in (['T','TRUE','.TRUE.'] if x else ['F','FALSE','.FALSE.']),k+' mismatch')
  elif isinstance(x,(int,float)):need(abs(float(y)-x)<max(1e-12,abs(x)*1e-9),k+' mismatch')
  else:need(y.upper()==str(x).upper(),k+' mismatch')
 need(val('LDAU').upper() in ['F','FALSE','.FALSE.'],'U not allowed');div=root.find("kpoints/generation/v[@name='divisions']");need(div is not None and list(map(int,div.text.split()))==c['mesh'],'Mesh mismatch')
 text=(p/'OUTCAR').read_text(errors='replace');need('General timing and accounting informations for this job' in text,'Incomplete OUTCAR');calc=root.findall('calculation');need(len(calc)==1,'Only static one-frame calculation');steps=calc[0].findall('scstep');need(1<len(steps)<a['NELM'],'SCF cap or incomplete')
 tok=[s.find("energy/i[@name='e_fr_energy']").text.strip() for s in steps];ok,precision=xml_energy_gate(tok[-2],tok[-1],a['EDIFF']);need(ok,'XML energy convergence')
 fp=r'([-+]?\d*\.?\d+[Ee][-+]\d{2})';pat=re.compile(r'^\s*(DAV:|RMM:)\s*(\d+)\s+'+fp+r'\s+'+fp+r'\s+'+fp);rows=[]
 for line in (p/'OSZICAR').read_text().splitlines():
  if line.split() and line.split()[0] in ['DAV:','RMM:']:
   m=pat.match(line);need(bool(m),'SCF residual parse');rows.append((int(m[2]),float(m[4]),float(m[5])))
 need(len(rows)==len(steps) and rows[-1][0]==len(steps),'SCF step count');need(abs(rows[-1][1])<a['EDIFF'] and abs(rows[-1][2])<a['EDIFF'],'SCF dE/deps')
 initial=root.find("structure[@name='initialpos']");final=root.find("structure[@name='finalpos']");need(initial is not None and final is not None,'Missing geometry');p0=matrix(initial,"varray[@name='positions']/v");p1=matrix(final,"varray[@name='positions']/v");b0=matrix(initial,"crystal/varray[@name='basis']/v");b1=matrix(final,"crystal/varray[@name='basis']/v")
 need(len(p0)==len(p1)==c['nsites'] and max(abs((x-y)-round(x-y)) for u,v in zip(p0,p1) for x,y in zip(u,v))<1e-7,'Geometry moved');need(max(abs(x-y) for u,v in zip(b0,b1) for x,y in zip(u,v))<1e-7,'Cell moved')
 lines=(p/'POSCAR').read_text().splitlines();need(lines[7].strip().lower().startswith('d'),'Direct POSCAR required');co=[list(map(float,z.split()[:3])) for z in lines[8:8+c['nsites']]];need(max(abs((x-y)-round(x-y)) for u,v in zip(co,p0) for x,y in zip(u,v))<1e-7,'Input coordinates mismatch')
 atoms=root.find("atominfo/array[@name='atoms']/set");need([x.findall('c')[0].text.strip() for x in atoms.findall('rc')]==c['symbols_per_site'],'Composition/order mismatch')
 top=calc[0].find('eigenvalues/array/set');need(top is not None,'Missing spectra');spinsets=top.findall('set');need(len(spinsets)==1,'Expected ISPIN1 eigenvalues');tables=[[[float(v) for v in r.text.split()] for r in k.findall('r')] for k in spinsets[0].findall('set')];need(all(len(t)==a['NBANDS'] for t in tables),'Band count')
 xmls=spectral_summary([tables],int(a['NELECT']));weights=root.find("kpoints/varray[@name='weights']");wt=[r.text.strip() for r in weights.findall('v')];wx=[float(x) for x in wt];need(len(wx)==len(tables),'k count');scale=2/xmls['occupation_full'];raw=sum(w*sum(r[1] for r in tables[k]) for k,w in enumerate(wx))*scale;bound=sum(wx[k]*sum(half_unit(r.text.split()[1]) for r in ks.findall('r')) for k,ks in enumerate(spinsets[0].findall('set')))*scale;ok,ax=normalized_electron_gate(a['NELECT'],raw,wx,wt,bound);need(ok,'XML electron count')
 el=(p/'EIGENVAL').read_text().splitlines();ne,nk,nb=map(int,el[5].split());need(ne==a['NELECT'] and nk==len(wx) and nb==a['NBANDS'],'EIGENVAL header');idx=6;es=[];w=[];wt2=[];raw=0.;bound=0.
 for k in range(nk):
  while not el[idx].strip():idx+=1
  point=el[idx].split();idx+=1;need(len(point)==4,'k row');w.append(float(point[3]));wt2.append(point[3]);t=[]
  for b in range(nb):
   z=el[idx].split();idx+=1;need(len(z)==3 and int(z[0])==b+1,'Expected scalar3column EIGENVAL');en,oc=float(z[1]),float(z[2]);need(abs(en-tables[k][b][0])<=1.01e-4,'XML/EIGENVAL disagreement');t.append([en,oc]);raw+=w[-1]*oc;bound+=w[-1]*half_unit(z[2])
  es.append(t)
 esum=spectral_summary([es],ne);scale=2/esum['occupation_full'];ok,ae=normalized_electron_gate(ne,raw*scale,w,wt2,bound*scale);need(ok,'EIGENVAL electron count');need(xmls['manifold_k_constant']==esum['manifold_k_constant'],'Manifold disagreement')
 if esum['nominal_band_separation_eV'] is not None:need(abs(esum['nominal_band_separation_eV']-xmls['nominal_band_separation_eV'])<=1.01e-4,'Gap disagreement')
 resolved=xmls['gap_eV'] is not None and esum['gap_eV'] is not None;force=matrix(calc[0],"varray[@name='forces']/v");need(len(force)==c['nsites'] and all(math.isfinite(x) for r in force for x in r),'Forces missing');need((p/'CONTCAR').is_file() and (p/'CONTCAR').stat().st_size>0,'CONTCAR missing');wall=re.findall(r'Elapsed time \(sec\):\s*([\d.]+)',text);need(bool(wall),'Timing missing')
 return dict(passed=True,technical_validity='passed',mode=c['mode'],SOC=False,ISPIN=1,EDIFF=a['EDIFF'],actual_parameters=a,scf_iterations=[len(steps)],SCF_final_dE_deps=rows[-1][1:],XML_precision=precision,XML_spectrum=xmls,EIGENVAL_spectrum=esum,electron_accounting=ae,XML_electron_accounting=ax,gap_eV=esum['gap_eV'] if resolved else None,nominal_band_separation_eV=esum['nominal_band_separation_eV'],spectral_class='resolved_semiconductor_on_sampled_mesh' if resolved else 'unresolved_nm_requires_review',wall_seconds=float(wall[-1]),max_force_eV_A=max(math.sqrt(sum(x*x for x in r)) for r in force),magnetic_ground_state_verified=False,metal_verified=False,scope='independent same-cluster nonSOC NM fixed-geometry finite reference; not original20 coverage')
if __name__=='__main__':
 p=Path('.');v=check(p,json.loads((p/'TASK.json').read_text()))
 with (p/'validation.json').open('x') as f:json.dump(v,f,indent=2)
 print(json.dumps(v))
