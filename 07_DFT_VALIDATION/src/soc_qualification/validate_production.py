"""Auditable nonmagnetic two-host gate. Standard library only on the cluster.
Mesh eigenvalue separation is not a full-BZ/quasiparticle/optical gap.
"""

# Portable repository locations and local scientific packages.
from pathlib import Path as _RepositoryPath
import sys as _repository_sys
_repository_root = next((p for p in _RepositoryPath(__file__).resolve().parents if (p / "repository_paths.py").is_file()), None)
if _repository_root is None:
    raise RuntimeError("Repository root marker is missing")
_repository_sys.path.insert(0, str(_repository_root))
from repository_paths import repo_path as _repo_path
_repository_sys.path.insert(0, str(_repo_path('07_DFT_VALIDATION/src/substitution_qualification')))
_repository_sys.path.insert(0, str(_repo_path('07_DFT_VALIDATION/src/input_guards')))
_repository_sys.path.insert(0, str(_repo_path('07_DFT_VALIDATION/src/validation')))
_repository_sys.path.insert(0, str(_repo_path('03_DESCRIPTOR_EXTRACTION/src/shared_features')))
_repository_sys.path.insert(0, str(_repo_path('03_DESCRIPTOR_EXTRACTION/src')))
_repository_sys.path.insert(0, str(_repo_path('05_CRISP_X/src')))
_repository_sys.path.insert(0, str(_repo_path('07_DFT_VALIDATION/src/soc_qualification')))
from pathlib import Path
import json,math,re,xml.etree.ElementTree as ET

def require(condition,message):
    if not condition: raise ValueError(message)

def xml_root(path):
    stack=[];root=None
    for event,elem in ET.iterparse(path,events=('start','end')):
        if event=='start':
            stack.append(elem)
            if root is None:root=elem
        else:
            if any(x.tag=='projected' for x in stack):
                elem.clear()
                if len(stack)>1:stack[-2].remove(elem)
            stack.pop()
    require(root is not None and root.tag=='modeling','Incomplete XML')
    return root

def matrix(node,path):return [[float(v) for v in x.text.split()] for x in node.findall(path)]
def check(folder,config):
    p=Path(folder);root=xml_root(p/'vasprun.xml')
    def val(k):
        e=root.findall(".//parameters//i[@name='%s']"%k)
        if not e:e=root.findall("incar/i[@name='%s']"%k)
        require(bool(e),'Missing actual parameter '+k)
        return e[-1].text.strip()
    for k,x in config.get('input_only_parameters',{}).items():
        supplied=(p/'INCAR').read_text()
        hit=re.findall(r'(?im)^\s*'+re.escape(k)+r'\s*=\s*([^;#\n]+)',supplied)
        require(len(hit)==1 and float(hit[0].strip())==x,'Input-only parameter mismatch '+k)
    if config.get('input_only_parameters',{}).get('IVDW')==0:
        require(val('LUSE_VDW').upper() in ['F','.FALSE.'],'Unexpected nonlocal vdW enabled')
    div=root.find("kpoints/generation/v[@name='divisions']")
    require(div is not None and [int(x) for x in div.text.split()]==config['mesh'],'Actual kmesh mismatch')
    text=(p/'OUTCAR').read_text(errors='replace');require('General timing and accounting informations for this job' in text,'Missing normal OUTCAR footer')
    for k,x in config['actual_parameters'].items():
        y=val(k)
        if isinstance(x,bool):require(y.upper() in (['T','.TRUE.','TRUE'] if x else ['F','.FALSE.','FALSE']),k+' mismatch')
        elif isinstance(x,(float,int)):require(abs(float(y)-x)<max(1e-12,abs(x)*1e-9),k+' mismatch')
        else:require(y.upper()==str(x).upper(),k+' mismatch')
    calculations=root.findall('calculation');require(bool(calculations),'No ionic/static step')
    niter=[len(c.findall('scstep')) for c in calculations]
    require(all(1<n<int(val('NELM')) for n in niter),'SCF reached NELM or missing')
    from scf_rounding import xml_energy_gate
    deltas=[];rounding=[]
    for c in calculations:
        tokens=[s.find("energy/i[@name='e_fr_energy']").text.strip() for s in c.findall('scstep')]
        good,detail=xml_energy_gate(tokens[-2],tokens[-1],val('EDIFF'))
        require(good,'XML energy change inconsistent with EDIFF and printed precision')
        rounding.append(detail);deltas.append(float(detail['exact_printed_delta_eV']))
    scf=[s.split() for s in (p/'OSZICAR').read_text().splitlines() if s.split() and s.split()[0] in ['DAV:','RMM:']]
    ends=[r for i,r in enumerate(scf) if i+1==len(scf) or int(scf[i+1][1])==1]
    require(len(ends)==len(calculations),'OSZICAR/XML ionic step count mismatch')
    require(all(abs(float(r[3]))<float(val('EDIFF')) and abs(float(r[4]))<float(val('EDIFF')) for r in ends),'Per-ionic-step SCF dE/deps failed')
    forces=matrix(calculations[-1],"varray[@name='forces']/v")
    require(bool(forces) and all(math.isfinite(v) for r in forces for v in r),'Missing/nonfinite forces')
    maxforce=max(math.sqrt(sum(v*v for v in r)) for r in forces)
    initial=root.find("structure[@name='initialpos']");final=root.find("structure[@name='finalpos']")
    require(initial is not None and final is not None,'Missing initial/final structure')
    a=matrix(initial,"crystal/varray[@name='basis']/v");b=matrix(final,"crystal/varray[@name='basis']/v")
    require(len(a)==len(b)==3 and max(abs(x-y) for ar,br in zip(a,b) for x,y in zip(ar,br))<1e-7,'Fixed cell changed')
    coords0=matrix(initial,"varray[@name='positions']/v");coords1=matrix(final,"varray[@name='positions']/v")
    require(len(coords0)==len(coords1)==config['nsites'],'Site count mismatch')
    result={'passed':True,'mode':config['mode'],'nionic':len(calculations),'scf_iterations':niter,'SCF_final_energy_deltas_eV':deltas,'XML_printing_precision':rounding,'OSZICAR_final_dE_deps':[[float(r[3]),float(r[4])] for r in ends],'max_force_eV_A':maxforce,'actual_parameters':{k:val(k) for k in config['actual_parameters']},'scope':'limited fixed-lattice PBE two-host protocol, not stability proof'}
    if config['mode']=='relax':
        require('reached required accuracy' in text and maxforce<=.01001,'Ionic force convergence failed')
        require(len(calculations)<int(val('NSW')),'Reached ionic step limit')
    else:
        require(len(calculations)==1,'More than one static step')
        require(max(abs((x-y)-round(x-y)) for aa,bb in zip(coords0,coords1) for x,y in zip(aa,bb))<1e-7,'Frozen atoms moved')
        ne=int(round(float(val('NELECT'))));require(abs(ne-float(val('NELECT')))<1e-6,'Noninteger spinor occupancy')
        tables=[]
        for e in calculations[-1].findall('eigenvalues/array/set'):
            for node in e.iter('set'):
                rows=node.findall('r')
                if rows:tables.append([[float(v) for v in r.text.split()] for r in rows])
        require(bool(tables) and all(len(t)==int(val('NBANDS')) and len(t)>=ne+8 for t in tables),'Insufficient bands')
        require(all(math.isfinite(v) for t in tables for r in t for v in r),'Nonfinite band output')
        vbm=max(t[ne-1][0] for t in tables);cbm=min(t[ne][0] for t in tables)
        # Do not label metals/strongly fractional occupations as a valid gapped response.
        occmin=min(r[1] for t in tables for r in t[:ne]);emptymax=max(r[1] for t in tables for r in t[ne:])
        require(occmin>.999 and emptymax<.001 and cbm>vbm,'Not a resolved insulating spinor spectrum; manual review needed')
        # Independently cross-check mesh edges and electron accounting from EIGENVAL.
        from weight_rounding import normalized_electron_gate
        lines=(p/'EIGENVAL').read_text().splitlines();en,kn,bn=map(int,lines[5].split())
        require(en==ne and bn==int(val('NBANDS')) and kn==len(tables),'EIGENVAL header mismatch')
        idx=6;weights=[];tokens=[];raw_electrons=0.;ev=[];ec=[]
        for ik in range(kn):
            while not lines[idx].strip():idx+=1
            k=lines[idx].split();idx+=1;tokens.append(k[3]);weights.append(float(k[3]));t=[]
            for band in range(bn):
                r=[float(v) for v in lines[idx].split()];idx+=1
                require(len(r)==3 and int(r[0])==band+1,'Unexpected spinor EIGENVAL shape')
                t.append(r)
            require(min(r[2] for r in t[:ne])>.999 and max(r[2] for r in t[ne:])<.001,'EIGENVAL fractional occupancy')
            raw_electrons+=weights[-1]*sum(r[2] for r in t);ev.append(t[ne-1][1]);ec.append(t[ne][1])
        good,account=normalized_electron_gate(ne,raw_electrons,weights,tokens)
        require(good,'EIGENVAL electron accounting failed')
        require(abs((min(ec)-max(ev))-(cbm-vbm))<=1.01e-4,'XML/EIGENVAL gap mismatch')
        result['electron_accounting']=account
        result['EIGENVAL_gap_eV']=min(ec)-max(ev)
        result.update(vbm_eV=vbm,cbm_eV=cbm,mesh_gap_eV=cbm-vbm,n_kpoints=len(tables),occupied_min=occmin,empty_max=emptymax,SOC_reference_force_warning=maxforce>.05)
    require((p/'CONTCAR').is_file() and (p/'CONTCAR').stat().st_size>0,'Missing final CONTCAR')
    result['wall_seconds']=float(re.findall(r'Elapsed time \(sec\):\s*([\d.]+)',text)[-1])
    result['final_forces_eV_A']=forces
    return result

if __name__=='__main__':
    import sys
    p=Path(sys.argv[1]);c=json.loads((p/'TASK.json').read_text())
    r=check(p,c)
    with (p/'validation.json').open('x') as f:json.dump(r,f,indent=2)
    print(json.dumps(r),flush=True)
