
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
from collections import Counter
import gzip,json,zipfile,re,hashlib,sys
import numpy as np
import pandas as pd
from pymatgen.core import Structure
from pymatgen.io.vasp import Poscar
from pymatgen.analysis.structure_matcher import StructureMatcher

RUN=_repo_path('07_DFT_VALIDATION/workflows/spectrum_qualification');RAW=_repo_path('07_DFT_VALIDATION/workflows/spectrum_qualification/raw');RES=_repo_path('outputs/07_DFT_VALIDATION/spectrum_qualification')
PROJECT=_repo_path('.');OLD=_repo_path('unavailable_inputs/source_959c59fa00e6/input_bundle')
sys.path.insert(0,str(_repo_path('07_DFT_VALIDATION/src/spectrum_qualification')))
from extraction import ConceptExtractor,FEATURES

with gzip.open(_repo_path('07_DFT_VALIDATION/workflows/spectrum_qualification/raw/stage_audits.jsonl.gz'),'rt',encoding='utf-8') as f:stages=[json.loads(l) for l in f]
with gzip.open(_repo_path('07_DFT_VALIDATION/workflows/spectrum_qualification/raw/inventory.jsonl.gz'),'rt',encoding='utf-8') as f:inventory=[json.loads(l) for l in f]
Z=zipfile.ZipFile(_repo_path('07_DFT_VALIDATION/workflows/spectrum_qualification/raw/small_source_artifacts.zip'));znames=set(Z.namelist())
by_path={s['relative_stage']:s for s in stages}
with gzip.open(_repo_path('07_DFT_VALIDATION/workflows/spectrum_qualification/raw/supplemental_stage_evidence.jsonl.gz'),'rt',encoding='utf-8') as f:supp={s['stage']:s for s in map(json.loads,f)}
for s in stages:
 if 'scan_error' in s:continue
 extra=supp[s['relative_stage']]
 if extra['OUTCAR_ENCUT_eV'] is not None:s['actual_parameters_XML']['ENCUT']=str(extra['OUTCAR_ENCUT_eV'])
 if 'IVDW' not in s['actual_parameters_XML'] and 'IVDW' in extra['XML_incar_snapshot']:
  s['actual_parameters_XML']['IVDW']=extra['XML_incar_snapshot']['IVDW']
links={s['path']:s for s in json.loads((_repo_path('07_DFT_VALIDATION/workflows/spectrum_qualification/raw/SYMLINK_PROVENANCE.json')).read_text())}
strict_matcher=StructureMatcher(ltol=.001,stol=.001,angle_tol=.1,primitive_cell=True,scale=False,attempt_supercell=True)
errors=[];structures={};dcache={};extractor=ConceptExtractor(_repo_path('07_DFT_VALIDATION/data/spectrum_qualification/atom_importance.csv'))
def current_structure(folder,name='POSCAR'):
 key=folder+'/'+name
 if key not in znames and not links.get(key,{}).get('text'):return None
 try:return Structure.from_str(Z.read(key).decode('utf-8','replace') if key in znames else links[key]['text'],'poscar')
 except Exception as exc:errors.append({'source':key,'error':str(exc)});return None
def xml_structure(s,key):
 if not s or not s.get(key):return None
 data=s[key];species=s.get('XML_atom_symbols')
 if not species:
  cur=current_structure(s['relative_stage'])
  species=[x.specie.symbol for x in cur] if cur else None
 if not species or len(species)!=len(data['frac_coords']):return None
 try:return Structure(data['lattice_A'],species,data['frac_coords'],coords_are_cartesian=False)
 except Exception as exc:errors.append({'source':s['relative_stage']+'/'+key,'error':str(exc)});return None
def actual_structure(s,which):
 if not s:return None
 key=(s['relative_stage'],which)
 if key not in structures:
  value=xml_structure(s,'XML_initial_structure' if which=='initial' else 'XML_final_structure')
  if value is None and which=='final':value=current_structure(s['relative_stage'],'CONTCAR')
  structures[key]=value
 return structures[key]
def compare(a,b):
 if a is None or b is None:return None
 if len(a)!=len(b) or a.species!=b.species:return {'species_match':False,'cell_max_A':None,'rms_MIC_A':None}
 cell=float(abs(a.lattice.matrix-b.lattice.matrix).max())
 # Fractional mapping measured using a's lattice; cell difference reported separately.
 moves=np.array([a.lattice.get_distance_and_image(x,y)[0] for x,y in zip(a.frac_coords,b.frac_coords)])
 return {'species_match':True,'cell_max_A':cell,'rms_MIC_A':float(np.sqrt(np.mean(moves*moves))),'max_MIC_A':float(moves.max()),
  'relative_volume_change':float(abs(a.volume-b.volume)/a.volume)}
def same(c,tol=1e-5):return bool(c and c['species_match'] and c['cell_max_A']<tol and c['rms_MIC_A']<tol)
def desc(s):
 if s is None:return None
 k=hashlib.sha256(json.dumps({'lattice':np.round(s.lattice.matrix,9).tolist(),'species':[x.specie.symbol for x in s],
   'coords':np.round(s.frac_coords%1,9).tolist()},sort_keys=True).encode()).hexdigest()
 if k not in dcache:
  try:dcache[k]=extractor.extract(s)[0]
  except Exception as exc:errors.append({'source':'descriptor','error':str(exc)});dcache[k]=None
 return dcache[k]
def health(s):
 if not s or 'scan_error' in s:return False
 return bool(s.get('OUTCAR_normal_footer') and s.get('XML_complete_tail') and s.get('final_scf_iteration_below_NELM')
  and s.get('last_scf_dE_and_deps_pass') and not s.get('changed_during_read') and not s.get('errors_in_windows'))
def gap(s):
 e=s.get('EIGENVAL') if s else None
 return e.get('provisional_occupation_gap_eV') if e else None
def parameter_sig(s):
 p=s.get('actual_parameters_XML',{}) if s else {}
 return json.dumps({k:p.get(k) for k in ['GGA','ENCUT','PREC','LREAL','EDIFF','ISMEAR','SIGMA','ISPIN','LSORBIT','IVDW']},sort_keys=True)

flat=[]
for s in stages:
 if 'scan_error' in s:flat.append({'stage':s['relative_stage'],'scan_error':s['scan_error']});continue
 p=s['actual_parameters_XML'];current=s['current_INCAR'];mismatch=[]
 for k in ['GGA','ENCUT','IBRION','ISIF','NSW','LSORBIT','ICHARG','NBANDS','LELF']:
  if k in current and k in p:
   a,b=current[k].strip().upper().strip('.'),p[k].strip().upper().strip('.')
   try:equiv=abs(float(a)-float(b))<1e-8
   except ValueError:equiv=(a==b or (a in ['T','TRUE'] and b in ['T','TRUE']) or (a in ['F','FALSE'] and b in ['F','FALSE']))
   if not equiv:mismatch.append(k)
 c=compare(current_structure(s['relative_stage']),actual_structure(s,'initial'))
 e=s.get('EIGENVAL') or {}
 flat.append({'stage':s['relative_stage'],'material_id':s['material_id'],'phase':s['phase'],
  'normal_footer':s['OUTCAR_normal_footer'],'XML_complete':s['XML_complete_tail'],'SCF_gate':health(s),
  'ionic_converged_message':s['ionic_required_accuracy'],'last_scf_cap_gate':s['final_scf_iteration_below_NELM'],
  'last_scf_dE_gate':s['last_scf_dE_and_deps_pass'],'last_scf_n':(s.get('OSZICAR') or {}).get('last_scf',{}).get('n') if (s.get('OSZICAR') or {}).get('last_scf') else None,
  'max_force_eV_A':s['last_force_eV_A'],'elapsed_s':s['elapsed_s'],'gap_diagnostic_eV':gap(s),'BAND_GAP_eV':s['BAND_GAP_declared_eV'],
  'fractional_occupation_entries':e.get('fractional_occupation_entries'),'nkpoints':e.get('nkpoints'),'nbands':e.get('nbands'),
  'current_INCAR_vs_XML_mismatch':','.join(mismatch),'LELF_input_execution_difference_not_gap_failure':mismatch==['LELF'],
  'current_POSCAR_matches_XML_initial':same(c),'actual_ENCUT_OUTCAR':supp[s['relative_stage']]['OUTCAR_ENCUT_eV'],
  'XML_original_INCAR_LELF':supp[s['relative_stage']]['XML_incar_snapshot'].get('LELF'),
  'KPOINTS_text':Z.read(s['relative_stage']+'/KPOINTS').decode('utf-8','replace') if s['relative_stage']+'/KPOINTS' in znames else links.get(s['relative_stage']+'/KPOINTS',{}).get('text'),
  'current_POSCAR_XML_comparison':json.dumps(c),'initial_XML_recovered':actual_structure(s,'initial') is not None,
  'final_XML_recovered':s.get('XML_final_structure') is not None,'final_geometry_available_with_CONTCAR_fallback':actual_structure(s,'final') is not None,
  'cohp_caution':s['cohp_overwrite_caution'],
  **{'actual_'+k:p.get(k) for k in ['GGA','ENCUT','PREC','LREAL','EDIFF','NSW','ISIF','IBRION','ICHARG','LSORBIT','ISPIN','IVDW','NBANDS','NPAR','NCORE','LELF']}})
pd.DataFrame(flat).to_csv(_repo_path('outputs/07_DFT_VALIDATION/spectrum_qualification/ALL_1573_STAGE_AUDIT.csv'),index=False,encoding='utf-8-sig')
stageflat={r['stage']:r for r in flat}
jobs={}
for s in stages:
 if s.get('phase') not in ['1_opt','2_scf','3_band']:continue
 job=str(Path(s['relative_stage']).parent).replace('\\','/')
 jobs.setdefault(job,{})[s['phase']]=s
jobrows=[]
for job,phases in jobs.items():
 material=job.split('/')[0];kind='length' if '/bond_adjustment/' in job else ('angle' if '/angle_BXB_adjustment/' in job else ('substitution' if '/perturbation/' in job else 'reference'))
 opt=phases.get('1_opt');scf=phases.get('2_scf');band=phases.get('3_band')
 oi=actual_structure(opt,'initial');of=actual_structure(opt,'final');si=actual_structure(scf,'initial');bi=actual_structure(band,'initial')
 jobrows.append({'job':job,'material_id':material,'kind':kind,'stem':Path(job).name,
  'opt_health':health(opt),'opt_ionic_converged':bool(opt and opt['ionic_required_accuracy']),
  'scf_health':health(scf),'band_health':health(band),
  'opt_to_scf_structure_matches':same(compare(of,si)),'scf_to_band_structure_matches':same(compare(si,bi)),
  'opt_geometry_change':compare(oi,of),'SCF_gap_diagnostic_eV':gap(scf),'band_gap_diagnostic_eV':gap(band),
  'BAND_GAP_eV':band.get('BAND_GAP_declared_eV') if band else None,
  'SCF_protocol':parameter_sig(scf),'cohp_caution':material=='6313886'})
pd.DataFrame(jobrows).to_csv(_repo_path('outputs/07_DFT_VALIDATION/spectrum_qualification/ALL_JOB_CHAIN_AUDIT.csv'),index=False,encoding='utf-8-sig')
ref_by_material={r['material_id']:r for r in jobrows if r['kind']=='reference' and Path(r['job']).name==r['material_id']}
old=pd.read_csv(_repo_path('07_DFT_VALIDATION/data/spectrum_qualification/OLD_POINT_TRIAGE.csv'),dtype={'material_id':str})
alternates=pd.read_csv(_repo_path('07_DFT_VALIDATION/data/spectrum_qualification/ANGLE_ALTERNATE_PATHS.csv'));altpaths=dict(zip(alternates.stem,alternates.paths))
outrows=[];descriptorrows=[];candidates=[]
for n,row in enumerate(old.itertuples(),1):
 matched=[r for r in jobrows if r['material_id']==row.material_id and r['kind']==row.kind and r['stem']==row.stem]
 common={'old_row_id':int(row.row_id),'material_id':row.material_id,'kind':row.kind,'stem':row.stem,'CSV_gap_eV':row.band_gap,
  'cohp_caution':row.material_id=='6313886','matched_job_count':len(matched)}
 if len(matched)!=1:
  outrows.append({**common,'decision':'HOLD_missing_or_multiple_actual_jobs','rerun_required_for_original_claim':'unknown'});continue
 j=matched[0];ph=jobs[j['job']];opt=ph.get('1_opt');scf=ph.get('2_scf');band=ph.get('3_band')
 initial=actual_structure(opt,'initial');final=actual_structure(scf,'initial')
 input_paths=json.loads(altpaths[row.stem]) if row.stem in altpaths else [row.structure_path]
 candidate_matches=[]
 for path in input_paths:
  try:
   ss=Structure.from_file(OLD/path);comp=compare(ss,initial)
   candidates.append({'material_id':row.material_id,'stem':row.stem,'candidate_path':path,'comparison':json.dumps(comp),'matches_actual_opt_input':same(comp)})
   if same(comp):candidate_matches.append(path)
  except Exception as exc:errors.append({'source':path,'error':str(exc)})
 ref=ref_by_material.get(row.material_id);ref_stages=jobs.get(ref['job'],{}) if ref else {}
 ref_i=actual_structure(ref_stages.get('1_opt'),'initial');ref_final=actual_structure(ref_stages.get('2_scf'),'initial')
 before=compare(ref_i,initial);after=compare(ref_final,final);own=compare(initial,final)
 ratio=after['rms_MIC_A']/before['rms_MIC_A'] if before and after and before['species_match'] and after['species_match'] and before['rms_MIC_A']>1e-8 else None
 primitive_reference_equivalent=None;matcher_error=None
 if final is not None and ref_final is not None:
  try:primitive_reference_equivalent=bool(strict_matcher.fit(final,ref_final))
  except Exception as exc:matcher_error=str(exc)
 collapsed=bool(primitive_reference_equivalent and float(row.max_site_MIC_displacement_A)>1e-4)
 g=j['BAND_GAP_eV'];gapdiff=abs(g-row.band_gap) if g is not None and pd.notna(row.band_gap) else None
 consistency=(bool(candidate_matches) and j['opt_to_scf_structure_matches'] and j['scf_to_band_structure_matches'])
 scf_ok=j['scf_health'];band_ok=j['band_health'];opt_ok=j['opt_health'] and j['opt_ionic_converged']
 mismatches=stageflat.get(scf['relative_stage'],{}).get('current_INCAR_vs_XML_mismatch','').split(',') if scf else []
 critical_mismatches=[k for k in mismatches if k and k not in ['LELF','NBANDS']]
 overwritten=bool(critical_mismatches or not stageflat.get(scf['relative_stage'],{}).get('current_POSCAR_matches_XML_initial')) if scf else True
 scf_encut=supp[scf['relative_stage']]['OUTCAR_ENCUT_eV'] if scf else None
 band_encut=supp[band['relative_stage']]['OUTCAR_ENCUT_eV'] if band else None
 refscf=ref_stages.get('2_scf')
 ref_encut=supp[refscf['relative_stage']]['OUTCAR_ENCUT_eV'] if refscf else None
 protocol_encut_match=scf_encut==band_encut==ref_encut
 flags=[]
 if not candidate_matches:flags.append('actual_opt_input_not_identified_in_local_candidates')
 if not scf_ok:flags.append('SCF_convergence_or_completion_gate_failed')
 if not band_ok:flags.append('band_completion_or_iteration_gate_failed')
 if not opt_ok:flags.append('optimization_completion_or_ionic_gate_failed')
 if not j['opt_to_scf_structure_matches']:flags.append('opt_final_to_SCF_input_mismatch')
 if not j['scf_to_band_structure_matches']:flags.append('SCF_to_band_geometry_mismatch')
 if gapdiff is not None and gapdiff>2e-4:flags.append('current_BAND_GAP_does_not_match_old_CSV')
 if collapsed:flags.append('near_reference_after_relaxation_by_declared_geometric_screen')
 if overwritten:flags.append('current_inputs_differ_from_XML_execution_snapshot')
 if not protocol_encut_match:flags.append('SCF_band_reference_ENCUT_protocol_difference')
 if opt and not opt.get('XML_final_structure'):flags.append('optimization_XML_incomplete_using_last_available_CONTCAR_not_converged_claim')
 if row.material_id=='6313886' and (not protocol_encut_match or overwritten or not consistency or gapdiff is None or gapdiff>2e-4):decision='HOLD_6313886_COHP_protocol_provenance';rerun='not_decidable_from_current_conflict'
 elif not scf_ok or not band_ok:decision='REPAIR_OR_RERUN_failed_stage_if_claim_retained';rerun='conditional_on_needed_claim_and_recoverable_originals'
 elif not consistency or (gapdiff is not None and gapdiff>2e-4):decision='HOLD_provenance_or_CSV_conflict';rerun='unknown_recover_lineage_first'
 elif not protocol_encut_match:decision='HOLD_protocol_difference';rerun='same_protocol_comparison_needed_if_claim_retained'
 elif collapsed:decision='NEW_FROZEN_POINTS_NEEDED_if_frozen_path_claim';rerun='yes_for_frozen_path_not_for_valid_relaxed_energy'
 elif not opt_ok:decision='REANALYZE_unconverged_relaxation_path';rerun='conditional_on_equilibrium_or_frozen_claim'
 else:decision='REANALYZE_valid_completed_relaxed_path';rerun='not_automatically_but_not_same_frozen_input'
 z0=desc(ref_final);z=desc(final)
 if z is not None:
  for k,f in enumerate(FEATURES):descriptorrows.append({'material_id':row.material_id,'stem':row.stem,'descriptor':f,
   'actual_SCF_value':float(z[k]),'reference_SCF_value':float(z0[k]) if z0 is not None else None,
   'delta_actual':float(z[k]-z0[k]) if z0 is not None else None})
 outrows.append({**common,**{k:j[k] for k in ['job','opt_health','opt_ionic_converged','scf_health','band_health','opt_to_scf_structure_matches','scf_to_band_structure_matches','SCF_gap_diagnostic_eV','band_gap_diagnostic_eV','BAND_GAP_eV']},
  'local_generated_paths_matching_actual_input':json.dumps(candidate_matches,ensure_ascii=False),'candidate_input_identified':bool(candidate_matches),
  'CSV_vs_current_BAND_GAP_absdiff_eV':gapdiff,'initial_vs_final_geometry':json.dumps(own),
  'initial_vs_reference_rms_A':before['rms_MIC_A'] if before and before.get('species_match') else None,
  'final_vs_reference_rms_A':after['rms_MIC_A'] if after and after.get('species_match') else None,
  'final_vs_reference_cell_max_A':after['cell_max_A'] if after and after.get('species_match') else None,
  'retained_displacement_fraction':ratio,'geometry_near_reference_screen':collapsed,
  'reference_equivalence_primitive_supercell_strict':primitive_reference_equivalent,'matcher_error':matcher_error,
  'reference_SCF_nsites':len(ref_final) if ref_final is not None else None,'point_SCF_nsites':len(final) if final is not None else None,
  'reference_SCF_ENCUT':ref_encut,'point_SCF_ENCUT':scf_encut,'point_band_ENCUT':band_encut,
  'opt_final_geometry_source':'XML_final' if opt and opt.get('XML_final_structure') else 'last_available_CONTCAR_fallback',
  'SCF_protocol_matches_reference':j['SCF_protocol']==ref['SCF_protocol'] if ref else None,
  'current_SCF_inputs_differ_from_execution':bool(overwritten),'flags':';'.join(flags),
  'decision':decision,'rerun_required_for_original_claim':rerun})
 if n%50==0:print('POINTS',n,flush=True)

pd.DataFrame(outrows).to_csv(_repo_path('outputs/07_DFT_VALIDATION/spectrum_qualification/POINT_LEVEL_416_DECISIONS.csv'),index=False,encoding='utf-8-sig')
pd.DataFrame(candidates).to_csv(_repo_path('outputs/07_DFT_VALIDATION/spectrum_qualification/GEOMETRY_VARIANT_DISAMBIGUATION.csv'),index=False,encoding='utf-8-sig')
pd.DataFrame(descriptorrows).to_csv(_repo_path('outputs/07_DFT_VALIDATION/spectrum_qualification/ACTUAL_FINAL_NINE_DESCRIPTORS.csv'),index=False,encoding='utf-8-sig')
df=pd.DataFrame(outrows);summary=[]
for material,g in df.groupby('material_id'):
 summary.append({'material_id':material,'CSV_rows':len(g),'decisions':json.dumps(dict(Counter(g.decision)),ensure_ascii=False),
  'SCF_gate_pass_rows':int(g.scf_health.fillna(False).sum()) if 'scf_health' in g else 0,
  'band_gate_pass_rows':int(g.band_health.fillna(False).sum()) if 'band_health' in g else 0,
  'matched_generated_rows':int(g.candidate_input_identified.fillna(False).sum()) if 'candidate_input_identified' in g else 0,
  'near_reference_rows':int(g.geometry_near_reference_screen.fillna(False).sum()) if 'geometry_near_reference_screen' in g else 0,
  'COHP_rewrite_caution':material=='6313886'})
pd.DataFrame(summary).to_csv(_repo_path('outputs/07_DFT_VALIDATION/spectrum_qualification/MATERIAL_20_SUMMARY.csv'),index=False,encoding='utf-8-sig')
result={'stages':len(stages),'job_chains':len(jobrows),'CSV_points':len(df),'stage_scan_errors':sum('scan_error' in s for s in stages),
 'decision_counts':dict(Counter(df.decision)),'stage_phase_counts':dict(Counter(s.get('phase','ERROR') for s in stages)),
 'XML_actual_parameters_recovered':sum(bool(s.get('actual_parameters_XML')) for s in stages),
 'strict_near_reference_equivalence_points':int(df.geometry_near_reference_screen.fillna(False).sum()),
 'SCF_stage_gate_pass':sum(health(s) for s in stages if s.get('phase')=='2_scf'),
 'local_errors':errors,'reference_materials_found':sorted(ref_by_material),'assumptions':{
 'ordered_geometry_tolerance_A':1e-5,'CSV_BAND_GAP_tolerance_eV':.0002,
 'near_reference_screen':'StructureMatcher primitive/supercell-aware, scale=False, ltol=.001 stol=.001 angle_tol=.1 degree; initial generated displacement>1e-4A. Empirical near-equivalence flag not phase/stability proof; normalized matcher distance not raw Angstrom.',
 'LELF':'current input vs historical XML differences recorded, never counted alone as physical failure or evidence of COHP overwriting',
 'incomplete_opt_XML':'last available CONTCAR matched to SCF; this recovers used geometry but never certifies ionic convergence',
 'all_new_frozen_path_claims':'legacy relaxed outputs cannot be substituted for unrelaxed frozen-input calculations',
 'scf_gate':'normal OUTCAR footer, complete XML tail, last iteration<NELM, |dE|&|d_eps|<EDIFF, no sampled fatal errors; not kmesh/cutoff proof'}}
(_repo_path('outputs/07_DFT_VALIDATION/spectrum_qualification/ANALYSIS_VERDICT.json')).write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
