"""Prepare byte-preserving corrected PAW in a NEW private audit directory; no VASP."""

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
_repository_sys.path.insert(0, str(_repo_path('07_DFT_VALIDATION/src/paw_assembly')))
from pathlib import Path
from datetime import datetime
import base64,hashlib,json,re
OUT=_repo_path('07_DFT_VALIDATION/inputs/FGM/nature_perovskite_revision/audits/substitution_run049_paw_fix_20261022')
assert not OUT.exists(),'Preserve previously prepared correction'
OUT.mkdir(parents=True,mode=0o700)
ROOT=_repo_path('07_DFT_VALIDATION/inputs/FGM/nature_perovskite_revision')
def sha(b):return hashlib.sha256(b).hexdigest()
def exact_blocks(data):
    result=[];start=0
    for match in re.finditer(br'(?m)^[ \t]*End of Dataset[ \t]*\r?\n',data):
        result.append(data[start:match.end()]);start=match.end()
    assert result and not data[start:].strip(),'Missing or incomplete PAW record'
    result[-1]+=data[start:]
    assert b''.join(result)==data,'Byte roundtrip failed'
    return result
def reader_layout(data,expected_symbols):
    blocks=exact_blocks(data);assert len(blocks)==len(expected_symbols)
    for block,symbol in zip(blocks,expected_symbols):
        lines=block.splitlines()
        assert re.match(br'^\s*PAW_PBE\s+',lines[0]),'Blank or malformed first record header'
        assert lines[0].split()[1].split(b'_')[0].decode()==symbol,'Species order mismatch'
        value=float(lines[1].strip());assert value>0,'Malformed valence record'
    return True
reference=_repo_path('07_DFT_VALIDATION/inputs/FGM/DFT_val/6235374/6235374/2_scf/POTCAR')
replacement=_repo_path('licensed_software/paw/PBE/K_sv/POTCAR')
aa=reference.read_bytes();kk=replacement.read_bytes()
assert sha(aa)=='75ed7152f956673e2f82dd69ad63a6926fd0ac4d7a187009baacfc156a3d903c'
assert sha(kk)=='bf8373ef592e31d27efa2dcc68371be6d6a25ce4db6c2ffaf9e92c44050ba21a'
ab=exact_blocks(aa);kb=exact_blocks(kk)
reader_layout(aa,['Cs','Rb','I','O']);reader_layout(kk,['K'])
correct=b''.join([ab[0],kb[0],ab[2],ab[3]])
reader_layout(correct,['Cs','K','I','O'])
cases=[]
for state,job in [('B_baseline','12050'),('B_dense','12051')]:
    old=_repo_path('07_DFT_VALIDATION/inputs/FGM/nature_perovskite_revision/batches/substitution_run046_20261022/6235374')/state
    broken=(old/'POTCAR').read_bytes()
    try:reader_layout(broken,['Cs','K','I','O'])
    except (AssertionError,ValueError):old_rejected=True
    else:old_rejected=False
    assert old_rejected,'Regression did not reject known failing file'
    assert [sha(x.strip()) for x in exact_blocks(broken)]==[sha(x.strip()) for x in exact_blocks(correct)],'PAW payload changed beyond boundary whitespace'
    p=_repo_path('07_DFT_VALIDATION/inputs/FGM/nature_perovskite_revision/audits/substitution_run049_paw_fix_20261022/6235374')/state;p.mkdir(parents=True)
    with (p/'POTCAR').open('xb') as f:f.write(correct)
    (p/'POTCAR').chmod(0o600)
    for name in ['INCAR','POSCAR','KPOINTS']:
        with (p/name).open('xb') as f:f.write((old/name).read_bytes())
    cases.append(dict(state=state,retry_of_job=job,old_POTCAR_sha256=sha(broken),corrected_POTCAR_sha256=sha(correct),same_PAW_payload_except_boundary_whitespace=True,broken_reader_layout_rejected=True,corrected_reader_layout_passed=True,non_PAW_input_sha256={name:sha((p/name).read_bytes()) for name in ['INCAR','POSCAR','KPOINTS']},ready_to_submit=False,why_not_ready='Requires new user node counts and full new-batch preflight; no real VASP reader test or DFT performed'))
text=correct.decode(errors='replace');titles=re.findall(r'TITEL\s*=\s*([^\n]+)',text);zval=[float(x) for x in re.findall(r'ZVAL\s*=\s*([\d.]+)',text)]
assert titles==['PAW_PBE Cs_sv 08Apr2002','PAW_PBE K_sv 06Sep2000','PAW_PBE I 08Apr2002','PAW_PBE O 08Apr2002'] and zval==[9.,9.,7.,6.]
record=dict(time=datetime.now().astimezone().isoformat(),cause='original split removed marker but appended LF while following blocks retained leading LF; extra blank first record before I and O; VASP Fortran unit10 severe59',repair='slice complete End-of-Dataset record lines preserving every original byte, then concatenate complete source datasets without inserting separators',scope='only two6235374 B tasks used custom assembly; all other submitted tasks copied whole original POTCAR',cases=cases,titles=titles,ZVAL=zval,new_submissions=0,new_DFT_calls=0,source_writes=0,real_VASP_parser_reexecution=False)
(_repo_path('07_DFT_VALIDATION/inputs/FGM/nature_perovskite_revision/audits/substitution_run049_paw_fix_20261022/CORRECTION_METADATA.json')).write_text(json.dumps(record,indent=2))
print('CORRECTION_BEGIN'+base64.b64encode(json.dumps(record).encode()).decode()+'CORRECTION_END')
