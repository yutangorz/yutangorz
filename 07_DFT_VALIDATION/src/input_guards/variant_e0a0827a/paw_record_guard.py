"""PAW records retain exact bytes; validate header and following valence record."""

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
_repository_sys.path.insert(0, str(_repo_path('07_DFT_VALIDATION/src/input_guards/variant_e0a0827a')))
import re
def exact_blocks(data):
    out=[];start=0
    for m in re.finditer(br'(?m)^[ \t]*End of Dataset[ \t]*\r?\n',data):
        out.append(data[start:m.end()]);start=m.end()
    assert out and not data[start:].strip(),'Incomplete PAW dataset'
    out[-1]+=data[start:]
    assert b''.join(out)==data
    return out
def reader_layout(data,symbols):
    blocks=exact_blocks(data);assert len(blocks)==len(symbols)
    for block,symbol in zip(blocks,symbols):
        lines=block.splitlines();assert len(lines)>2 and re.match(br'^\s*PAW_PBE\s+',lines[0]),'PAW blank or malformed first record'
        assert lines[0].split()[1].split(b'_')[0].decode()==symbol,'PAW order mismatch'
        assert float(lines[1].strip())>0,'Malformed valence record'
    return True
