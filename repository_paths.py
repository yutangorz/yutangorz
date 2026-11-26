"""Repository and prepared-workspace locations. No scientific operation at import."""
from pathlib import Path
import json,os
ROOT=Path(__file__).resolve().parent
def _relative(relative):
 p=Path(str(relative).replace('\\','/'))
 if p.is_absolute() or '..' in p.parts:raise ValueError('Use a repository-relative location')
 return p
def _bindings(kind):
 f=ROOT/'INPUT_PATH_BINDINGS.json'
 return json.loads(f.read_text(encoding='utf-8-sig')).get(kind,{}) if f.exists() else {}
def _alias(p,kind):
 s=p.as_posix()
 for a,b in sorted(_bindings(kind).items(),key=lambda x:-len(x[0])):
  if s==a or s.startswith(a+'/'):return _relative(b+s[len(a):])
 return p
def repo_path(relative):
 p=_alias(_relative(relative),'aliases')
 alternate=os.environ.get('JCTC_INPUT_ROOT')
 if alternate:
  q=Path(alternate).expanduser().resolve()/p
  if q.exists():return q
 if p.parts and p.parts[0]=='outputs':
  return Path(os.environ.get('JCTC_OUTPUT_ROOT',ROOT/'outputs')).expanduser().resolve()/Path(*p.parts[1:])
 if len(p.parts)>1 and p.parts[1]=='workflows':
  base=Path(os.environ.get('JCTC_WORKSPACE_ROOT',ROOT/'workspaces')).expanduser().resolve()
  return base/p.parts[0]/Path(*p.parts[2:])
 return ROOT/p
def input_path(relative):
 """Resolve a READ to a saved input. Never use this function for new outputs."""
 return repo_path(_alias(_relative(relative),'read_aliases'))
