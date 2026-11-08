"""Configurable repository locations; no scientific operations or implicit outputs."""
from pathlib import Path
import os
ROOT=Path(__file__).resolve().parent
def repo_path(relative):
 p=Path(relative)
 if p.is_absolute():raise ValueError('Use a repository-relative location')
 if '..' in p.parts:raise ValueError('Parent traversal is not a configured location')
 alternate=os.environ.get('JCTC_INPUT_ROOT')
 if alternate:
  q=Path(alternate).expanduser().resolve()/p
  if q.exists():return q
 if str(p).replace('\\','/').startswith('outputs/'):
  return Path(os.environ.get('JCTC_OUTPUT_ROOT',ROOT/'outputs')).expanduser().resolve()/Path(*p.parts[1:])
 return ROOT/p
