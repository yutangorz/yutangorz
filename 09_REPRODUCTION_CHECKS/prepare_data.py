"""Losslessly decompress packed inputs; never run scientific code."""
from pathlib import Path
import argparse,gzip,hashlib,json
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);p.add_argument('--dry-run',action='store_true');a=p.parse_args();out=Path(a.output).resolve()
 if out.exists() or out==ROOT or ROOT in out.parents:raise SystemExit('Use a new location outside this repository')
 files=list(ROOT.rglob('*.gz'))
 if not a.dry_run:
  for f in files:
   q=out/f.relative_to(ROOT).with_suffix('');q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes(gzip.decompress(f.read_bytes()))
 print(json.dumps({'archives':len(files),'dry_run':a.dry_run,'scientific_computation_executed':False,'next_step':'Set JCTC_INPUT_ROOT to this output root before stages requiring decompressed inputs.'},indent=2))
if __name__=='__main__':main()
