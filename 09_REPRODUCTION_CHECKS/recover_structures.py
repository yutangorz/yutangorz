"""Copy SHA-verified frozen parents, or export existing raw OQMD CIF text."""
from pathlib import Path
import argparse,csv,json,hashlib,shutil,gzip
ROOT=Path(__file__).resolve().parents[1]
def rows(p):
 with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def copy_exact(src,dst,expected):
 if sha(src)!=expected:raise RuntimeError('Frozen source hash mismatch: '+str(src))
 if dst.exists():
  if sha(dst)!=expected:raise RuntimeError('Will not overwrite a different structure: '+str(dst))
 else:dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dst)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--source',choices=['oq24','j38','oqmd-raw-cif'],required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--ids',help='Comma-separated fixed IDs for raw OQMD CIF export');a=ap.parse_args()
 if a.source=='oqmd-raw-cif':
  if not a.ids:raise SystemExit('--ids is required; no re-selection is performed')
  wanted=set(a.ids.split(','));found=set()
  with gzip.open(ROOT/'01_DATA_CURATION/data/raw/DPVK-OQMD-cif.csv.gz','rt',encoding='utf-8-sig',newline='') as f:
   for r in csv.DictReader(f):
    ident=r['structure_id']
    if ident not in wanted:continue
    target=a.output/(ident+'.cif');data=r['cif'].encode('utf-8')
    if target.exists() and target.read_bytes()!=data:raise RuntimeError('Different existing CIF: '+str(target))
    target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data);found.add(ident)
  if found!=wanted:raise RuntimeError('Missing IDs: '+str(sorted(wanted-found)))
  print(json.dumps(dict(exported=len(found),format='raw database CIF text',warning='not a claim of byte identity to historical POSCAR files')));return
 if a.source=='oq24':
  table=rows(ROOT/'05_CRISP_X/data/local_reconstruction/new_parent_manifest.csv')
  for r in table:copy_exact(ROOT/'03_DESCRIPTOR_EXTRACTION/data/structures'/(r['ID']+'.vasp'),a.output/(r['ID']+'.vasp'),r['structure_sha256'])
 else:
  table=rows(ROOT/'06_EXTERNAL_TRANSFER/data/external_response/PILOT38.csv')
  for r in table:copy_exact(ROOT/'06_EXTERNAL_TRANSFER/data/external_response/cif'/(r['external_id']+'.cif'),a.output/(r['external_id']+'.cif'),r['structure_sha256'])
 print(json.dumps(dict(structures=len(table),byte_identity_verified=True,new_structures_generated=False)))
if __name__=='__main__':main()
