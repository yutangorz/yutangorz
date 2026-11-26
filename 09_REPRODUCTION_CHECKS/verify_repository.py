"""Verify release integrity, syntax, structured data and local documentation."""
from pathlib import Path
import ast,csv,hashlib,json,re,sys
ROOT=Path(__file__).resolve().parents[1]
def main():
 with (ROOT/'GITHUB_RELEASE_FILE_LIST.csv').open(encoding='utf-8-sig',newline='') as f:manifest=list(csv.DictReader(f))
 for row in manifest:
  p=ROOT/row['path'];assert p.is_file(),row['path'];assert hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256'],row['path']
 py=js=cs=0
 for p in ROOT.rglob('*'):
  if not p.is_file() or '.git' in p.relative_to(ROOT).parts:continue
  if p.suffix=='.py':ast.parse(p.read_text(encoding='utf-8-sig'));py+=1
  if p.suffix=='.json':json.loads(p.read_text(encoding='utf-8-sig'));js+=1
  if p.suffix=='.csv':
   with p.open(encoding='utf-8-sig',newline='') as f:
    r=csv.reader(f);head=next(r,None)
    if head:
     for line in r:assert len(line)==len(head),(str(p.relative_to(ROOT)),len(line),len(head))
   cs+=1
  if p.suffix=='.md':
   for link in re.findall(r'\]\(([^)]+)\)',p.read_text(encoding='utf-8')):
    if not re.match(r'https?://|#',link):assert (p.parent/link.split('#')[0]).exists(),(str(p),link)
 print(json.dumps({'manifest_entries':len(manifest),'Python_AST':py,'JSON':js,'CSV':cs,'scientific_code_imported':False,'scientific_computation_executed':False},indent=2))
if __name__=='__main__':main()
