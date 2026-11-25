"""Verify the frozen publication manifest without loading models or private data."""
from pathlib import Path
import hashlib,json,re
ROOT=Path(__file__).resolve().parents[1]
ALLOWED_DRIVE={'1DgXQnvK3keGP9BKB-cqpixnKaNLl2746','1OnA3rHn5uKAMEuHsQc11oxueTT-TQQwJ'}
FORBIDDEN={'.jpg','.jpeg','.gif','.webp','.mp4','.mov','.mkv','.avi','.pdf','.doc','.docx','.ppt','.pptx','.zip','.sqlite','.sqlite3','.db','.pyc','.pyo','.pth'}
def check(paths):
 for rel in paths:
  p=Path(rel)
  if p.suffix.lower()in FORBIDDEN:raise ValueError('Forbidden publication file: '+rel)
  if any(x in ('.git','vendor','__pycache__','screenshots','data')for x in p.parts):raise ValueError('Private/cache/runtime directory: '+rel)
  if 'report'in p.name.lower()or 'presentation'in p.name.lower():raise ValueError('Academic deliverable: '+rel)
  raw=(ROOT/p).read_bytes()
  if len(raw)>50*1024*1024:raise ValueError('Unexpected large artifact: '+rel)
  if p.suffix in ('.py','.ipynb','.md','.json','.csv','.jsonl','.txt','.bib'):
   text=raw.decode('utf-8-sig')
   if re.search(r'(?i)[A-Z]:[\\/]+Users[\\/]+',text):raise ValueError('Personal Windows path: '+rel)
   if re.search(r'ghp_[A-Za-z0-9]{25,}|github_pat_[A-Za-z0-9_]{25,}|-----BEGIN (?:RSA |OPENSSH )?PRIVATE KEY-----',text):raise ValueError('Potential secret: '+rel)
   for match in re.findall(r'drive\.google\.com/drive/(?:mobile/)?folders/([A-Za-z0-9_-]+)',text):
    if match not in ALLOWED_DRIVE:raise ValueError('Private Drive link: '+rel)
   if p.suffix=='.ipynb':
    nb=json.loads(text)
    for c in nb['cells']:
     if c.get('attachments'):raise ValueError('Notebook attachment: '+rel)
     for o in c.get('outputs',[]):
      if set(o.get('data',{}))-{'text/plain'}:raise ValueError('Notebook rich-media output: '+rel)
 return True
def main():
 path=ROOT/'PUBLICATION_SHA256.json'
 expected=json.loads(path.read_text(encoding='utf-8'))['files']
 check(expected)
 for rel,digest in expected.items():
  if hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()!=digest:raise ValueError('Publication file changed: '+rel)
 print(f'PASS: {len(expected)} publication files match SHA-256 and privacy checks. Ignored local data is not part of the publication manifest.')
if __name__=='__main__':main()
