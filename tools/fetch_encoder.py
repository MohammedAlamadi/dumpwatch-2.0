"""Obtain the original official encoder, with full pinned checksum verification."""
from pathlib import Path
import argparse,hashlib,os,shutil,tempfile,urllib.request
ROOT=Path(__file__).resolve().parents[1]
DIGEST='f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec'
URL='https://download.pytorch.org/models/resnet18-f37072fd.pth'
def sha(p):
 with Path(p).open('rb')as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--from-file',type=Path,help='Copy an existing authorized official checkpoint, offline.')
 args=parser.parse_args();target=ROOT/'Demo/models/resnet18-f37072fd.pth'
 if target.exists():
  if sha(target)!=DIGEST:raise ValueError('Existing encoder differs. It will not be overwritten.')
  print('PASS: existing original encoder checksum.');return
 target.parent.mkdir(parents=True,exist_ok=True)
 fd,name=tempfile.mkstemp(prefix='encoder-',dir=target.parent);os.close(fd);tmp=Path(name)
 try:
  if args.from_file:shutil.copyfile(args.from_file,tmp)
  else:
   with urllib.request.urlopen(URL,timeout=60)as response,tmp.open('wb')as out:shutil.copyfileobj(response,out)
  if sha(tmp)!=DIGEST:raise ValueError('Encoder checksum mismatch; nothing installed.')
  # Exclusive destination creation; never replace an existing checkpoint.
  with tmp.open('rb')as src,target.open('xb')as out:shutil.copyfileobj(src,out)
  print('PASS: original official encoder installed and hash-verified; ignored by Git.')
 finally:tmp.unlink(missing_ok=True)
if __name__=='__main__':main()
