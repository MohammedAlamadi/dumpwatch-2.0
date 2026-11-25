"""Prepare the eight original demonstration media locally; NEVER upload them."""
from pathlib import Path
import csv,hashlib,json,sys,zipfile
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from project_setup import project_paths
def sha(raw):return hashlib.sha256(raw).hexdigest()
def main():
 paths=project_paths();examples=ROOT/'Demo/examples'
 manifest=json.loads((examples/'manifest.json').read_text(encoding='utf-8'))
 expected=json.loads((examples/'media_checksums.json').read_text(encoding='utf-8'))
 model=json.loads((ROOT/'Reproduction/Task_B/frozen_real_only_svm.json').read_text(encoding='utf-8'))
 with paths['real_zip'].open('rb')as f:
  if hashlib.file_digest(f,'sha256').hexdigest()!=model['archive_sha256']:raise ValueError('Private ZIP is not the frozen original.')
 with(ROOT/'Reproduction/Task_B/frozen_train_development_pair_inventory.csv').open(encoding='utf-8-sig',newline='')as f:pairs={r['pair_id']:r for r in csv.DictReader(f)}
 planned={}
 for key in ('detected','negative'):
  file=manifest[key]['file'];planned[file]=(paths['task_a_root']/'TRAINING_SET/videos'/file).read_bytes()
 with zipfile.ZipFile(paths['real_zip'])as z:
  for key in ('remains','review','review_closure'):
   row=pairs[manifest[key]['pair_id']]
   if row['split']!='development':raise ValueError('Demo must use the original development pairs.')
   for role in ('W','inspection'):planned[manifest[key][role]]=z.read(row[role+'_path'])
 if set(planned)!=set(expected):raise ValueError('Media inventory differs.')
 for name,raw in planned.items():
  if sha(raw)!=expected[name]:raise ValueError('Frozen demonstration media differs: '+name)
  target=examples/name
  if target.exists()and target.read_bytes()!=raw:raise ValueError('Existing local media differs; refusing overwrite: '+name)
 for name,raw in planned.items():
  target=examples/name
  if not target.exists():
   with target.open('xb')as f:f.write(raw)
 print('PASS: 8 exact original demo media prepared locally, all ignored by Git. No source, model or result changed.')
if __name__=='__main__':main()
