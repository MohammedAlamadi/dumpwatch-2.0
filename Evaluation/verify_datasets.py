"""Read-only verification of configured original data against the locked manifests. No fitting or extraction."""
from pathlib import Path
from collections import Counter,defaultdict
import concurrent.futures,csv,hashlib,json,sys,zipfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from project_setup import project_paths
P=project_paths()
def rows(f):
 with Path(f).open(encoding='utf-8-sig')as s:return list(csv.DictReader(s))
def sha(f):
 with f.open('rb')as s:return hashlib.file_digest(s,'sha256').hexdigest()
a=rows(ROOT/'Reproduction/Task_A/evidence/decoded_dataset_manifest.csv')
def verify_a(r):
 part=r['source_partition'];root=P['task_a_root']/part
 video=root/'videos'/(r['id']+'.mp4');label=root/'labels'/(r['id']+'.json')
 j=json.loads(label.read_text(encoding='utf-8-sig'))
 assert sha(video)==r['sha256'],r['id']
 assert j['metadata']['event_type']==r['event']and int(j['label']=='true')==int(r['dumping'])and float(j['timestamp'])==float(r['annotated_onset_seconds'])
 return video.stat().st_size
with concurrent.futures.ThreadPoolExecutor(max_workers=4)as pool:sizes=list(pool.map(verify_a,a))
splits={s:[r for r in a if r['assignment']==s]for s in ['train','development','holdout']}
assert not {r['scene_group']for r in splits['train']}&{r['scene_group']for r in splits['development']}
assert len({r['id']for r in a})==len({r['sha256']for r in a})==500
aclass={s:dict(Counter('dumping'if r['dumping']=='1'else'no_dumping'for r in rr))for s,rr in splits.items()}
pair_base=ROOT/'Reproduction/Task_B'
dev=[r for r in rows(pair_base/'frozen_train_development_pair_inventory.csv')if r['status']=='usable']
test=rows(pair_base/'frozen_test_pair_inventory.csv')
brows={s:[r for r in dev if r['split']==s]for s in ['training','development']};brows['test']=test
bclass={s:dict(Counter(r.get('target',r.get('ground_truth'))for r in rr))for s,rr in brows.items()}
location_sets={s:{r['location_id']for r in rr}for s,rr in brows.items()}
assert all(not location_sets[s]&location_sets[t]for s in location_sets for t in location_sets if s!=t)
image_sets={s:{r[k]for r in rr for k in ['W_sha256','inspection_sha256']}for s,rr in brows.items()}
assert all(not image_sets[s]&image_sets[t]for s in image_sets for t in image_sets if s!=t)
archive=P['real_zip'];assert sha(archive)=='1e97cfe51b4ef7fc9e61aaa7cbd2cbce4ac5a24fd274f07503d7deb5138f0412'
with zipfile.ZipFile(archive)as z:
 manifest=list(csv.DictReader(__import__('io').StringIO(z.read('dataset_manifest.csv').decode('utf-8-sig'))))
 jpegs=[r for r in manifest if r['relative_path'].lower().endswith(('.jpg','.jpeg'))]
 for r in jpegs:assert hashlib.sha256(z.read(r['relative_path'])).hexdigest()==r['file_hash']
plan=json.loads((ROOT/'Data/Manifests/Task_B/synthetic_v1_allowlist.json').read_text())
selected=[]
for r in plan['cases']:
 if r['case']!='case_008':continue
 for item in [r['metadata'],r['images']['W'],plan['baselines'][r['location']]]:
  parts=item['path'].split('/')
  if P['synthetic_layout']=='archive_mirror':parts[0]=f"location_{int(parts[0].split()[-1]):02}"
  f=P['synthetic_root'].joinpath(*parts)
  assert f.stat().st_size==item['size']and sha(f)==item['sha256']
  selected.append(f)
overlap=json.loads((ROOT/'Reproduction/Task_A/evidence/holdout_scene_overlap_candidates.json').read_text())
result=dict(task_a=dict(videos_verified=500,annotation_labels_verified=500,video_bytes=sum(sizes),classes_by_split=aclass,
 scene_groups_disjoint_train_development=True,unique_file_hashes=500,possible_train_holdout_scene_pairs=len(overlap),
 holdout_clips_in_possible_scene_pairs=len({r[1]for r in overlap}),verified_unseen_camera_split=False),
 task_b=dict(real_jpegs_verified=len(jpegs),archive_bytes=archive.stat().st_size,classes_by_split=bclass,
 location_disjoint=True,exact_image_hashes_disjoint=True,selected_synthetic_source_files_verified=len(selected),
 shared_case_or_view_independence_not_claimed=True),
 permissions=dict(note='This command verifies bytes and labels, not legal permissions. See Data/Access_And_Permissions.md.'),
 source='Counts computed from unchanged packaged manifests; raw file and annotation checks performed against configured original inputs.')
print(json.dumps(result,indent=2))
