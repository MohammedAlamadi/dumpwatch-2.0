"""Verify marking data only: hashes, labels and locked inventories. No model imports or inference."""
from pathlib import Path
import argparse,csv,hashlib,io,json,sys,zipfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from project_setup import project_paths

def require(condition,message):
    if not condition:raise ValueError(message)

def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def rows(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as stream:return list(csv.DictReader(stream))

def task_a(paths,scope):
    manifest=rows(ROOT/'Reproduction/Task_A/evidence/decoded_dataset_manifest.csv')
    require(len(manifest)==500 and len({r['id'] for r in manifest})==500,'Locked Task A manifest differs')
    selected=manifest if scope=='all' else [r for r in manifest if r['assignment']=='holdout']
    require(len(selected)==(500 if scope=='all' else 100),'Task A membership differs')
    for r in selected:
        base=paths['task_a_root']/r['source_partition']
        video=base/'videos'/(r['id']+'.mp4');annotation=base/'labels'/(r['id']+'.json')
        require(sha(video)==r['sha256'],'Video hash mismatch: '+str(video))
        label=json.loads(annotation.read_text(encoding='utf-8-sig'))
        require(int(label['label']=='true')==int(r['dumping']),'Class differs: '+r['id'])
        require(label['metadata']['event_type']==r['event'],'Event differs: '+r['id'])
        require(float(label['timestamp'])==float(r['annotated_onset_seconds']),'Onset differs: '+r['id'])
    return dict(videos=len(selected),annotation_targets=len(selected),scope=scope,matched=True,
                annotation_check='Class, event and onset semantics; JSON byte identity is not asserted')

def task_b(paths):
    base=ROOT/'Reproduction/Task_B';model=json.loads((base/'frozen_real_only_svm.json').read_text(encoding='utf-8'))
    archive=paths['real_zip'];require(sha(archive)==model['archive_sha256'],'Real ZIP hash differs')
    require(sha(base/'real_location_split.csv')==model['split_sha256'],'Location split differs')
    require(sha(base/'frozen_train_development_pair_inventory.csv')==model['pair_config_sha256'],'Reviewed training/development inventory differs')
    metrics=json.loads((base/'results/final_test_metrics.json').read_text(encoding='utf-8'))
    require(sha(base/'frozen_test_pair_inventory.csv')==metrics['test_inventory_sha256'],'Reviewed test inventory differs')
    development=[r for r in rows(base/'frozen_train_development_pair_inventory.csv') if r['status']=='usable']
    test=rows(base/'frozen_test_pair_inventory.csv')
    require(len(development)==28 and len(test)==9,'Pair counts differ')
    counts={split:sum(r['split']==split for r in development)for split in ['training','development']}
    require(counts==dict(training=20,development=8),'Training/development counts differ')
    with zipfile.ZipFile(archive) as z:
        manifest=list(csv.DictReader(io.StringIO(z.read('dataset_manifest.csv').decode('utf-8-sig'))))
        images=[r for r in manifest if r['relative_path'].lower().endswith(('.jpg','.jpeg'))]
        require(len(images)==149,'Expected 149 manifest JPEGs')
        hashes={r['relative_path']:hashlib.sha256(z.read(r['relative_path'])).hexdigest()for r in images}
        for r in images:require(hashes[r['relative_path']]==r['file_hash'],'Image differs: '+r['relative_path'])
        for r in development+test:
            for key in ['W','inspection']:
                require(hashes.get(r[key+'_path'])==r[key+'_sha256'],'Pair image missing or differs: '+r['pair_id'])
    return dict(jpegs=149,training_pairs=20,development_pairs=8,test_pairs=9,matched=True)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task',choices=['a','b','both'],default='both')
    parser.add_argument('--a-scope',choices=['holdout','all'],default='holdout')
    args=parser.parse_args();paths=project_paths()
    result=dict(training=False,inference=False,files_modified=False,synthetic_data_required=False)
    try:
        if args.task in ['a','both']:result['task_a']=task_a(paths,args.a_scope)
        if args.task in ['b','both']:result['task_b']=task_b(paths)
        result['status']='PASS';print(json.dumps(result,indent=2));return 0
    except Exception as e:
        result.update(status='FAIL',error=type(e).__name__+': '+str(e));print(json.dumps(result,indent=2));return 1

if __name__=='__main__':sys.exit(main())
