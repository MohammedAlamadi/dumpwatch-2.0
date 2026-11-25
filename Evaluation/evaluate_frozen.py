"""Repeat frozen-model checks without fitting. Existing scientific outputs stay read-only."""
from pathlib import Path
import argparse, ast, csv, hashlib, io, json, math, sys, tempfile, zipfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from project_setup import project_paths, enable_demo_vendor

def read_csv(f):
    with Path(f).open(encoding='utf-8-sig',newline='')as s:return list(csv.DictReader(s))

def sha(f):
    with Path(f).open('rb')as s:return hashlib.file_digest(s,'sha256').hexdigest()

def require(ok,message):
    if not ok:raise ValueError(message)

def classification(rows,labels,truth='truth',prediction='prediction'):
    matrix=[[sum(r[truth]==a and r[prediction]==b for r in rows)for b in labels]for a in labels]
    per={};f1=[]
    for i,label in enumerate(labels):
        tp=matrix[i][i];support=sum(matrix[i]);predicted=sum(r[i]for r in matrix)
        p=tp/predicted if predicted else None;r=tp/support if support else None
        value=2*tp/(support+predicted)if support+predicted else 0
        per[label]=dict(precision=p,recall=r,f1=value,support=support);f1.append(value)
    return dict(n=len(rows),accuracy=sum(matrix[i][i]for i in range(len(labels)))/len(rows),macro_f1=sum(f1)/len(labels),row_labels=labels,column_labels=labels,matrix=matrix,classes=per)

def check_numeric(actual,expected,path=''):
    if isinstance(expected,dict):
        for k,v in expected.items():
            if k in actual:check_numeric(actual[k],v,path+'/'+k)
    elif isinstance(expected,(int,float))and not isinstance(expected,bool):
        require(math.isclose(actual,expected,abs_tol=1e-8,rel_tol=1e-8),f'Metric mismatch {path}: {actual} != {expected}')
    elif isinstance(expected,list):
        require(actual==expected,'Recorded list/matrix mismatch: '+path)

def compare_complete_metrics(actual,expected,path=''):
    """Compare the full fresh Task A schema, including missing keys and nonnumeric fields."""
    if isinstance(expected,dict):
        require(isinstance(actual,dict)and set(actual)==set(expected),'Metric keys differ: '+path)
        for key,value in expected.items():compare_complete_metrics(actual[key],value,path+'/'+key)
    elif isinstance(expected,(int,float))and not isinstance(expected,bool):
        require(math.isclose(actual,expected,abs_tol=1e-8,rel_tol=1e-8),'Fresh metric differs: '+path)
    else:require(actual==expected,'Fresh metric/label differs: '+path)

def notebook_functions(path,cell,names,namespace):
    book=json.loads(path.read_text(encoding='utf-8'))
    tree=ast.parse(''.join(book['cells'][cell]['source']))
    definitions=[node for node in tree.body if isinstance(node,ast.FunctionDef)and node.name in names]
    require({n.name for n in definitions}==set(names),'Original inference function missing')
    exec(compile(ast.Module(body=definitions,type_ignores=[]),str(path),'exec'),namespace)

def task_b(paths,output):
    import numpy as np
    from PIL import Image
    base=ROOT/'Reproduction/Task_B';archive=paths['real_zip']
    model_path=base/'frozen_real_only_svm.json';model=json.loads(model_path.read_text())
    require(sha(model_path)=='b6bc6391b3af0e7ea2999e0702876c1afa980f51f83470fcce4238b430f0b6b5','Selected model differs')
    require(sha(archive)==model['archive_sha256'],'Task B archive differs')
    require(sha(base/'real_location_split.csv')==model['split_sha256'],'Split differs')
    require(sha(base/'frozen_train_development_pair_inventory.csv')==model['pair_config_sha256'],'Pair configuration differs')
    compare_root=ROOT/'Additional_Evidence/Task_B_Model_Comparison'
    spec_path=compare_root/'recorded_run/frozen_new_methods.json';spec=json.loads(spec_path.read_text())
    lock=json.loads((compare_root/'recorded_run/pre_holdout_lock.json').read_text())
    require(sha(spec_path)==lock['model_sha256'],'Frozen MLP specification changed')
    rows=read_csv(base/'frozen_train_development_pair_inventory.csv')
    train=[r for r in rows if r['status']=='usable'and r['split']=='training']
    require(spec['training_ids']==[r['pair_id']for r in train],'MLP training membership differs')
    groups={'development':[dict(r,ground_truth=r['target'])for r in rows if r['status']=='usable'and r['split']=='development'],
            'retrospective':read_csv(base/'frozen_test_pair_inventory.csv')}
    require(sha(base/'frozen_test_pair_inventory.csv')==json.loads((base/'results/final_test_metrics.json').read_text())['test_inventory_sha256'],'Test inventory differs')
    ns=dict(np=np,Image=Image,io=io,zipfile=zipfile,require=require,sha=lambda raw:hashlib.sha256(raw).hexdigest())
    notebook_functions(base/'dumpwatch_task_b_final_evaluation.ipynb',3,['descriptor','paired_vector','select_clearance','frozen_score','frozen_pair_features','box_from_string'],ns)
    notebook_functions(compare_root/'dumpwatch_task_b_comparison.ipynb',6,['crop','features'],ns)
    all_results={};all_predictions=[]
    originals=json.loads((compare_root/'recorded_run/metrics.json').read_text())
    for partition,pairs in groups.items():
        archived=read_csv(compare_root/'recorded_run'/(partition+'_predictions.csv'))
        saved={(r['method'],r['pair_id']):r for r in archived}
        scores=[];fresh=[]
        with zipfile.ZipFile(archive)as z:
            for row in pairs:
                w,a=ns['crop'](z,row,'W'),ns['crop'](z,row,'inspection')
                x=np.asarray([ns['features'](w,a)],dtype=np.float64)
                w1,w2=map(np.asarray,spec['coefficients']);b1,b2=map(np.asarray,spec['intercepts'])
                logit=float((np.tanh(((x-spec['mean'])/spec['scale'])@w1+b1)@w2+b2).ravel()[0])
                descriptors=[ns['descriptor'](z.read(row[key+'_path']),list(map(float,row[key+'_roi_xyxy'].split())))for key in ['W','inspection']]
                margin=float(ns['frozen_score'](ns['frozen_pair_features'](*descriptors,model),model['pair']))
                fixed=str(ns['select_clearance'](margin,True,model['policy']['low'],model['policy']['high']).item())
                methods=[('Image Difference',float(x[0,0]),spec['class_order'][int(x[0,0]>=spec['difference_threshold'])]),
                         ('Frozen Real Only SVM',margin,spec['class_order'][int(margin>=0)]),
                         ('Compact MLP',logit,spec['class_order'][int(logit>=0)]),('Training Majority',None,spec['majority'])]
                for method,score,pred in methods:
                    previous=saved[method,row['pair_id']]
                    require(pred==previous['prediction'],f'Fresh Task B prediction changed: {partition}/{row["pair_id"]}/{method}')
                    if score is not None:require(math.isclose(score,float(previous['score']),abs_tol=1e-7,rel_tol=1e-7),'Fresh Task B score changed')
                    fresh.append(dict(partition=partition,method=method,pair_id=row['pair_id'],truth=row['ground_truth'],prediction=pred,score=score))
                scores.append(dict(truth=row['ground_truth'],prediction=fixed))
        result={}
        for method in ['Image Difference','Frozen Real Only SVM','Compact MLP','Training Majority']:
            subset=[r for r in fresh if r['method']==method]
            result[method]=classification(subset,spec['class_order'])
            check_numeric(result[method],originals[partition][method])
            require(result[method]['matrix']==originals[partition][method]['matrix'],'Task B matrix differs')
        accepted=[r for r in scores if r['prediction']!='abstain']
        result['Frozen SVM fixed policy']=dict(n=len(scores),accepted=len(accepted),correct_accepted=sum(r['truth']==r['prediction']for r in accepted),abstained=len(scores)-len(accepted),coverage=len(accepted)/len(scores),accepted_accuracy=sum(r['truth']==r['prediction']for r in accepted)/len(accepted)if accepted else None)
        all_results[partition]=result;all_predictions.extend(fresh)
        print('Task B',partition,len(pairs),'pairs, four frozen methods: all saved decisions/scores matched.',flush=True)
    (output/'task_b_frozen_metrics.json').write_text(json.dumps(all_results,indent=2),encoding='utf-8')
    (output/'task_b_frozen_predictions.json').write_text(json.dumps(all_predictions,indent=2),encoding='utf-8')
    return dict(scope='Fresh inference: 8 development and 9 previously evaluated holdout pairs, four frozen methods. No fitting. Holdout comparison remains retrospective.',matched=True)

def task_a(paths,output,full=False):
    enable_demo_vendor()
    sys.path.insert(0,str(ROOT/'Reproduction/Task_A'))
    try:
        import task_a as a
        import joblib,torch
        from torchvision.models import resnet18
    except ImportError as e:
        raise RuntimeError('Task A evaluation needs the documented environment. See Evaluation/README.md and requirements-evaluation-windows.txt.')from e
    base=ROOT/'Reproduction/Task_A/evidence';lock=json.loads((base/'model_lock.json').read_text())
    for name,value in lock['locked_files'].items():require(sha(base/name)==value,'Frozen artifact differs: '+name)
    records=[json.loads(l)for l in(base/'holdout_ingest.jsonl').read_text().splitlines()]
    manifest={r['id']:r for r in read_csv(base/'decoded_dataset_manifest.csv')if r['assignment']=='holdout'}
    require(len(records)==100 and set(manifest)=={r['id']for r in records},'Holdout membership differs')
    saved=read_csv(base/'holdout_predictions.csv');metrics=json.loads((base/'holdout_metrics.json').read_text())
    # First check metrics for all 100 clips using archived predictions, separately from fresh inference.
    for method in ['majority','motion_svm','resnet18_gru']:
        byid={r['id']:r for r in saved if r['method']==method}
        prior=[(int(byid[r['id']]['prediction']),float(byid[r['id']]['predicted_onset_s'])if byid[r['id']]['predicted_onset_s']else None,float(byid[r['id']]['score']))for r in records]
        check_numeric(a.measure(records,prior),metrics[method])
    if not full:records=[r for y in [0,1]for r in [q for q in records if q['y']==y][:2]]
    for r in records:
        r['path']=paths['task_a_root']/'VALIDATION_SET/videos'/(r['id']+'.mp4')
        require(sha(r['path'])==manifest[r['id']]['sha256']==r['sha'],'Video hash differs: '+r['id'])
    torch.set_num_threads(2);torch.manual_seed(a.SEED);device=torch.device('cpu')
    weights=ROOT/'Demo/models/resnet18-f37072fd.pth'
    require(sha(weights)=='f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec','Encoder differs')
    net=resnet18(weights=None);net.load_state_dict(torch.load(weights,map_location='cpu',weights_only=True));net.fc=torch.nn.Identity();net=net.eval().requires_grad_(False)
    gru=a.GRU();gru.load_state_dict(torch.load(base/'gru.pt',map_location='cpu',weights_only=True));gru.eval()
    svm=joblib.load(base/'motion_svm.joblib')
    with tempfile.TemporaryDirectory(prefix='dumpwatch_frozen_features_')as tmp:
        a.ingest(records,Path(tmp),net,device,output/'task_a_ingest.jsonl')
        a.evaluate(records,'task_a_fresh',output,lock['majority'],lock['median_onset'],svm['detector'],svm['onset'],gru,device)
    fresh=read_csv(output/'task_a_fresh_predictions.csv');saved={(r['method'],r['id']):r for r in saved}
    for row in fresh:
        original=saved[row['method'],row['id']]
        require(row['prediction']==original['prediction'],'Task A class differs')
        require(row['predicted_onset_s']==original['predicted_onset_s'],'Task A onset differs')
        require(math.isclose(float(row['score']),float(original['score']),abs_tol=1e-4,rel_tol=1e-4),'Task A score differs')
    if full:
        compare_complete_metrics(json.loads((output/'task_a_fresh_metrics.json').read_text()),metrics)
    print('Task A',len(records),'clips x three frozen methods: saved outputs matched. Archived 100-clip metrics also checked.',flush=True)
    return dict(fresh_videos=len(records),fresh_predictions=len(fresh),archived_metrics_videos=100,matched=True,full_fresh_holdout=full)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task',choices=['a','b','both'],default='b')
    parser.add_argument('--full-a',action='store_true',help='Decode all 100 original holdout clips; longer CPU inference, never training.')
    parser.add_argument('--output',type=Path,help='A NEW directory only. Existing directories are refused.')
    args=parser.parse_args();paths=project_paths()
    if args.output:args.output.mkdir(parents=True,exist_ok=False);output=args.output
    else:output=Path(tempfile.mkdtemp(prefix='dumpwatch_frozen_check_'))
    receipt=dict(training_performed=False,source_package=str(ROOT),output=str(output))
    print('New output directory:',output,flush=True)
    try:
        if args.task in ['a','both']:receipt['task_a']=task_a(paths,output,args.full_a)
        if args.task in ['b','both']:receipt['task_b']=task_b(paths,output)
        receipt['status']='PASS'
    except Exception as e:
        receipt.update(status='FAILED',error=type(e).__name__+': '+str(e))
        (output/'check_receipt.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
        raise
    (output/'check_receipt.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    print('PASS. No models fitted; original model and result files unchanged.',flush=True)

if __name__=='__main__':main()
