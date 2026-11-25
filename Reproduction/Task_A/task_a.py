"""DumpWatch Task A. Run the smoke stage before the full stage."""
import argparse
import csv
import hashlib
import json
import math
import random
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC
import joblib

SEED = 114
FPS = 4
SIDE = 224
BAD_ONSET = "vid0481"
TRAIN, HOLDOUT = "TRAINING_SET", "VALIDATION_SET"


def require(condition, msg):
    if not condition:
        raise ValueError(msg)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf8")


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def inventory(root, partition, smoke, stage=None):
    base = root / partition
    videos = {p.stem: p for p in (base / "videos").glob("*.mp4")}
    labels = {p.stem: p for p in (base / "labels").glob("*.json")}
    expected = 400 if partition == TRAIN else 100
    require(set(videos) == set(labels), f"Missing MP4/JSON pairs in {base}")
    require(len(videos) >= 4 if smoke else len(videos) == expected,
            f"Unexpected video count in {base}: {len(videos)}")
    records = []
    for key, path in sorted(videos.items()):
        require(path.stat().st_size > 0, f"Empty clip: {path}")
        d = json.loads(labels[key].read_text(encoding="utf-8-sig"))
        metadata = d["metadata"]
        require(metadata["video_id"] == path.name, f"Video ID differs: {path}")
        require(d["label"] in ("true", "false"), f"Unknown label: {key}")
        positive = d["label"] == "true"
        event = metadata["event_type"]
        require(event in (("static dumping", "dynamic dumping") if positive else ("normal",)),
                f"Label and modality differ: {key}")
        onset = float(d["timestamp"])
        require(math.isfinite(onset) and (onset >= 0 if positive else onset == -1),
                f"Invalid annotation: {key}")
        records.append(dict(id=key, path=path, y=int(positive), event=event,
                            onset=onset, duration_label=float(metadata["duration"]),
                            resolution=metadata["resolution"], fps_label=float(metadata["frame_rate"]),
                            light=metadata.get("lighting"), tod=metadata.get("time_of_day"),
                            site=metadata.get("camera_id") or metadata.get("site_id")))
    if not smoke:
        require(Counter(r["event"] for r in records) ==
                {"normal": expected//2, "static dumping": expected//4, "dynamic dumping": expected//4},
                f"Unexpected class distribution: {partition}")
    if stage is not None:
        folder=stage/partition/"videos"
        folder.mkdir(parents=True,exist_ok=True)
        def stage_video(r):
            source=r["path"]
            destination=folder/source.name
            size=source.stat().st_size
            if not destination.exists() or destination.stat().st_size!=size:
                temp=destination.with_suffix(".part")
                shutil.copyfile(source,temp)
                require(temp.stat().st_size==size,f"Short Drive read: {source.name}")
                temp.replace(destination)
            r["path"]=destination
        with ThreadPoolExecutor(max_workers=12) as pool:
            for i,_ in enumerate(pool.map(stage_video,records),1):
                if i%50==0 or i==len(records):
                    print(f"Staged {partition}: {i}/{len(records)} clips",flush=True)
    return records


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def probe(path):
    cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
           "stream=width,height,avg_frame_rate,duration:format=duration", "-of", "json", str(path)]
    meta = json.loads(subprocess.check_output(cmd))
    stream = meta["streams"][0]
    duration = float(stream.get("duration") or meta["format"]["duration"])
    return duration, f'{stream["width"]}x{stream["height"]}', stream["avg_frame_rate"]


def decode(path):
    vf = (f"fps={FPS}:start_time=0,scale={SIDE}:{SIDE}:force_original_aspect_ratio=decrease,"
          f"pad={SIDE}:{SIDE}:(ow-iw)/2:(oh-ih)/2")
    command = ["ffmpeg", "-v", "error", "-xerror", "-threads", "2", "-i", str(path), "-map", "0:v:0",
               "-an", "-filter_threads", "1", "-vf", vf, "-pix_fmt", "rgb24", "-f", "rawvideo",
               "-threads", "1", "pipe:1"]
    payload = subprocess.check_output(command)
    require(len(payload) > 0 and len(payload) % (SIDE*SIDE*3) == 0, f"Bad frame stream: {path}")
    return np.frombuffer(payload, dtype=np.uint8).reshape(-1, SIDE, SIDE, 3)


def scene_hash(frame):
    a = np.asarray(Image.fromarray(frame).convert("L").resize((9, 8)), dtype=int)
    bits = np.packbits((a[:, 1:] > a[:, :-1]).ravel())
    return int.from_bytes(bits.tobytes(), "big")


def motion_features(frames):
    gray = frames[:, ::4, ::4].astype(np.float32).mean(3) / 255
    delta = np.abs(np.diff(gray, axis=0, prepend=gray[:1]))
    background = np.abs(gray - np.median(gray[:min(FPS, len(gray))], axis=0))
    def describe(a):
        h,w = a.shape[1:]
        blocks = a.reshape(len(a), 4, h//4, 4, w//4).mean((2,4))
        return np.column_stack([a.mean((1,2)), (a>.10).mean((1,2)),
                                blocks.reshape(len(a),16).max(1),
                                np.quantile(blocks.reshape(len(a),16),.75,axis=1)])
    return np.column_stack((describe(delta),describe(background))).astype(np.float32)


def encoder(device):
    from torchvision.models import resnet18, ResNet18_Weights
    weights = ResNet18_Weights.IMAGENET1K_V1
    net = resnet18(weights=weights)
    net.fc = nn.Identity()
    return net.eval().requires_grad_(False).to(device)


def embeddings(frames, net, device):
    mean = torch.tensor([.485,.456,.406], device=device)[None,:,None,None]
    std = torch.tensor([.229,.224,.225], device=device)[None,:,None,None]
    out = []
    with torch.inference_mode():
        for batch in np.array_split(frames, math.ceil(len(frames)/32)):
            x = torch.from_numpy(batch.copy()).to(device).permute(0,3,1,2).float()/255
            out.append(net((x-mean)/std).cpu().numpy())
    return np.concatenate(out).astype(np.float32)


def ingest(records, cache, net, device, log, motion_only=False):
    for i,r in enumerate(records,1):
        digest = sha(r["path"])
        key = cache / f'{r["id"]}_{digest}_fps4_224_v1.npz'
        cached = False
        if key.exists():
            with np.load(key) as archive:
                cached = motion_only or "emb" in archive.files
        if not cached:
            duration, resolution, native_fps = probe(r["path"])
            frames = decode(r["path"])
            times = np.arange(len(frames), dtype=np.float32)/FPS
            keep = times < duration
            frames, times = frames[keep], times[keep]
            require(len(times) > 0, f"No usable sampled frame: {r['id']}")
            fingerprints = [scene_hash(frames[k]) for k in (0,len(frames)//2,len(frames)-1)]
            payload = dict(times=times, motion=motion_features(frames), duration=duration,
                           native_fps=native_fps, resolution=resolution, fingerprints=np.array(fingerprints,dtype=np.uint64))
            if not motion_only:
                payload["emb"] = embeddings(frames,net,device)
            temporary = key.with_suffix(".tmp.npz")
            np.savez_compressed(temporary,**payload)
            temporary.replace(key)
        with np.load(key) as a:
            duration = float(a["duration"])
            native_fps = str(a["native_fps"])
            fps_parts = native_fps.split("/")
            actual_fps = float(fps_parts[0])/float(fps_parts[1]) if len(fps_parts)==2 and float(fps_parts[1]) else None
            r.update(sha=digest, cache=key, duration=duration, frames=len(a["times"]),
                     fingerprints=[int(v) for v in a["fingerprints"]],
                     decoded_resolution=str(a["resolution"]), decoded_fps=actual_fps)
        r["timing_valid"] = bool(r["y"] and r["id"] != BAD_ONSET and r["onset"] < duration)
        r["timing_exception"] = ("known_invalid_onset" if r["id"] == BAD_ONSET else
                                 "onset_outside_clip" if r["y"] and not r["timing_valid"] else "")
        r["metadata_discrepancy"] = ("duration;" if abs(r["duration_label"]-duration)>.12 else "") + (
            "resolution;" if r["resolution"]!=r["decoded_resolution"] else "") + (
            "fps;" if actual_fps is not None and abs(r["fps_label"]-actual_fps)>.15 else "")
        with log.open("a",encoding="utf8") as f:
            f.write(json.dumps({k:str(v) if isinstance(v,Path) else v for k,v in r.items()})+"\n")
        if i%20==0 or i==len(records):
            print(f"Verified and cached {i}/{len(records)} clips", flush=True)
    return records


def scene_groups(records):
    parent = list(range(len(records)))
    def find(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]]
            i=parent[i]
        return i
    def join(i,j):
        a,b=find(i),find(j)
        if a!=b:
            parent[max(a,b)]=min(a,b)
    for i,a in enumerate(records):
        for j,b in enumerate(records[:i]):
            cues = sorted(min((x^y).bit_count() for y in b["fingerprints"]) for x in a["fingerprints"])
            reverse = sorted(min((x^y).bit_count() for x in a["fingerprints"]) for y in b["fingerprints"])
            if a["sha"]==b["sha"] or (a["site"] and a["site"]==b["site"]) or (cues[1]<=5 and reverse[1]<=5):
                join(i,j)
    return np.array([records[find(i)]["id"] for i in range(len(records))])


def split_records(records):
    groups = scene_groups(records)
    strata = np.array([r["event"] for r in records])
    require(len(set(groups))>=5, "Too few scene groups")
    splitter = StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=SEED)
    choices = list(splitter.split(np.zeros(len(groups)),strata,groups))
    def balance(pair):
        _,dev=pair
        return abs(len(dev)-len(records)/5)+sum(
            abs(sum(strata[dev]==c)-sum(strata==c)/5) for c in sorted(set(strata)))
    acceptable = [p for p in choices if len(set(strata[p[0]]))==3 and len(set(strata[p[1]]))==3]
    require(acceptable,"No class-complete grouped development fold")
    tr,dev = min(acceptable,key=balance)
    require(not set(groups[tr])&set(groups[dev]),"Group leaked")
    for r,g in zip(records,groups):
        r["group"]=str(g)
    return [records[i] for i in tr],[records[i] for i in dev]


def load(r,key):
    with np.load(r["cache"]) as a:
        return a[key]


def clip_motion(r):
    m=load(r,"motion")
    return np.concatenate([m.mean(0),m.std(0),np.quantile(m,.95,axis=0)])


def local_motion(r):
    m=load(r,"motion")
    padded=np.pad(m,((2,2),(0,0)),mode="edge")
    return np.concatenate([padded[i:i+len(m)] for i in range(5)],axis=1)


def fit_svm(train,dev):
    x=np.array([clip_motion(r) for r in train])
    y=[r["y"] for r in train]
    dx=[clip_motion(r) for r in dev]
    dy=[r["y"] for r in dev]
    best=None
    scores={}
    for c in (.1,1.,10.):
        model=make_pipeline(StandardScaler(),SVC(C=c,class_weight="balanced",kernel="rbf"))
        model.fit(x,y)
        score=f1_score(dy,model.predict(dx),average="macro",zero_division=0)
        scores[str(c)]=float(score)
        if best is None or score>best[0]:
            best=(score,c,model)
    features,targets,sample_weights=[],[],[]
    for r in train:
        if r["y"] and not r["timing_valid"]:
            continue
        t=load(r,"times")
        target=(np.abs(t-r["onset"])<=.5).astype(int) if r["y"] else np.zeros(len(t),int)
        features.append(local_motion(r))
        targets.extend(target)
        sample_weights.extend(np.full(len(t),1/len(t)))
    onset=make_pipeline(StandardScaler(),LinearSVC(C=1,class_weight="balanced",random_state=SEED))
    w=np.asarray(sample_weights)
    onset.fit(np.concatenate(features),targets,linearsvc__sample_weight=w/w.mean())
    return best[2],onset,best[1],scores


class GRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn=nn.GRU(512,64,batch_first=True)
        self.event=nn.Linear(64,1)
        self.timing=nn.Linear(64,1)

    def forward(self, x, sizes):
        packed=pack_padded_sequence(x,sizes.cpu(),batch_first=True,enforce_sorted=False)
        hidden,_=self.rnn(packed)
        hidden,_=pad_packed_sequence(hidden,batch_first=True)
        mask=torch.arange(hidden.shape[1],device=x.device)[None,:]<sizes[:,None]
        event=(hidden*mask[:,:,None]).sum(1)/sizes[:,None]
        return self.event(event).squeeze(1),self.timing(hidden).squeeze(2).masked_fill(~mask,-1e9)


def gru_batch(records, device):
    x=[torch.from_numpy(load(r,"emb")) for r in records]
    sizes=torch.tensor([len(a) for a in x],device=device)
    return pad_sequence(x,batch_first=True).to(device),sizes


def gru_predictions(model, records, device):
    model.eval()
    results=[]
    with torch.inference_mode():
        for offset in range(0,len(records),16):
            batch=records[offset:offset+16]
            x,s=gru_batch(batch,device)
            scores,onsets=model(x,s)
            for r,score,index in zip(batch,scores.tolist(),onsets.argmax(1).tolist()):
                results.append((int(score>=0),float(load(r,"times")[index]) if score>=0 else None,score))
    return results


def fit_gru(train,dev,device,epochs):
    model=GRU().to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
    rng=random.Random(SEED)
    best=None
    history=[]
    for epoch in range(epochs):
        model.train()
        order=list(train)
        rng.shuffle(order)
        losses=[]
        for start in range(0,len(order),16):
            batch=order[start:start+16]
            x,s=gru_batch(batch,device)
            cls,loc=model(x,s)
            labels=torch.tensor([r["y"] for r in batch],dtype=torch.float32,device=device)
            loss=nn.functional.binary_cross_entropy_with_logits(cls,labels)
            usable=[i for i,r in enumerate(batch) if r["timing_valid"]]
            if usable:
                nearest=[int(np.argmin(np.abs(load(batch[i],"times")-batch[i]["onset"]))) for i in usable]
                loss=loss+.5*nn.functional.cross_entropy(loc[usable],torch.tensor(nearest,device=device))
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),1.)
            optimizer.step()
            losses.append(float(loss.detach()))
        preds=gru_predictions(model,dev,device)
        metric=measure(dev,preds)
        score=(metric["macro_f1"], metric["event_f1"])
        history.append(dict(epoch=epoch+1,loss=float(np.mean(losses)),macro_f1=score[0],event_f1=score[1]))
        print(history[-1],flush=True)
        if best is None or score>best[0]:
            best=(score,{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},epoch+1)
    model.load_state_dict(best[1])
    return model,history,best[2]


def measure(records, preds):
    y=[r["y"] for r in records]
    p=[v[0] for v in preds]
    tp=fp=fn=0
    errors=[]
    eligible=sum(r["timing_valid"] for r in records)
    for r,(decision,t,_) in zip(records,preds):
        if r["y"] and not r["timing_valid"]:
            continue
        if not r["y"]:
            fp+=int(decision)
        elif t is None:
            fn+=1
        else:
            signed=t-r["onset"]
            errors.append(signed)
            matched=-5<=signed<=10
            tp+=int(matched)
            fp+=int(not matched)
            fn+=int(not matched)
    div=lambda a,b: a/b if b else 0.
    return dict(n=len(y), accuracy=float(accuracy_score(y,p)), macro_f1=float(f1_score(y,p,labels=[0,1],
        average="macro",zero_division=0)), classes=classification_report(y,p,labels=[0,1],
        target_names=["no_dumping","dumping"],output_dict=True,zero_division=0),
        matrix=confusion_matrix(y,p,labels=[0,1]).tolist(),matrix_labels=["no_dumping","dumping"],
        event_window_s=[-5,10], event_tp=tp,event_fp=fp,event_fn=fn,event_precision=div(tp,tp+fp),
        event_recall=div(tp,tp+fn),event_f1=div(2*tp,2*tp+fp+fn),
        timing_eligible=eligible,timing_reported=len(errors),onset_mae_s=float(np.mean(np.abs(errors))) if errors else None,
        onset_median_ae_s=float(np.median(np.abs(errors))) if errors else None,
        signed_mean_error_s=float(np.mean(errors)) if errors else None,
        excluded_timing_ids=[r["id"] for r in records if r["y"] and not r["timing_valid"]])


def evaluate(records, label, directory, majority, median, svm, onset, gru, device):
    def median_for(r):
        return min(median,float(load(r,"times")[-1]))

    def svm_prediction(r):
        x=clip_motion(r)[None,:]
        score=float(svm.decision_function(x)[0])
        decision=int(svm.predict(x)[0])
        idx=int(onset.decision_function(local_motion(r)).argmax()) if decision else None
        return decision,float(load(r,"times")[idx]) if decision else None,score
    methods={
        "majority":[(majority,median_for(r) if majority else None,float(majority)) for r in records],
        "motion_svm":[svm_prediction(r) for r in records],
    }
    if gru is not None:
        methods["resnet18_gru"]=gru_predictions(gru,records,device)
    grouped={name:measure(records,p) for name,p in methods.items()}
    timing_reference=[(1,median_for(r),None) if r["timing_valid"]
                      else (0,None,None) for r in records]
    only_oracle=measure(records,timing_reference)
    grouped["median_onset_oracle_positive_only"]={k:only_oracle[k] for k in
        ("timing_eligible","timing_reported","onset_mae_s","onset_median_ae_s","signed_mean_error_s")}
    rows=[]
    for name,values in methods.items():
        for r,(pred,t,score) in zip(records,values):
            rows.append(dict(method=name,id=r["id"],event=r["event"],lighting=r["light"],time_of_day=r["tod"],
                             actual=r["y"],prediction=pred,score=score,reference_onset_s=r["onset"],
                             onset_valid=r["timing_valid"],predicted_onset_s=t,
                             signed_error_s=t-r["onset"] if t is not None and r["timing_valid"] else None))
    write_csv(directory/f"{label}_predictions.csv",rows)
    errors=[r for r in rows if r["actual"]!=r["prediction"] or
            (r["onset_valid"] and (r["signed_error_s"] is None or not -5<=r["signed_error_s"]<=10))]
    write_json(directory/f"{label}_errors.json",errors)
    write_json(directory/f"{label}_metrics.json",grouped)
    from PIL import ImageDraw
    canvas=Image.new("RGB",(680,210*len(methods)+50),"white")
    draw=ImageDraw.Draw(canvas)
    draw.text((12,7),f"{label}: classification (actual rows, predicted columns)",fill="black")
    for i,(name,p) in enumerate(methods.items()):
        y=39+i*210
        draw.text((12,y),name,fill="black")
        for j,c in enumerate(("No dumping","Dumping")):
            draw.text((220+j*160,y+28),c,fill="black")
        for j,c in enumerate(("No dumping","Dumping")):
            draw.text((12,y+65+45*j),c,fill="black")
            for k,n in enumerate(grouped[name]["matrix"][j]):
                l=205+k*160;t=y+55+j*45
                draw.rectangle((l,t,l+140,t+38),fill="#dcebee" if j==k else "#f5dddd")
                draw.text((l+60,t+10),str(n),fill="black")
    canvas.save(directory/f"{label}_confusion.png")
    print(label, {m:{k:round(grouped[m][k],3) for k in ("accuracy","macro_f1","event_f1")} for m in methods},flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--cache",type=Path,required=True)
    parser.add_argument("--mode",choices=["smoke","full"],required=True)
    parser.add_argument("--motion-only",action="store_true",help="Smoke only: check motion paths without pretrained weights.")
    parser.add_argument("--resume",action="store_true",help="Resume an existing run and reuse its verified model lock.")
    parser.add_argument("--stage",type=Path,help="Stage each partition locally, when first opened.")
    args=parser.parse_args()
    smoke=args.mode=="smoke"
    require(smoke or not args.motion_only,"Full comparison requires all models")
    random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED)
    torch.set_num_threads(2)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic=True
        torch.backends.cudnn.benchmark=False
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.out.mkdir(parents=True,exist_ok=args.resume)
    args.cache.mkdir(parents=True,exist_ok=True)
    print("Device:",device,flush=True)
    from importlib.metadata import version
    libraries=["numpy","Pillow","scikit-learn","joblib","torch"]
    if not args.motion_only:
        libraries.append("torchvision")
    write_json(args.out/"run_config.json",dict(seed=SEED,fps=FPS,device=str(device),mode=args.mode,
        weights="ResNet18 IMAGENET1K_V1",invalid_onset=BAD_ONSET,packages={p:version(p) for p in libraries}))
    source=inventory(args.root,TRAIN,smoke,args.stage)
    net=None if args.motion_only else encoder(device)
    if smoke:
        # Select four clips per class from the full Drive set, or use the small staged local set.
        source=[r for y in (0,1) for r in [q for q in source if q["y"]==y][:4]]
    ingest(source,args.cache,net,device,args.out/"training_ingest.jsonl",args.motion_only)
    if smoke:
        positive_dev=[r for r in source if r["timing_valid"]]
        negative_dev=[r for r in source if not r["y"]]
        require(positive_dev and negative_dev,"Smoke needs a timed positive and a negative")
        dev=[negative_dev[-1],positive_dev[-1]]
        dev_ids={r["id"] for r in dev}
        train=[r for r in source if r["id"] not in dev_ids]
        require(len({r["y"] for r in train})==2 and len({r["y"] for r in dev})==2,
                "Smoke needs at least two videos per class")
    else:
        train,dev=split_records(source)
    write_csv(args.out/"split.csv",[dict(id=r["id"],partition=tag,event=r["event"],sha256=r["sha"],
                scene_group=r.get("group","smoke_only"),onset_valid=r["timing_valid"])
                for tag,subset in (("train",train),("development",dev)) for r in subset])
    majority=Counter(r["y"] for r in train)[1]>Counter(r["y"] for r in train)[0]
    median=float(np.median([r["onset"] for r in train if r["timing_valid"]]))
    lock_path=args.out/"model_lock.json"
    gru=None
    if args.resume and lock_path.is_file():
        settings=json.loads(lock_path.read_text(encoding="utf8"))
        require(settings["mode"]==args.mode and all(sha(args.out/name)==digest
                for name,digest in settings["locked_files"].items()),"Resume model/split checksum mismatch")
        saved=joblib.load(args.out/"motion_svm.joblib")
        svm,onset=saved["detector"],saved["onset"]
        if not args.motion_only:
            gru=GRU().to(device)
            gru.load_state_dict(torch.load(args.out/"gru.pt",map_location=device,weights_only=True))
        print("Reused verified model lock",flush=True)
    else:
        svm,onset,c,scores=fit_svm(train,dev)
        joblib.dump({"detector":svm,"onset":onset},args.out/"motion_svm.joblib")
        settings=dict(majority=int(majority),median_onset=median,svm_C=c,svm_development_grid=scores,
                      classification_threshold=0,event_window=[-5,10],mode=args.mode)
        if not args.motion_only:
            gru,log,best=fit_gru(train,dev,device,2 if smoke else 16)
            torch.save(gru.state_dict(),args.out/"gru.pt")
            write_json(args.out/"gru_history.json",log)
            settings["gru_selected_epoch"]=best
        settings["locked_files"]={p.name:sha(p) for p in args.out.iterdir()
                                  if p.name in ("split.csv","motion_svm.joblib","gru.pt")}
        write_json(lock_path,settings)
    evaluate(dev,"development",args.out,int(majority),median,svm,onset,gru,device)
    if not smoke:
        # The 100-video folder is first enumerated after all models and choices are locked.
        hold=inventory(args.root,HOLDOUT,False,args.stage)
        require(not {r["id"] for r in source}&{r["id"] for r in hold},"Cross-partition video ID")
        ingest(hold,args.cache,net,device,args.out/"holdout_ingest.jsonl")
        require(not {r["sha"] for r in source}&{r["sha"] for r in hold},"Cross-partition byte duplicate")
        overlaps=[]
        for b in hold:
            for a in source:
                distance=sorted(min((x^y).bit_count() for y in b["fingerprints"]) for x in a["fingerprints"])[1]
                reverse=sorted(min((x^y).bit_count() for x in a["fingerprints"]) for y in b["fingerprints"])[1]
                if distance<=5 and reverse<=5 or (a["site"] and a["site"]==b["site"]):
                    overlaps.append([a["id"],b["id"]])
        write_json(args.out/"holdout_scene_overlap_candidates.json",overlaps)
        evaluate(hold,"holdout",args.out,int(majority),median,svm,onset,gru,device)


if __name__=="__main__":
    main()
