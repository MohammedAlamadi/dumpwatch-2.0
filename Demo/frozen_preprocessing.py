"""Inference-only functions copied from Task A and the frozen Task B notebook."""
import io
import json
import math
import subprocess
import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from PIL import Image

FPS=4
SIDE=224

def require(condition, msg):
    if not condition:
        raise ValueError(msg)

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

def embeddings(frames, net, device):
    mean = torch.tensor([.485,.456,.406], device=device)[None,:,None,None]
    std = torch.tensor([.229,.224,.225], device=device)[None,:,None,None]
    out = []
    with torch.inference_mode():
        for batch in np.array_split(frames, math.ceil(len(frames)/32)):
            x = torch.from_numpy(batch.copy()).to(device).permute(0,3,1,2).float()/255
            out.append(net((x-mean)/std).cpu().numpy())
    return np.concatenate(out).astype(np.float32)

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

def descriptor(raw, box):
    left,top,right,bottom=box
    require(0<=left<right<=1 and 0<=top<bottom<=1, f"Invalid normalized ROI {box}")
    with Image.open(io.BytesIO(raw)) as source:
        w,h=source.size
        pixels=source.convert("RGB").crop((round(left*w),round(top*h),round(right*w),round(bottom*h)))
    require(min(pixels.size)>=64, "ROI too small to assess")
    rgb=np.asarray(pixels.resize((128,128),Image.Resampling.BILINEAR),dtype=np.float32)/255
    gray=rgb @ np.array([.299,.587,.114],dtype=np.float32)
    dy,dx=np.gradient(gray)
    magnitude=np.hypot(dx,dy)
    orientation=np.floor((np.arctan2(dy,dx)%np.pi)*9/np.pi).astype(int).clip(0,8)
    features=[]
    for y in range(0,128,32):
        for x in range(0,128,32):
            region=np.s_[y:y+32,x:x+32]
            hist=np.bincount(orientation[region].ravel(),weights=magnitude[region].ravel(),minlength=9)
            features.extend(hist/(np.linalg.norm(hist)+1e-6))
            patch=rgb[region]
            features.extend(patch.mean(axis=(0,1)))
            features.extend(patch.std(axis=(0,1)))
    return np.asarray(features,dtype=np.float32)

def paired_vector(w,a,scores):
    return np.r_[w,a,np.abs(w-a),scores[0],scores[1],scores[0]-scores[1]]

def select_clearance(score, visually_assessable, low, high):
    require(np.isfinite(low) and np.isfinite(high) and low<high,"Invalid abstention interval")
    return np.where(~np.asarray(visually_assessable,dtype=bool),"abstain",
           np.where(score>=high,"fully_cleared",np.where(score<=low,"waste_still_present","abstain")))

def frozen_score(values,spec):
    values=np.array(values,copy=True)
    # Match StandardScaler's in-place float32 rounding for appearance inputs.
    values-=np.asarray(spec["mean"])
    values/=np.asarray(spec["scale"])
    return values @ np.asarray(spec["coefficients"])+spec["intercept"]

def frozen_pair_features(w,a,model):
    scores=frozen_score(np.stack((w,a)),model["appearance"])
    return paired_vector(w,a,scores)
