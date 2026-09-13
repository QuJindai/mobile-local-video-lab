"""Qualify the actual FP32 photo-swap models before pinning Android assets.

Reference algorithms: InsightFace MIT Python model_zoo, pinned in the report.
No user photos, embeddings or credentials are read or uploaded by this script.
"""
import argparse
import gc
import hashlib
import json
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import onnx
import onnxruntime as ort
from onnx import numpy_helper

RELEASE = "https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/"
UPSTREAM = "1480e705287bc5d59f923b46c260ec6e3e4150f6"
CATALOG = {"scrfd_2.5g.onnx": 3295067,
           "arcface_w600k_r50.onnx": 174388474,
           "inswapper_128.onnx": 555303150}
TEMPLATE = np.array([[38.2946,51.6963],[73.5318,51.5014],[56.0252,71.7366],
                     [41.5493,92.3655],[70.7299,92.2041]], np.float64)


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def download(url, path, expected_size):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        partial = path.with_suffix(path.suffix + ".partial")
        for attempt in range(4):
            try:
                req = urllib.request.Request(url, headers={"User-Agent":"FACE-RE-model-qualification"})
                with urllib.request.urlopen(req, timeout=60) as source, partial.open("wb") as dest:
                    while block := source.read(1024 * 1024):
                        dest.write(block)
                assert partial.stat().st_size == expected_size, (path.name, partial.stat().st_size)
                partial.replace(path)
                break
            except Exception:
                partial.unlink(missing_ok=True)
                if attempt == 3:
                    raise
    assert path.stat().st_size == expected_size


def session(path):
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 4
    opts.inter_op_num_threads = 1
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    return ort.InferenceSession(str(path), sess_options=opts, providers=["CPUExecutionProvider"])


def tensor(rgb, mean, std):
    return ((rgb.astype(np.float32) - mean) / std).transpose(2,0,1)[None].copy()


def io_info(s):
    return {"inputs":[{"name":x.name,"shape":x.shape,"type":x.type} for x in s.get_inputs()],
            "outputs":[{"name":x.name,"shape":x.shape,"type":x.type} for x in s.get_outputs()]}


def align(rgb, landmarks, size):
    dst = TEMPLATE.copy()
    if size == 128:
        dst[:,0] += 8
    src = np.asarray(landmarks, np.float64)
    s_mean, d_mean = src.mean(axis=0), dst.mean(axis=0)
    s, d = src-s_mean, dst-d_mean
    den = (s*s).sum()
    a = (s*d).sum()/den
    b = (s[:,0]*d[:,1]-s[:,1]*d[:,0]).sum()/den
    rot = np.array([[a,-b],[b,a]])
    mat = np.column_stack([rot, d_mean-rot@s_mean])
    return cv2.warpAffine(rgb, mat, (size,size), flags=cv2.INTER_LINEAR), mat


def detect(s, rgb):
    h,w = rgb.shape[:2]
    scale = min(640/w,640/h,1.)
    rw,rh = max(1,round(w*scale)),max(1,round(h*scale))
    padded = np.zeros((640,640,3),np.uint8)
    padded[:rh,:rw] = cv2.resize(rgb,(rw,rh))
    out = s.run(None, {s.get_inputs()[0].name:tensor(padded,127.5,128.)})
    assert len(out)==9, [x.shape for x in out]
    faces = []
    for i,stride in enumerate((8,16,32)):
        scores=out[i].reshape(-1)
        boxes=out[i+3].reshape(-1,4)*stride
        kps=out[i+6].reshape(-1,5,2)*stride
        assert len(scores)==2*(640//stride)**2
        for n in np.flatnonzero(scores>=.6):
            y,x=divmod(n//2,640//stride)
            center=np.array([x*stride,y*stride])
            box=np.concatenate([center-boxes[n,:2],center+boxes[n,2:]])
            pts=kps[n]+center
            if box[2]<=box[0] or box[3]<=box[1] or box[0]>=rw or box[1]>=rh:
                continue
            factor=np.array([w/rw,h/rh])
            faces.append((float(scores[n]),box*np.tile(factor,2),pts*factor))
    keep=[]
    for face in sorted(faces,key=lambda x:-x[0]):
        _,box,_=face
        reject=False
        for _,old,_ in keep:
            inter=np.maximum(0,np.minimum(box[2:],old[2:])-np.maximum(box[:2],old[:2])).prod()
            union=(box[2:]-box[:2]).prod()+(old[2:]-old[:2]).prod()-inter
            if union>0 and inter/union>.4:
                reject=True
                break
        if not reject:
            keep.append(face)
    assert keep, "No face found in official fixture"
    return sorted(keep,key=lambda x:-(x[1][2]-x[1][0])*(x[1][3]-x[1][1]))


def normalized(x):
    assert np.isfinite(x).all() and np.linalg.norm(x)>1e-8
    return x/np.linalg.norm(x)


def qualify(directory, output):
    output.mkdir(parents=True,exist_ok=True)
    entries=[]
    for name,size in CATALOG.items():
        path=directory/name
        download(RELEASE+name,path,size)
        entry={"name":name,"url":RELEASE+name,"bytes":size,"sha256":sha(path)}
        entries.append(entry)
        print(json.dumps(entry),flush=True)
    emap=np.asarray(numpy_helper.to_array(onnx.load(directory/'inswapper_128.onnx').graph.initializer[-1]),dtype='<f4')
    assert emap.shape==(512,512) and np.isfinite(emap).all()
    emap.tofile(output/'emap.bin')
    fixtures=[]
    for name,size,git_sha in [('Tom_Hanks_54745.png',12123,'906315d13fa29bb3a5ded3e162592f2c7f041b23'),
                              ('t1.jpg',128824,'0d1d64a59675c9590fd12429db647eb169cecff8')]:
        url=f'https://raw.githubusercontent.com/deepinsight/insightface/{UPSTREAM}/python-package/insightface/data/images/{name}'
        path=directory/name
        download(url,path,size)
        raw=path.read_bytes()
        assert hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()==git_sha
        fixtures.append(cv2.cvtColor(cv2.imread(str(path)),cv2.COLOR_BGR2RGB))
    report={"schema_version":1,"runtime":ort.__version__,"provider":"CPUExecutionProvider",
            "upstream_source":UPSTREAM,"fixtures":"official InsightFace test images; largest detected face selected for host test",
            "handset_verified":False,"models":entries,"io":{},"stages":{}}
    det=session(directory/'scrfd_2.5g.onnx')
    report['io']['detector']=io_info(det)
    faces=[detect(det,im) for im in fixtures]
    report['stages']['detected_faces']=[len(x) for x in faces]
    del det;gc.collect()
    aligned=[align(im,f[0][2],112)[0] for im,f in zip(fixtures,faces)]
    rec=session(directory/'arcface_w600k_r50.onnx')
    report['io']['recognizer']=io_info(rec)
    embeddings=[rec.run(None,{rec.get_inputs()[0].name:tensor(im,127.5,127.5)})[0].reshape(-1) for im in aligned]
    assert all(x.shape==(512,) and np.isfinite(x).all() for x in embeddings)
    latents=[normalized(normalized(x)@emap).reshape(1,512).astype(np.float32) for x in embeddings]
    del rec;gc.collect()
    target,mat=align(fixtures[1],faces[1][0][2],128)
    swap=session(directory/'inswapper_128.onnx')
    report['io']['swapper']=io_info(swap)
    assert {x.name for x in swap.get_inputs()}=={'source','target'}
    outputs=[]
    for latent in latents:
        started=time.monotonic()
        y=swap.run(None,{'source':latent,'target':tensor(target,0,255.)})[0]
        assert y.shape==(1,3,128,128) and np.isfinite(y).all()
        outputs.append(y)
        report['stages'].setdefault('swap_seconds',[]).append(time.monotonic()-started)
    delta=float(np.mean(np.abs(outputs[0]-outputs[1])))
    assert delta>1e-4,('identity input did not influence output',delta)
    assert float(outputs[0].std())>.01,'constant output'
    report['stages']['identity_input_mean_absolute_effect']=delta
    report['stages']['source_target_embedding_cosine']=float(normalized(embeddings[0])@normalized(embeddings[1]))
    report['stages']['swapped_crop_std']=float(outputs[0].std())
    fake=(np.clip(outputs[0][0].transpose(1,2,0),0,1)*255).astype(np.uint8)
    cv2.imwrite(str(output/'host-swapped-crop.png'),cv2.cvtColor(fake,cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(output/'host-target-crop.png'),cv2.cvtColor(target,cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(output/'host-source-crop.png'),cv2.cvtColor(aligned[0],cv2.COLOR_RGB2BGR))
    del swap;gc.collect()
    rec=session(directory/'arcface_w600k_r50.onnx')
    result_embedding=rec.run(None,{rec.get_inputs()[0].name:tensor(fake[:112,8:120],127.5,127.5)})[0].reshape(-1)
    report['stages']['source_result_embedding_cosine']=float(normalized(embeddings[0])@normalized(result_embedding))
    report['status']='HOST_INFERENCE_PASS_HANDSET_PENDING'
    (output/'qualification.json').write_text(json.dumps(report,indent=2)+'\n')
    manifest={"schema_version":1,"model_id":"inswapper128-fp32-v1","models":entries,
              "emap_sha256":sha(output/'emap.bin'),"emap_bytes":512*512*4,"io":report['io'],
              "execution":"ONNX Runtime CPU","model_license":"InsightFace pretrained models: non-commercial research"}
    (output/'models.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--directory',type=Path,default=Path('build/face-models'))
    p.add_argument('--output',type=Path,default=Path('build/face-qualification'))
    args=p.parse_args()
    qualify(args.directory,args.output)
