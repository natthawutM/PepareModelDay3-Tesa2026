"""Compare resolution and selective INT8 after optimize_day3.py has plateaued."""
import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static

from data import Coco, letterbox, normalize
from optimize_day3 import Search, ROOT, save


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    search=Search(args.out)
    search.rows=json.loads((args.out/'search_history.json').read_text())
    for row in search.rows:
        row.setdefault('timing_backend','CUDA' if row['name'].startswith('finetune') else 'CPU')
    search.baseline=next(r for r in search.rows if r['name']=='baseline_full')
    champion=search.best()
    origin=Path(champion.get('onnx',ROOT/'models/A0_classifier_day3/model.onnx'))
    sys.path.append(str(ROOT.parent/'overnight_01/src'))
    from model import Segmenter
    state=torch.load(champion.get('checkpoint',ROOT/'checkpoints/A0/best.pt'),map_location='cpu')
    model=Segmenter(widths=state['config']['widths'])
    model.load_state_dict(state['model']);model.eval()
    torch.set_num_threads(4)
    opts=ort.SessionOptions();opts.intra_op_num_threads=4;opts.inter_op_num_threads=1
    if champion.get('timing_backend')=='CUDA':
        row=search.evaluate('finetune_onnx_verified',champion['manifest'],ort.InferenceSession(str(origin),opts))
        row['onnx']=str(origin)
    # Static ONNX shapes must match the requested resolution.
    for size in (384,448,576,640):
        file=args.out/f'resolution_{size}.onnx'
        torch.onnx.export(model,(torch.zeros(1,3,size,size),),file,dynamo=False,
            opset_version=17,input_names=['image'],output_names=['logits'])
        cfg=copy.deepcopy(champion['manifest']);cfg['variants']['selected']['size']=size
        for key in ('min_area','min_marker'):
            cfg['watershed'][key]=max(1,round(cfg['watershed'][key]*(size/512)**2))
        row=search.evaluate(f'resolution_{size}',cfg,ort.InferenceSession(str(file),opts))
        row['onnx']=str(file)
        save(args.out/'search_history.json',search.rows)
    # Use Train only for calibration. Keep sensitive decoder/head in FP32 in
    # the encoder-only trial, and compare with Conv-only per-channel PTQ.
    source=Coco(ROOT.parent/'coffee_ready_reviewed_v1/coco','train')
    adaptation=Coco(ROOT.parent/'day3_classifier_split_v1','train')
    calibration=[]
    for dataset,limit in ((source,20),(adaptation,5)):
        for i in np.linspace(0,len(dataset.images)-1,min(limit,len(dataset.images)),dtype=int):
            image,_=dataset.load(int(i))
            calibration.append({'image':normalize(letterbox(image,512)[0])[None]})
    class Reader(CalibrationDataReader):
        def __init__(self):self.it=iter(calibration)
        def get_next(self):return next(self.it,None)
    graph=onnx.load(str(origin))
    for label,scope in (('encoder','/encoder'),('conv_except_head','/')):
        nodes=[n.name for n in graph.graph.node if n.op_type=='Conv'
               and n.name.startswith(scope) and not n.name.startswith('/head')]
        file=args.out/f'int8_{label}.onnx'
        quantize_static(str(origin),str(file),Reader(),quant_format=QuantFormat.QDQ,
            per_channel=True,weight_type=QuantType.QInt8,activation_type=QuantType.QUInt8,
            nodes_to_quantize=nodes,op_types_to_quantize=['Conv'])
        row=search.evaluate(f'int8_{label}',champion['manifest'],ort.InferenceSession(str(file),opts))
        row['onnx']=str(file)
        save(args.out/'search_history.json',search.rows)
    search.finalize()


if __name__=='__main__':main()
