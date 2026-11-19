"""Refine the fast 384 candidate until three rounds show <0.05 / 15 gain."""
import argparse
import copy
import json

import onnxruntime as ort

from optimize_day3 import Search, save


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',required=True)
    args=p.parse_args()
    from pathlib import Path
    out=Path(args.out)
    search=Search(out)
    search.rows=json.loads((out/'search_history.json').read_text())
    search.baseline=next(r for r in search.rows if r['name']=='baseline_full')
    seed=next(r for r in search.rows if r['name']=='resolution_384')
    branch=[seed];seen=set();stale=0
    opts=ort.SessionOptions();opts.intra_op_num_threads=4;opts.inter_op_num_threads=1
    session=ort.InferenceSession(seed['onnx'],opts)
    def best():return max(branch,key=lambda r:r['selection_proxy_out_of_15'])
    for round_id in range(1,9):
        before=best()['selection_proxy_out_of_15']
        for key,deltas in [('foreground',(-.05,.05)),('boundary',(-.05,.05)),
                           ('min_marker',(-2,2)),('min_area',(-2,2))]:
            cfg=copy.deepcopy(best()['manifest'])
            for delta in deltas:
                trial=copy.deepcopy(cfg)
                value=round(trial['watershed'][key]+delta/(2**((round_id-1)//2)),4)
                trial['watershed'][key]=max(1,int(value)) if key.startswith('min_') else value
                fingerprint=json.dumps(trial,sort_keys=True)
                if fingerprint in seen:continue
                seen.add(fingerprint)
                row=search.evaluate(f'fast_round{round_id}_{key}_{trial["watershed"][key]}',trial,session)
                row['onnx']=seed['onnx'];branch.append(row)
                save(out/'search_history.json',search.rows)
        gain=best()['selection_proxy_out_of_15']-before
        stale=stale+1 if gain<.05 else 0
        save(out/'lowres_status.json',{'round':round_id,'gain':gain,'stale_rounds':stale,
             'best':best()['name']})
        if stale>=3:break
    search.finalize()


if __name__=='__main__':main()
