"""Interleave repeated complete CLI runs; report cold and per-image timing separately."""
import argparse
import csv
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

from run_test import ROOT, evaluate


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--candidate',nargs='+',choices=('original','accuracy','speed','border'))
    p.add_argument('--reference-ref',default='8f82104209d7c6c6f215152685948db49e678caa',
                   help='Pinned pre-optimization Git commit for a stable baseline')
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    reference=args.out/'reference';reference.mkdir(exist_ok=True)
    for name in ('infer.py','post.py','data.py'):
        (reference/name).write_bytes(subprocess.check_output(['git','show',f'{args.reference_ref}:{name}'],cwd=ROOT))
    candidates={
        'original':(reference/'infer.py',ROOT/'models/A0_classifier_day3'),
        'accuracy':(ROOT/'infer.py',ROOT/'models/A0_day3_search_accuracy'),
        'speed':(ROOT/'infer.py',ROOT/'models/A0_day3_search_speed'),
    }
    if (ROOT/'models/A0_day3_border_refined/manifest.json').is_file():
        candidates['border']=(ROOT/'infer.py',ROOT/'models/A0_day3_border_refined')
    if args.candidate:
        candidates={name:candidates[name] for name in args.candidate}
    roots={'legacy':ROOT,'triangle':ROOT/'datasets/day3_triangle'}
    records=[]
    env=dict(os.environ,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
    for repeat in range(args.repeats):
        order=list(candidates)
        order=order[repeat%len(order):]+order[:repeat%len(order)]
        for name in order:
            script,weights=candidates[name]
            for group,root in roots.items():
                target=args.out/name/group/str(repeat+1);target.mkdir(parents=True,exist_ok=True)
                with (target/'infer.log').open('w',encoding='utf-8') as log:
                    start=time.perf_counter()
                    subprocess.run([sys.executable,str(script),'--input',str(root/'valid'),
                        '--weights',str(weights),'--out',str(target),'--threads','4'],
                        check=True,env=env,stdout=log,stderr=subprocess.STDOUT)
                    elapsed=time.perf_counter()-start
                metrics=evaluate(target,'valid',root)
                stats=json.loads((target/'benchmark.json').read_text())
                row={'candidate':name,'group':group,'repeat':repeat+1,
                     'iou':metrics['instance_iou'],'counting':metrics['matched_counting'],
                     'classification_f1_unmodified':metrics['class_aware_f1_at_iou_50'],
                     'cli_seconds_per_image':elapsed/metrics['images'],
                     'median_per_image_seconds':stats['median_seconds']}
                print(json.dumps(row),flush=True);records.append(row)
                (args.out/'repeated_runs.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
    summary=[]
    for name in candidates:
        for group in roots:
            values=[r for r in records if r['candidate']==name and r['group']==group]
            # Metrics must repeat exactly; timing may vary.
            assert len({r['iou'] for r in values})==1
            assert len({r['counting'] for r in values})==1
            summary.append({'candidate':name,'group':group,'repeats':len(values),
                **{key:statistics.median(r[key] for r in values) for key in
                   ('iou','counting','classification_f1_unmodified','cli_seconds_per_image','median_per_image_seconds')},
                'cli_min_s':min(r['cli_seconds_per_image'] for r in values),
                'cli_max_s':max(r['cli_seconds_per_image'] for r in values)})
    with (args.out/'summary.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(summary[0]));writer.writeheader();writer.writerows(summary)
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
