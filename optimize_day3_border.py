"""Test a generic broad border-strip rejection rule on both validation sets."""
import argparse
import copy
import json
from pathlib import Path
import shutil

from optimize_day3 import Search, ROOT, save
from run_test import run


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    search=Search(args.out)
    cfg=json.loads((ROOT/'models/A0_day3_search_accuracy/manifest.json').read_text())
    base=search.evaluate('previous_champion',cfg);search.baseline=base
    # Test neighbouring geometry thresholds, not image IDs or target counts.
    for span,depth in ((.6,.2),(.5,.2),(.7,.2),(.6,.15),(.6,.25)):
        trial=copy.deepcopy(cfg)
        trial['watershed'].update(border_strip_span=span,border_strip_depth=depth)
        search.evaluate(f'border_span_{span}_depth_{depth}',trial)
    # Equal accuracy does not justify a more aggressive background filter.
    # Prefer strips that span more of the border and extend less deeply.
    winner=max((r for r in search.rows if search.eligible(r)),key=lambda r:(
        r['selection_proxy_out_of_15'],r['manifest']['watershed'].get('border_strip_span',0),
        -r['manifest']['watershed'].get('border_strip_depth',1)))
    if winner['selection_proxy_out_of_15'] <= base['selection_proxy_out_of_15']:
        save(args.out/'result.json',{'improved':False,'baseline':base});return
    dest=ROOT/'models/A0_day3_border_refined';dest.mkdir(exist_ok=True)
    for name in ('model.onnx','classifier.onnx'):
        shutil.copy2(ROOT/'models/A0_day3_search_accuracy'/name,dest/name)
    save(dest/'manifest.json',winner['manifest'])
    result={'improved':True,'baseline':base,'winner':winner,'cli_reports':{}}
    for group,root in search.roots.items():
        target=args.out/'final'/group;target.mkdir(parents=True,exist_ok=True)
        with (target/'infer.log').open('w',encoding='utf-8') as log:
            result['cli_reports'][group]=run(dest,target,'valid',4,log,data_root=root)
    save(args.out/'result.json',result)


if __name__=='__main__':main()
