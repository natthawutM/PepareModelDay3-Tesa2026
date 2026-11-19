"""Reproducible Day 3 segmentation/latency search; never uses Test for selection.

Windows timings are not Pi timings. Scores here are selection proxies, not an
official Day 3 score: classification and MQTT are deliberately held fixed.
"""
import argparse
import contextlib
import copy
import csv
import hashlib
import io
import json
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from data import Coco, letterbox, normalize, targets
from infer import Predictor, submission_instances
from post import match
from run_test import ROOT, run


def save(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2), encoding='utf-8')
    tmp.replace(path)


class Search:
    def __init__(self, out):
        self.out = out
        out.mkdir(parents=True, exist_ok=True)
        self.rows = []
        self.scenes = {}
        self.roots = {'legacy': ROOT, 'triangle': ROOT / 'datasets/day3_triangle'}
        for name, root in self.roots.items():
            source = Coco(root, 'valid')
            self.scenes[name] = [(info['file_name'], *source.load(i))
                                 for i, info in enumerate(source.images)]
        self.base_manifest = json.loads((ROOT/'models/A0_classifier_day3/manifest.json').read_text())
        self.predictor = Predictor(ROOT/'models/A0_classifier_day3', 'selected', 4)
        self.initial_session = self.predictor.segmenter
        self.seen = set()

    def evaluate(self, name, manifest, session=None):
        self.predictor.manifest = copy.deepcopy(manifest)
        self.predictor.roi_margin = manifest.get('roi_margin')
        self.predictor.size = manifest['variants']['selected']['size']
        self.predictor.segmenter = session if session is not None else self.initial_session
        metrics = {}
        for group, scenes in self.scenes.items():
            results, times = [], []
            for filename, image, truth in scenes:
                start = time.perf_counter()
                labels, classes, regions = self.predictor.predict_instances(image)
                with contextlib.redirect_stderr(io.StringIO()):
                    labels, objects, _ = submission_instances(labels, classes, regions, filename)
                times.append(time.perf_counter()-start)
                results.append(match(labels, truth))
            metrics[group] = {
                'iou': float(np.mean([r['instance_iou'] for r in results])),
                'counting': float(np.mean([r['counting'] for r in results])),
                'warm_compute_median_s': statistics.median(times),
                'images': len(scenes),
            }
        score = float(np.mean([7.5*(m['iou']+m['counting']) for m in metrics.values()]))
        speed = max(m['warm_compute_median_s'] for m in metrics.values())
        row = {'name': name, 'selection_proxy_out_of_15': score,
               'timing_backend': 'CUDA' if session is not None and type(session).__name__ == 'TorchSession' else 'CPU',
               'windows_warm_compute_max_group_median_s': speed,
               'metrics': metrics, 'manifest': copy.deepcopy(manifest)}
        self.rows.append(row)
        save(self.out/'search_history.json', self.rows)
        print(json.dumps({k:v for k,v in row.items() if k!='manifest'}), flush=True)
        return row

    def eligible(self, row):
        return all(row['metrics'][g][k] >= self.baseline['metrics'][g][k] - .02
                   for g in self.scenes for k in ('iou','counting'))

    def best(self):
        return max((r for r in self.rows if self.eligible(r)
                    and (not r['name'].startswith('finetune') or 'onnx' in r)),
                   key=lambda r: (r['selection_proxy_out_of_15'],
                                  -r['windows_warm_compute_max_group_median_s']))

    def sweep(self):
        self.baseline = self.evaluate('baseline_full', self.base_manifest)
        for margin in (.15, .30, .50):
            for locator in (0, 512):
                cfg = copy.deepcopy(self.base_manifest)
                cfg.update(roi_margin=margin, roi_locator_side=locator)
                self.evaluate(f'roi_{margin}_locator_{locator}', cfg)
        # Coordinate search narrows around the current best. Stop after three
        # complete rounds without a >= 0.05 / 15 improvement (not timing noise).
        stale = 0
        for round_id in range(1, 7):
            start_score = self.best()['selection_proxy_out_of_15']
            for key, deltas in [('foreground',(-.05,.05)), ('boundary',(-.05,.05)),
                                ('min_marker',(-4,4)), ('min_area',(-4,4))]:
                center = copy.deepcopy(self.best()['manifest'])
                for delta in deltas:
                    cfg = copy.deepcopy(center)
                    cfg['watershed'][key] = round(cfg['watershed'][key] + delta/(2**((round_id-1)//2)), 4)
                    if key.startswith('min_'):
                        cfg['watershed'][key] = max(1, int(cfg['watershed'][key]))
                    fingerprint = json.dumps(cfg, sort_keys=True)
                    if fingerprint in self.seen:
                        continue
                    self.seen.add(fingerprint)
                    self.evaluate(f'round{round_id}_{key}_{cfg["watershed"][key]}', cfg)
            gain = self.best()['selection_proxy_out_of_15']-start_score
            stale = stale+1 if gain < .05 else 0
            save(self.out/'search_status.json', {'phase':'watershed','round':round_id,
                 'gain':gain,'stale_rounds':stale,'best':self.best()['name']})
            if stale >= 3:
                break

    def train(self, train_root, adapt_root):
        import torch
        import torch.nn.functional as F
        from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
        sys.path.append(str(ROOT.parent/'overnight_01/src'))
        from model import Segmenter
        torch.set_num_threads(4)
        torch.manual_seed(41)
        if not torch.cuda.is_available():
            raise RuntimeError('Training requires the available CUDA environment')
        valid_hashes = {hashlib.sha256((root/'valid'/info['file_name']).read_bytes()).digest()
                        for root in self.roots.values() for info in Coco(root,'valid').images}
        xs, ys, vs, shares = [], [], [], []
        audit = []
        for root, share in ((train_root,.75),(adapt_root,.25)):
            source = Coco(root,'train')
            for i, info in enumerate(source.images):
                digest = hashlib.sha256((source.folder/info['file_name']).read_bytes()).digest()
                if digest in valid_hashes:
                    raise ValueError(f'Train/Valid byte duplicate: {info["file_name"]}')
                image, instances = source.load(i)
                image, box = letterbox(image,512)
                target, valid = targets(instances,box,512)
                xs.append(normalize(image)); ys.append(target); vs.append(valid)
                shares.append(share/len(source.images))
                audit.append({'root':str(root),'file':info['file_name'],'sha256':digest.hex()})
        save(self.out/'train_audit.json', {'images':audit,'validation_used_for_training':False,
             'note':'Byte-level disjointness does not prove different physical beans.'})
        dataset = TensorDataset(torch.from_numpy(np.stack(xs)), torch.from_numpy(np.stack(ys)),
                                torch.from_numpy(np.stack(vs)))
        del xs, ys, vs
        state = torch.load(ROOT/'checkpoints/A0/best.pt',map_location='cpu')
        class TorchSession:
            def __init__(self, model): self.model=model
            def run(self, _, inputs):
                with torch.inference_mode():
                    return [self.model(torch.from_numpy(inputs['image']).cuda()).float().cpu().numpy()]
        # BN statistics drift was a problem in earlier full-model fine-tuning.
        # Hold the encoder and its BN statistics fixed; alter only the decoder.
        for trial,(lr,boundary_weight) in enumerate(((1e-5,1.0),(3e-6,1.5),(1e-6,.75)),1):
            cfg = copy.deepcopy(self.best()['manifest'])
            model = Segmenter(widths=state['config']['widths']).cuda()
            model.load_state_dict(state['model'])
            for p in model.encoder.parameters(): p.requires_grad_(False)
            optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=lr)
            best_score = self.best()['selection_proxy_out_of_15']
            stale = 0
            for epoch in range(1,13):
                model.train(); model.encoder.eval()
                sampler=WeightedRandomSampler(shares,len(dataset),replacement=True,
                    generator=torch.Generator().manual_seed(41+epoch))
                losses=[]
                for x,y,v in DataLoader(dataset,batch_size=4,sampler=sampler):
                    x,y,v=x.cuda(),y.cuda(),v.cuda()
                    rot=int(torch.randint(4,()).item())
                    x,y,v=(torch.rot90(a,rot,(-2,-1)) for a in (x,y,v))
                    if torch.rand(()).item()<.5: x,y,v=(a.flip(-1) for a in (x,y,v))
                    optimizer.zero_grad(set_to_none=True)
                    logits=model(x)
                    bce=(F.binary_cross_entropy_with_logits(logits,y,reduction='none')*v).sum((0,2,3))/v.sum()
                    prob=logits.sigmoid()*v
                    dice=1-(2*(prob*y).sum((0,2,3))+1)/(prob.sum((0,2,3))+(y*v).sum((0,2,3))+1)
                    weights=torch.tensor([1.,boundary_weight],device='cuda')
                    loss=((bce+dice)*weights).sum()/weights.sum()
                    loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
                    optimizer.step();losses.append(loss.item())
                model.eval()
                row=self.evaluate(f'finetune{trial}_epoch{epoch}',cfg,TorchSession(model))
                row.update(lr=lr,boundary_weight=boundary_weight,loss=float(np.mean(losses)))
                if self.eligible(row) and row['selection_proxy_out_of_15'] > best_score+.01:
                    best_score=row['selection_proxy_out_of_15'];stale=0
                    checkpoint=self.out/f'finetune{trial}_best.pt'
                    torch.save({'model':model.state_dict(),'config':state['config'],
                                'epoch':epoch,'lr':lr,'boundary_weight':boundary_weight},checkpoint)
                    row['checkpoint']=str(checkpoint)
                    # Keep an immutable snapshot for each accepted epoch.
                    onnx_path=self.out/f'{row["name"]}.onnx'
                    torch.onnx.export(model.cpu(),(torch.zeros(1,3,512,512),),onnx_path,
                        input_names=['image'],output_names=['logits'],opset_version=17,dynamo=False)
                    model.cuda();row['onnx']=str(onnx_path)
                else: stale+=1
                save(self.out/'search_history.json',self.rows)
                save(self.out/'search_status.json',{'phase':'finetune','trial':trial,
                     'epoch':epoch,'stale_epochs':stale,'best':self.best()['name']})
                if stale>=3: break
            del model,optimizer
            torch.cuda.empty_cache()
        self.predictor.segmenter=self.initial_session

    def finalize(self):
        # Export only baseline and Pareto-relevant candidates; full CLI includes
        # loading, decoding, masks, JSON and previews. No warm-up is hidden.
        eligible=[r for r in self.rows if self.eligible(r)
                  and (not r['name'].startswith('finetune') or 'onnx' in r)]
        peak=max(r['selection_proxy_out_of_15'] for r in eligible)
        near=[r for r in eligible if r['selection_proxy_out_of_15']>=peak-.15
              and r.get('timing_backend','CPU')=='CPU']
        selections={
            'accuracy':self.best(),
            'balanced':min(near,key=lambda r:r['windows_warm_compute_max_group_median_s']),
            'speed':min((r for r in eligible if r.get('timing_backend','CPU')=='CPU'),
                        key=lambda r:r['windows_warm_compute_max_group_median_s']),
        }
        result={}
        for label,row in selections.items():
            dest=ROOT/'models'/f'A0_day3_search_{label}'
            dest.mkdir(exist_ok=True)
            import shutil
            origin=ROOT/'models/A0_classifier_day3'
            shutil.copy2(row.get('onnx',origin/'model.onnx'),dest/'model.onnx')
            shutil.copy2(origin/'classifier.onnx',dest/'classifier.onnx')
            save(dest/'manifest.json',row['manifest'])
            reports={}
            for group,root in self.roots.items():
                target=self.out/'final'/label/group
                target.mkdir(parents=True,exist_ok=True)
                with (target/'infer.log').open('w',encoding='utf-8') as log:
                    reports[group]=run(dest,target,'valid',4,log,data_root=root)
            result[label]={'candidate':row,'model_folder':str(dest),'cli_reports':reports}
        save(self.out/'final_results.json',result)
        with (self.out/'comparison.csv').open('w',newline='',encoding='utf-8') as handle:
            fields=['candidate','selection_proxy_out_of_15','windows_warm_compute_s','timing_backend',
                    'legacy_iou','legacy_counting','triangle_iou','triangle_counting','eligible']
            writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader()
            for row in self.rows:
                writer.writerow({'candidate':row['name'],
                    'selection_proxy_out_of_15':row['selection_proxy_out_of_15'],
                    'windows_warm_compute_s':row['windows_warm_compute_max_group_median_s'],
                    'timing_backend':row.get('timing_backend','CPU'),
                    'eligible':self.eligible(row),
                    **{f'{g}_{k}':row['metrics'][g][k] for g in self.scenes for k in ('iou','counting')}})
        save(self.out/'search_status.json',{'phase':'complete','trials':len(self.rows),
             'note':'Windows only; Pi <=1s and official latency rank not yet verified.'})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--train-root',type=Path,default=ROOT.parent/'coffee_ready_reviewed_v1/coco')
    p.add_argument('--adapt-root',type=Path,default=ROOT.parent/'day3_classifier_split_v1')
    p.add_argument('--skip-training',action='store_true')
    args=p.parse_args()
    search=Search(args.out.resolve())
    search.sweep()
    if not args.skip_training: search.train(args.train_root,args.adapt_root)
    search.finalize()


if __name__=='__main__': main()
