"""Position-only one-to-one P/R/F1 for existing DV3 and RadarGen SS50 sweeps."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import argparse
import shutil
from pathlib import Path
import time
import h5py
import numpy as np
import torch
from evaluation.metrics import distance_attr_recall_precision_f1_hungarian
from scripts.ss50_sampling_budget import sample_scaled

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'output/paper_material/count_sweep_evaluation'
DINO=BASE/'ss50_sampling_budget';RADAR=BASE/'ss50_radargen_count_sweep'
OUT=ROOT/'output/paper_material/count_sweep_comparisons/single_column/dino_final/position_only'


def radar_points(amplitude,threshold,radius):
    rows,cols=np.nonzero(amplitude>threshold);h,w=amplitude.shape
    xy=np.column_stack((cols,h-rows-1)).astype(np.float32)+.5
    xy=xy/np.array([w,h],dtype=np.float32)*(2*radius)-radius
    return np.column_stack((xy,np.zeros((len(xy),2))))


def native_points(mass,radius,index):
    generator=torch.Generator(device='cpu').manual_seed(int(np.random.SeedSequence([42,index]).generate_state(1)[0]))
    counts=torch.poisson(torch.from_numpy(mass),generator=generator)
    indices=torch.repeat_interleave(torch.arange(mass.size),counts.flatten().long()).numpy()
    h,w=mass.shape;rows,cols=np.unravel_index(indices,mass.shape)
    return np.column_stack(((cols+.5)/w*2*radius-radius,(h-rows-.5)/h*2*radius-radius,np.zeros((len(indices),2))))


def score(points,gt):
    # Existing distance-and-attribute matcher, with both mark gates disabled.
    return distance_attr_recall_precision_f1_hungarian(torch.from_numpy(gt).float(),torch.from_numpy(points).float(),
        distance_threshold=1.,rcs_threshold=float('inf'),doppler_threshold=float('inf'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir',type=Path,default=DINO)
    parser.add_argument('--output',type=Path,default=OUT)
    parser.add_argument('--model-key',default='dv3')
    args=parser.parse_args()
    model_dir=args.model_dir;output=args.output;model_key=args.model_key
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    output.mkdir(exist_ok=True,parents=True);started=time.perf_counter()
    manifest=json.loads((model_dir/'experiment.json').read_text());rm=json.loads((RADAR/'experiment.json').read_text())
    assert manifest['ss50_identities']==rm['identities']
    original_dv3=model_dir.resolve()==DINO.resolve()
    reference_dir=Path(manifest['baseline_dino']) if original_dv3 else model_dir/'native'
    reference=[json.loads(line) for line in (reference_dir/'metrics.jsonl').read_text().splitlines()]
    # Reuse the already evaluated RadarGen sweep on the identical SS50 GT.
    for file in OUT.glob('radargen*.json'):
        target=output/file.name
        if target.resolve()!=file.resolve():shutil.copy2(file,target)
    expected_counts={}
    for budget in range(100,2001,100):
        for seed in [42,43,44]:
            rows=[json.loads(line) for line in (model_dir/f'budget_{budget:04d}_seed_{seed}/metrics.jsonl').read_text().splitlines()]
            assert [(r['scene'],r['frame']) for r in rows]==[(r['scene'],r['frame']) for r in reference]
            expected_counts[budget,seed]=[r['metrics']['generated_count'] for r in rows]
    frames=[]
    archive=Path(manifest['archive'])
    if archive.is_dir():archive=archive/'predictions.h5'
    with h5py.File(archive,'r') as h:
        radius=float(h.attrs['coordinate_range']);assert h.attrs['complete']
        for i,(scene,frame) in enumerate(manifest['ss50_identities']):
            g=h[f'scenes/{scene}/frames/{frame:06d}'];assert g.attrs['complete']
            with h5py.File(str(g.attrs['source']),'r') as f:
                gt=np.column_stack([f[f'ppp_targets/{k}'][()] for k in ('xy','rcs','doppler')])
            with np.load(RADAR/f'optimized_maps/{i:02d}.npz') as cache:amplitude=cache['amplitude']
            frames.append(dict(gt=gt,mass=g['fields/cell_mass_grid'][()],amplitude=amplitude,archived_points=g['points'][()]))
    mean_mass=float(np.mean([f['mass'].sum() for f in frames]));assert abs(mean_mass-manifest['mean_original_expected_count'])<1e-8
    tasks=[(model_key,b,s) for b in range(100,2001,100) for s in [42,43,44]]
    tasks += [('radargen',int(b),None) for b in rm['target_thresholds']]
    tasks += [(model_key+'_native',None,None),('radargen_native',None,None)]
    def evaluate(task):
        model,budget,seed=task
        name=model if budget is None else f'{model}_{budget:04d}'+(f'_seed_{seed}' if seed is not None else '')
        path=output/f'{name}.json'
        if path.exists():return json.loads(path.read_text())
        rows=[]
        with torch.inference_mode():
            for i,f in enumerate(frames):
                if model==model_key:
                    points=sample_scaled(f['mass'],f['mass'],f['mass'],radius,budget/mean_mass,
                        np.random.default_rng(np.random.SeedSequence([seed,budget,i])))
                    points[:,2:]=0
                    assert len(points)==expected_counts[budget,seed][i]
                elif model==model_key+'_native':
                    points=native_points(f['mass'],radius,i) if original_dv3 else f['archived_points']
                    assert len(points)==reference[i]['metrics']['generated_count'],(i,len(points),reference[i]['metrics']['generated_count'])
                else:
                    threshold=.1 if model=='radargen_native' else rm['target_thresholds'][str(budget)]
                    points=radar_points(f['amplitude'],threshold,radius)
                result=score(points,f['gt']);result.update(count=len(points),gt_count=len(f['gt']),index=i)
                rows.append(result)
        summary=dict(model=model,target=budget,seed=seed,frames=50,
            mean={k:float(np.mean([r[k] for r in rows])) for k in ['precision','recall','f1','count','gt_count']},rows=rows)
        path.write_text(json.dumps(summary,indent=2));print('Completed '+name,flush=True)
        return summary
    with ThreadPoolExecutor(max_workers=24) as executor:
        results=[future.result() for future in as_completed([executor.submit(evaluate,t) for t in tasks])]
    (output/'provenance.json').write_text(json.dumps(dict(checkpoint=manifest['checkpoint'],model_key=model_key,model_source=str(model_dir),radargen_reused_from=str(OUT),matching='Existing Hungarian one-to-one matching; distance <= 1 m; RCS/Doppler gates disabled.',
        frames=50,sweep_samples_verified_against_original_counts=True,
        f1_aggregation='Per-frame F1 averaged across frames and then seeds; iso-F1 contours instead use harmonic mean of plotted aggregate P/R.',
        wall_seconds=time.perf_counter()-started,runs=len(results)),indent=2))
    print('Position-only sweep finished',time.perf_counter()-started,flush=True)


if __name__=='__main__':main()
