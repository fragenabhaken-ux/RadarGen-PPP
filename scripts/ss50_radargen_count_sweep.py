"""RadarGen SS50: fix optimized maps and sweep one shared recovery threshold."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import resource
import time
from types import SimpleNamespace

import h5py
import numpy as np
import torch
import yaml

from evaluation.validation import (foreground_metrics, json_values, sample_metrics,
                                   validation_boxes, write_reports)
from evaluation.metrics import MMDLoss, RBF
from radargen.core.normalization import NormalizationConfig
# Load the standalone recovery implementation without importing the package's
# eager diffusion/depth model imports. No model inference is needed here.
import importlib.util
_recovery_path = Path(__file__).resolve().parents[1] / 'radargen/inference/pcl_recovery.py'
_spec = importlib.util.spec_from_file_location('_count_sweep_recovery', _recovery_path)
_recovery = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_recovery)
solve_irl1 = _recovery.solve_irl1
from scripts.ss50_sampling_budget import read_rows, DINO, RADARGEN, ROOT

ARCHIVE = ROOT / 'validation/FULL_validation_20261108/radargen'
OUTPUT = Path('output/paper_material/count_sweep_evaluation/ss50_radargen_count_sweep')


def threshold_for_mean(maps, target):
    """Nearest attainable mean under a single nonnegative strict threshold."""
    positive = np.sort(np.concatenate([m[m > 0].ravel() for m in maps]))
    requested = int(round(target * len(maps)))
    if requested > len(positive):
        raise ValueError(f'Target {target} exceeds maximum attainable mean {len(positive)/len(maps):.2f}')
    if requested == len(positive):
        return 0.
    if requested == 0:
        return float(positive[-1])
    lower, upper = positive[-requested-1], positive[-requested]
    if lower != upper:
        return float(lower)
    # Ties cannot be split without introducing a frame-specific or random rule.
    candidates = [float(lower), float(np.nextafter(lower, np.array(-np.inf, dtype=positive.dtype)))]
    return min(candidates, key=lambda t: abs(np.count_nonzero(positive > t)-requested))


def points_from_map(amplitude, rcs, doppler, threshold, radius):
    rows, cols = np.nonzero(amplitude > threshold)
    height, width = amplitude.shape
    # Match recover_pcl_attributes: float32 pixel-center coordinates, rounded RCS.
    xy = np.column_stack((cols, height-rows-1)).astype(np.float32) + .5
    xy = xy / np.array([width, height], dtype=np.float32) * (2*radius) - radius
    return np.column_stack((xy, np.rint(rcs[rows, cols]), doppler[rows, cols]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--workers', type=int, default=24)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('workers must be positive')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    started = time.perf_counter()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    cache = output/'optimized_maps'
    cache.mkdir(exist_ok=True)
    rows = read_rows(RADARGEN)
    dino = read_rows(DINO)
    identity = lambda rr: [(r['scene'], r['frame']) for r in rr]
    assert len(rows) == 50 and identity(rows) == identity(dino)
    provenance = json.loads((ARCHIVE/'validation_config.json').read_text())
    original_provenance = json.loads((RADARGEN/'validation_config.json').read_text())
    frames = []
    with h5py.File(ARCHIVE/'predictions.h5','r') as h:
        assert h.attrs['complete']
        stored_provenance = json.loads(h['provenance_json'].asstr()[()])
        for key in ('checkpoint_path','seed','num_inference_steps'):
            assert stored_provenance[key] == provenance[key] == original_provenance[key]
        norm = NormalizationConfig(**json.loads(h['normalization_json'].asstr()[()]))
        radius = float(h.attrs['coordinate_range'])
        for index,row in enumerate(rows):
            g=h[f"scenes/{row['scene']}/frames/{row['frame']:06d}"]
            assert g.attrs['complete']
            metadata=json.loads(g.attrs['metadata_json'])
            assert (metadata['scene_token'],int(metadata['frame_index'])) == (row['scene'],row['frame'])
            fields=[g[f'fields/{k}'][()] for k in ('point_density','rcs','doppler')]
            with h5py.File(str(g.attrs['source']),'r') as target:
                gt=np.column_stack([target[f'ppp_targets/{k}'][()] for k in ('xy','rcs','doppler')])
            # Preprocessing was relocated; compare actual continuous targets,
            # rather than requiring historical and current path strings to match.
            for reference in (row,dino[index]):
                with h5py.File(reference['source'],'r') as target:
                    reference_gt=np.column_stack([target[f'ppp_targets/{k}'][()] for k in ('xy','rcs','doppler')])
                if not np.array_equal(gt,reference_gt):
                    raise ValueError(f'Continuous GT differs for SS50 frame {index}')
            frames.append(dict(index=index, row=row, pd=fields[0], rcs=fields[1], doppler=fields[2],
                gt=gt, metadata=json.loads(g.attrs['metadata_json']), archived_points=g['points'][()]))
    def optimize(frame):
        path=cache/f"{frame['index']:02d}.npz"
        digest=hashlib.sha256(np.ascontiguousarray(frame['pd']).tobytes()).hexdigest()
        if path.exists():
            with np.load(path) as saved:
                assert str(saved['map_sha256']) == digest
                amplitude=saved['amplitude']
        else:
            pd=frame['pd']
            assert pd.ndim == 2 and np.isfinite(pd).all() and pd.max()>0
            with torch.inference_mode():
                amplitude=solve_irl1(torch.from_numpy(pd/pd.max()).float(), 2.,2.,
                    alpha=.0018,n_irl1_iter=5,n_fista_iter=300).numpy()
            assert np.isfinite(amplitude).all()
            np.savez_compressed(path, amplitude=amplitude,map_sha256=digest)
        frame['amplitude']=amplitude
        native=points_from_map(amplitude,frame['rcs'],frame['doppler'],.1,radius)
        frame['native_count']=len(native)
        frame['archived_count']=len(frame['archived_points'])
        print(f"Optimized {frame['index']+1}/50; count@0.1={len(native)}; positive cells={(amplitude>0).sum()}",flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        list(executor.map(optimize,frames))
    optimization_seconds=time.perf_counter()-started
    amplitudes=[f['amplitude'] for f in frames]
    max_mean=float(np.mean([(a>0).sum() for a in amplitudes]))
    targets=list(range(100,2001,100))
    thresholds={b:threshold_for_mean(amplitudes,b) for b in targets if b<=max_mean}
    unavailable=[b for b in targets if b>max_mean]
    manifest=dict(model='radargen',archive=str(ARCHIVE),checkpoint=provenance['checkpoint_path'],
        frames=50,workers=args.workers,device='cpu',sigma_pix=2.,lam=.0018,
        n_irl1_iter=5,n_fista_iter=300,original_threshold=.1,
        maximum_positive_cell_mean=max_mean,unavailable_targets=unavailable,
        target_thresholds=thresholds,optimization_seconds=optimization_seconds,
        identities=identity(rows),
        provenance_note='Archived full-validation maps on exact SS50 identities. Per-sample RNG depends on evaluation index, so these maps can differ from historical SS50 predictions. Native threshold 0.1 baseline is recomputed from these same fixed maps; historical SS50 references are separately labeled.',
        native_cpu_counts=[f['native_count'] for f in frames],archived_counts=[f['archived_count'] for f in frames])
    (output/'experiment.json').write_text(json.dumps(manifest,indent=2))
    print(f'Maximum mean count at threshold 0: {max_mean:.2f}; unattainable targets: {unavailable}',flush=True)
    from truckscenes import TruckScenes
    from radargen.datasets.truckscenes import TruckScenesAdapter
    from radargen.datasets.truckscenes.config import TRUCKSCENES_REFERENCE_SENSOR
    from radargen.core.protocols import DatasetConfig
    config=yaml.safe_load(Path(provenance['dataset_config']).read_text())['data']
    db=TruckScenes(config['extra']['dataset_version'],config['dataset_dir'],verbose=False)
    adapter=TruckScenesAdapter(db,config=DatasetConfig(name='truckscenes',camera_views=config['extra']['camera_views'],
        reference_sensor=TRUCKSCENES_REFERENCE_SENSOR,normalization=norm,data_root=db.dataroot,version=db.version))
    for f in frames:
        f['boxes']=validation_boxes(SimpleNamespace(adapter=adapter),f['metadata'])
    device=torch.device('cpu')
    mmd=MMDLoss(kernel=RBF(device=device),device=device)
    tasks=[('native_threshold_0p1',.1)]+[(f'count_{b:04d}',t) for b,t in thresholds.items()]
    def evaluate(task):
        name,threshold=task
        directory=output/name
        directory.mkdir(exist_ok=True)
        begin=time.perf_counter()
        evaluated=[]
        with (directory/'metrics.jsonl').open('w') as stream, torch.inference_mode():
            for f in frames:
                tick=time.perf_counter()
                points=points_from_map(f['amplitude'],f['rcs'],f['doppler'],threshold,radius)
                seconds=time.perf_counter()-tick
                metrics=sample_metrics(points,f['gt'],norm,mmd,device)
                metrics['thresholding_time_seconds']=seconds
                from radargen.ppp_targets.continuous import locate_detections
                valid,rr,cc=locate_detections(f['gt'],f['rcs'].shape,radius)
                metrics['target_collision_count']=int(len(valid)-len(set(zip(rr.tolist(),cc.tolist()))))
                for column,key in [(2,'rcs'),(3,'doppler')]:
                    error=float(np.abs(f[key][rr,cc]-valid[:,column]).mean()) if len(valid) else None
                    metrics[key+'_mae_at_detections']=error
                    metrics[key+'_mae_at_target_cells']=error
                foreground,clouds=foreground_metrics(points,f['gt'],f['boxes'],norm,device,include_points=True)
                row=json_values(dict(index=f['index'],scene=f['row']['scene'],frame=f['row']['frame'],
                    source=f['row']['source'],status='ok' if len(points) else 'empty_cloud',plot='',metrics=metrics,
                    foreground=foreground,foreground_points=clouds,foreground_box_count=len(f['boxes'])))
                stream.write(json.dumps(row,allow_nan=False)+'\n'); stream.flush(); evaluated.append(row)
        write_reports(directory,evaluated,dict(provenance,sample_count=50),device)
        timing=dict(name=name,threshold=threshold,seconds=time.perf_counter()-begin)
        print('Completed '+json.dumps(timing),flush=True)
        return timing
    with ThreadPoolExecutor(max_workers=min(args.workers,len(tasks))) as executor:
        timings=list(executor.map(evaluate,tasks))
    (output/'timing.json').write_text(json.dumps(dict(optimization_seconds=optimization_seconds,
        total_seconds=time.perf_counter()-started,evaluations=timings),indent=2))
    from scripts.report_radargen_count_sweep import report
    report(output)


if __name__=='__main__':
    import sys
    start=time.perf_counter(); success=False
    try:
        main(); success=True
    finally:
        usage=resource.getrusage(resource.RUSAGE_SELF)
        stats=dict(success=success,wall_seconds=time.perf_counter()-start,user_cpu_seconds=usage.ru_utime,
                   system_cpu_seconds=usage.ru_stime,max_rss_kib=usage.ru_maxrss)
        directory=Path(sys.argv[sys.argv.index('--output')+1]) if '--output' in sys.argv else OUTPUT
        if directory.is_dir(): (directory/'resource_usage.json').write_text(json.dumps(stats,indent=2))
        print('Resource usage: '+json.dumps(stats),flush=True)
