"""Shared-intensity SS50 count sweeps for three ablations, CPU only."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import resource
import time
from types import SimpleNamespace

import h5py
import numpy as np
import torch
import yaml

from evaluation.validation import (sample_metrics, foreground_metrics, validation_boxes,
                                   write_reports, json_values)
from evaluation.metrics import MMDLoss, RBF
from evaluation.ppp_metrics import compute_ppp_count_metrics, make_ppp_grid_xy
from radargen.core.normalization import NormalizationConfig
from radargen.ppp_targets.continuous import locate_detections
from scripts.ss50_sampling_budget import sample_scaled, read_rows, RADARGEN, ROOT

OUT=Path('output/paper_material/count_sweep_evaluation')
BUNDLES=ROOT/'validation/FULL_ablation_bundles'
SPECS=[('dit_no_bottleneck','DiT no bottleneck — step 58,057','dit_dcae_no_bottleneck_2867524_step58057',58057),
       ('dino_compact','DINO Compact — step 20,595','dino_neck_compact_2867528_step20595',20595),
       ('dino_random','Random-initialized DINO — step 29,915','dino_random_init_2863217_step29915',29915)]
RADAR_OUTPUT=OUT/'ss50_radargen_count_sweep'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers',type=int,default=24)
    parser.add_argument('--specs-json',type=Path,
                        help='Optional list of [slug,label,bundle,step] checkpoint specifications')
    parser.add_argument('--usage-output',type=Path,
                        help='Separate resource-usage record for a custom checkpoint batch')
    args=parser.parse_args()
    if args.workers<1: parser.error('workers must be positive')
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    started=time.perf_counter()
    reference=read_rows(RADARGEN)
    assert len(reference)==50
    radar_rows=read_rows(RADAR_OUTPUT/'native_threshold_0p1')
    assert [(r['scene'],r['frame']) for r in radar_rows]==[(r['scene'],r['frame']) for r in reference]
    models=[]
    specs=json.loads(args.specs_json.read_text()) if args.specs_json else SPECS
    if not specs: parser.error('At least one checkpoint specification is required')
    for slug,label,bundle,step in specs:
        source=BUNDLES/bundle/'metrics'
        output=OUT/f'ss50_{slug}_count_sweep';output.mkdir(exist_ok=True)
        with h5py.File(source/'predictions.h5','r') as h:
            if not h.attrs['complete']:raise ValueError(f'Incomplete archive: {source}')
            provenance=json.loads(h['provenance_json'].asstr()[()])
            if f'_step_{step}.pth' not in provenance['checkpoint_path']:
                raise ValueError(f'Unexpected checkpoint: {provenance["checkpoint_path"]}')
            norm=NormalizationConfig(**json.loads(h['normalization_json'].asstr()[()]))
            radius=float(h.attrs['coordinate_range'])
            frames=[]
            for index,row in enumerate(reference):
                g=h[f"scenes/{row['scene']}/frames/{row['frame']:06d}"]
                if not g.attrs['complete']:raise ValueError('Incomplete frame')
                metadata=json.loads(g.attrs['metadata_json'])
                assert (metadata['scene_token'],int(metadata['frame_index']))==(row['scene'],row['frame'])
                mass,rcs,doppler=[g[f'fields/{name}'][()] for name in ('cell_mass_grid','rcs','doppler')]
                assert mass.ndim==2 and np.isfinite(mass).all() and (mass>=0).all() and mass.sum()>0
                source_path=str(g.attrs['source'])
                with h5py.File(source_path,'r') as target:
                    gt=np.column_stack([target[f'ppp_targets/{k}'][()] for k in ('xy','rcs','doppler')])
                with h5py.File(row['source'],'r') as target:
                    reference_gt=np.column_stack([target[f'ppp_targets/{k}'][()] for k in ('xy','rcs','doppler')])
                if not np.array_equal(gt,reference_gt):raise ValueError('GT differs from RadarGen reference')
                valid,rr,cc=locate_detections(gt,rcs.shape,radius)
                fixed={'target_collision_count':int(len(valid)-len(set(zip(rr.tolist(),cc.tolist()))))}
                for column,name,field in [(2,'rcs',rcs),(3,'doppler',doppler)]:
                    mae=float(np.abs(field[rr,cc]-valid[:,column]).mean()) if len(valid) else None
                    fixed[name+'_mae_at_detections']=mae;fixed[name+'_mae_at_target_cells']=mae
                frames.append(dict(index=index,source=source_path,metadata=metadata,mass=mass,rcs=rcs,
                    doppler=doppler,gt=gt,archived_points=g['points'][()],fixed=fixed))
        mean_mass=float(np.mean([f['mass'].sum() for f in frames]))
        manifest=dict(slug=slug,label=label,checkpoint=provenance['checkpoint_path'],archive=str(source/'predictions.h5'),
            frames=50,budgets=list(range(100,2001,100)),seeds=[42,43,44],workers=args.workers,device='cpu',
            mean_original_expected_count=mean_mass,ss50_identities=[(r['scene'],r['frame']) for r in reference],
            radargen_reference=str(RADAR_OUTPUT),native_point_provenance='Original archived points on the same fixed SS50 fields; full-validation per-frame RNG indices.',
            sampling='One shared multiplier B/mean_SS50(total intensity); Poisson total and categorical cells; retain scene density variation.')
        manifest_path=output/'experiment.json'
        if manifest_path.exists():
            existing=json.loads(manifest_path.read_text())
            for key in ['checkpoint','archive','ss50_identities','budgets','seeds']:
                if existing[key]!=json.loads(json.dumps(manifest[key])):raise ValueError(f'Existing run mismatch: {key}')
        manifest_path.write_text(json.dumps(manifest,indent=2))
        models.append(dict(slug=slug,label=label,source=source,output=output,provenance=provenance,
                           norm=norm,radius=radius,frames=frames,mean_mass=mean_mass,timings=[]))
        print(f'Loaded {label}: expected native mean={mean_mass:.2f}',flush=True)
    # A single annotation database and shared CPU worker pool for all three models.
    from truckscenes import TruckScenes
    from radargen.datasets.truckscenes import TruckScenesAdapter
    from radargen.datasets.truckscenes.config import TRUCKSCENES_REFERENCE_SENSOR
    from radargen.core.protocols import DatasetConfig
    config=yaml.safe_load(Path(models[0]['provenance']['dataset_config']).read_text())['data']
    database=TruckScenes(config['extra']['dataset_version'],config['dataset_dir'],verbose=False)
    norm=models[0]['norm']
    assert all(vars(m['norm'])==vars(norm) for m in models)
    adapter=TruckScenesAdapter(database,config=DatasetConfig(name='truckscenes',camera_views=config['extra']['camera_views'],
        reference_sensor=TRUCKSCENES_REFERENCE_SENSOR,normalization=norm,data_root=database.dataroot,version=database.version))
    boxes=[validation_boxes(SimpleNamespace(adapter=adapter),f['metadata']) for f in models[0]['frames']]
    for model in models:
        grid=make_ppp_grid_xy(*model['frames'][0]['mass'].shape,model['radius'])
        for frame,frame_boxes in zip(model['frames'],boxes):
            frame['boxes']=frame_boxes
            frame['analytic']=compute_ppp_count_metrics(frame['mass'],frame['gt'],frame_boxes,grid)
    setup_seconds=time.perf_counter()-started
    device=torch.device('cpu')
    def evaluate(task):
        model,budget,seed=task
        name='native' if budget is None else f'budget_{budget:04d}_seed_{seed}'
        directory=model['output']/name;directory.mkdir(exist_ok=True)
        complete=directory/'summary.json'
        if complete.exists():
            saved=json.loads(complete.read_text())
            if saved.get('complete') and saved['evaluated_samples']==50:return None
        tick=time.perf_counter();evaluated=[]
        scale=1. if budget is None else budget/model['mean_mass']
        mmd=MMDLoss(kernel=RBF(device=device),device=device)
        with (directory/'metrics.jsonl').open('w') as stream,torch.inference_mode():
            for frame in model['frames']:
                begin=time.perf_counter()
                points=(frame['archived_points'] if budget is None else sample_scaled(frame['mass'],frame['rcs'],
                    frame['doppler'],model['radius'],scale,np.random.default_rng(np.random.SeedSequence([seed,budget,frame['index']]))))
                sampling_seconds=time.perf_counter()-begin
                metrics=sample_metrics(points,frame['gt'],model['norm'],mmd,device)
                metrics.update(frame['fixed'])
                if budget is not None:metrics['sampling_time_seconds']=sampling_seconds
                foreground,clouds=foreground_metrics(points,frame['gt'],frame['boxes'],model['norm'],device,include_points=True)
                # PPP expectations scale linearly; reuse exact original box integrals.
                analytic=frame['analytic']
                total=float(frame['mass'].sum())*scale
                metrics.update(ppp_expected_total_count=total,ppp_total_count_abs_error=abs(total-len(frame['gt'])))
                original_counts=analytic['ppp_expected_count']
                foreground['ppp_expected_count']=[v*scale for v in original_counts]
                # Recover each box's GT count from the original absolute error is ambiguous,
                # so calculate GT occupancy once explicitly below, not from that error.
                foreground['ppp_expected_count_abs_error']=[abs(v*scale-g) for v,g in zip(original_counts,frame['box_gt_counts'])]
                foreground['ppp_hit_probability_gt_positive']=[float(-np.expm1(-v*scale)) for v,g in zip(original_counts,frame['box_gt_counts']) if g>0]
                row=json_values(dict(index=frame['index'],scene=frame['metadata']['scene_token'],frame=int(frame['metadata']['frame_index']),
                    source=frame['source'],status='ok' if len(points) else 'empty_cloud',plot='',metrics=metrics,
                    foreground=foreground,foreground_points=clouds,foreground_box_count=len(frame['boxes'])))
                stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush();evaluated.append(row)
        write_reports(directory,evaluated,dict(model['provenance'],sample_count=50),device)
        timing=dict(name=name,seconds=time.perf_counter()-tick,budget=budget,seed=seed)
        print(f'{model["slug"]}: {name} completed in {timing["seconds"]:.1f}s',flush=True)
        return model,timing
    from evaluation.box_utils import get_points_in_box
    for model in models:
        for frame in model['frames']:
            frame['box_gt_counts']=[len(get_points_in_box(frame['gt'],box.bottom_corners())[0]) for box in frame['boxes']]
    tasks=[(m,None,None) for m in models]+[(m,b,s) for b in range(100,2001,100) for s in [42,43,44] for m in models]
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for future in as_completed([executor.submit(evaluate,task) for task in tasks]):
            result=future.result()
            if result is not None:
                model,timing=result;model['timings'].append(timing)
                (model['output']/'timing.json').write_text(json.dumps(dict(setup_seconds_shared=setup_seconds,evaluations=model['timings']),indent=2))
    from scripts.report_ablation_count_sweep import report
    for model in models:report(model['output'])
    usage=resource.getrusage(resource.RUSAGE_SELF)
    stats=dict(success=True,wall_seconds=time.perf_counter()-started,user_cpu_seconds=usage.ru_utime,
               system_cpu_seconds=usage.ru_stime,max_rss_kib=usage.ru_maxrss,workers=args.workers,
               models=[m['slug'] for m in models])
    usage_output=args.usage_output or OUT/'ablation_count_sweeps_resource_usage.json'
    usage_output.parent.mkdir(parents=True,exist_ok=True)
    usage_output.write_text(json.dumps(stats,indent=2))
    print('All requested sweeps complete: '+json.dumps(stats),flush=True)


if __name__=='__main__':main()
