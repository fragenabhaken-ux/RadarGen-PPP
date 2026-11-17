"""Tables and individual figures for the fixed-map RadarGen threshold sweep."""
import csv
import json
from pathlib import Path
import numpy as np

DINO_OUTPUT=Path('output/paper_material/count_sweep_evaluation/ss50_sampling_budget')


def value(s,scope,key):
    v=s[scope].get(key)
    return v.get('mean') if isinstance(v,dict) else v


def report(output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output=Path(output)
    manifest=json.loads((output/'experiment.json').read_text())
    sweep=[]
    for f in sorted(output.glob('count_*/summary.json')):
        budget=int(f.parent.name.split('_')[1])
        sweep.append((budget,json.loads(f.read_text())))
    native=json.loads((output/'native_threshold_0p1/summary.json').read_text())
    references={name:json.loads((DINO_OUTPUT/f'baseline_{key}.json').read_text()) for name,key in
        [('DINO final','dino'),('DINO 15K','dino_15k'),('RadarGen historical','radargen')]}
    for name,key in [('DINO final','dino'),('DINO 15K','dino_15k'),('RadarGen historical','radargen')]:
        (output/f'baseline_{key}.json').write_text(json.dumps(references[name],indent=2))
    gt=value(native,'metrics','gt_count')
    records=[(f'Target {budget}',s,False) for budget,s in sweep]
    records += [('RadarGen native threshold 0.1',native,True)]+[(n,s,True) for n,s in references.items()]
    records.sort(key=lambda item:value(item[1],'metrics','generated_count'))
    whole=['generated_count','count_abs_error','chamfer_distance_xy','chamfer_distance_all_norm','iou_th1',
        'distance_attr_recall','distance_attr_precision','distance_attr_f1','pcl_mmd_xy','pcl_mmd_rcs','pcl_mmd_doppler']
    headings=['Mean points','Count error ↓','CD location ↓','CD full ↓','IoU ↑','Recall ↑','Precision ↑','F1 ↑','MMD location ↓','MMD RCS ↓','MMD Doppler ↓']
    fg=['per_box_chamfer_distance_xy','per_box_chamfer_distance_all_norm','per_box_num_points_similarity','hit_rate']
    lines=[f'# RadarGen SS50 count sweep — GT mean: {gt:,.2f} points/frame','',
        'The same 50 frame identities as the DINO sweep. Each target uses one shared threshold across all optimized maps, with fixed IRL1 settings. Counts vary by frame. Deterministic recovery: one evaluation per threshold, no sampling seeds.','',
        '**Map provenance:** Saved full-validation predictions are reused. Their RNG indices differ from the historical SS50 run, so the historical RadarGen marker is a separate reference. The native threshold 0.1 baseline is recomputed from the exact maps used for this sweep.','',
        f'Maximum attainable mean at threshold zero: **{manifest["maximum_positive_cell_mean"]:.2f}**. Unattainable requested targets: {manifest["unavailable_targets"]}. Optimization settings: Gaussian sigma 2 pixels, lambda 0.0018, 5 IRL1 iterations, up to 300 FISTA iterations per outer iteration.','']
    for scope,keys,labels,title in [('metrics',whole,headings,'Whole area'),('foreground',fg,['FG CD location ↓','FG CD full ↓','Density similarity ↑','Hit rate ↑'],'Foreground')]:
        lines += [f'## {title}','','| Operating point | '+' | '.join(labels)+' |','| --- | '+' | '.join(['---:']*len(keys))+' |']
        table=[(value(s,'metrics','generated_count'),n,s,b) for n,s,b in records]
        table.append((gt,'Ground truth',None,True));table.sort(key=lambda x:x[0])
        for count,name,s,bold in table:
            label=f'{name} ({count:,.2f} points/frame)' if bold else name
            cells=[label]+[('' if s is None else 'NA' if value(s,scope,k) is None else f'{value(s,scope,k):.3f}') for k in keys]
            if s is None and scope=='metrics':cells[1]=f'{gt:.3f}'
            if bold: cells=[f'**{c}**' if c else '' for c in cells]
            lines.append('| '+' | '.join(cells)+' |')
        lines.append('')
    lines+=['[Complete CSV](results.csv) · [Individual plots](plots/README.md)','']
    (output/'results.md').write_text('\n'.join(lines))
    csvrows=[]
    for name,s,_ in records:
        row={'operating_point':name}
        for scope in ['metrics','foreground']:
            for key in s[scope]:
                v=value(s,scope,key)
                if isinstance(v,(int,float)):row[scope+'.'+key]=v
        csvrows.append(row)
    columns=['operating_point']+sorted({k for r in csvrows for k in r if k!='operating_point'})
    with (output/'results.csv').open('w') as stream:
        w=csv.DictWriter(stream,fieldnames=columns);w.writeheader();w.writerows(csvrows)
    plots=output/'plots'; plots.mkdir(exist_ok=True)
    dino={}
    for f in DINO_OUTPUT.glob('budget_*/summary.json'):
        b=int(f.parent.name.split('_')[1]);dino.setdefault(b,[]).append(json.loads(f.read_text()))
    styles=[('RadarGen native threshold 0.1',native,'*','#b2182b'),
            ('DINO final',references['DINO final'],'o','#636363'),
            ('DINO 15K',references['DINO 15K'],'D','#d95f02'),
            ('RadarGen historical',references['RadarGen historical'],'s','#7570b3')]
    friendly = {
        'chamfer_distance_xy': 'CD location ↓', 'chamfer_distance_all_norm': 'CD full ↓',
        'count_abs_error': 'Point-count absolute error ↓', 'distance_attr_f1': 'Detection F1 ↑',
        'distance_attr_precision': 'Detection precision ↑', 'distance_attr_recall': 'Detection recall ↑',
        'iou_th1': 'IoU @ 1 m ↑', 'pcl_mmd_xy': 'MMD location ↓',
        'pcl_mmd_rcs': 'MMD RCS ↓', 'pcl_mmd_doppler': 'MMD Doppler ↓',
        'rcs_mae_at_detections': 'RCS MAE at GT detections ↓',
        'doppler_mae_at_detections': 'Doppler MAE at GT detections ↓',
        'per_box_chamfer_distance_xy': 'Foreground CD location ↓',
        'per_box_chamfer_distance_all_norm': 'Foreground CD full ↓',
        'per_box_num_points_similarity': 'Density similarity ↑', 'hit_rate': 'Hit rate ↑',
        'ppp_total_count_abs_error': 'PPP expected total-count error ↓',
        'ppp_expected_count_abs_error': 'PPP expected-count error per box ↓',
        'ppp_hit_probability_gt_positive': 'PPP hit probability in GT-positive boxes ↑',
        'ppp_expected_total_count': 'PPP expected total count',
        'ppp_expected_count': 'PPP expected count per box',
        'generated_count': 'Mean generated points', 'gt_count': 'Mean GT points',
        'target_collision_count': 'GT raster collisions (diagnostic)',
    }
    index=['# All individual count-sweep plots', '',
        'RadarGen: fixed optimized maps, shared recovery threshold. DINO 19,914: PPP intensity scaling, three-seed mean ± SD. Curves use realized mean generated counts. Historical RadarGen uses different predicted maps from the native threshold reference.', '',
        'PPP analytical metrics have no RadarGen equivalent. GT counts, raster collisions, and evaluated-box counts are diagnostics, not prediction-quality scores. Mark MAE is evaluated at fixed GT locations and therefore does not change with count. The target-cell MAE keys are compatibility aliases of the detection MAE keys. Timings measure different postprocessing stages and are not a model-speed comparison.', '']
    catalog=[]
    all_summaries=[native, *references.values(), *[s for _,s in sweep],
                   *[s for group in dino.values() for s in group]]
    for scope in ['metrics', 'foreground']:
        index += [f'## {"Whole area" if scope == "metrics" else "Foreground"}', '']
        keys=sorted({k for summary in all_summaries for k in summary[scope]
                     if isinstance(value(summary,scope,k),(int,float))})
        for key in keys:
            label=friendly.get(key, key.replace('_',' ').capitalize())
            if key.startswith('invariant_mmd_'):
                parts=key.split('_')
                label=f'Foreground {parts[-1]} MMD {"location" if parts[-2]=="xy" else parts[-2].upper()} ↓'
            if key.endswith('_mae_at_target_cells'):
                label=key.split('_')[0].capitalize()+' MAE (compatibility alias) ↓'
            fig,ax=plt.subplots(figsize=(7.5,6))
            radar_points=[(value(summary,'metrics','generated_count'),value(summary,scope,key))
                          for _,summary in sweep if value(summary,scope,key) is not None]
            if radar_points:
                ax.plot(*zip(*radar_points),color='#b2182b',lw=2,label='RadarGen threshold sweep')
            db=sorted(dino)
            dx=[];means=[];stds=[]
            for budget in db:
                values=[value(summary,scope,key) for summary in dino[budget]
                        if value(summary,scope,key) is not None]
                if values:
                    dx.append(np.mean([value(summary,'metrics','generated_count') for summary in dino[budget]]))
                    means.append(np.mean(values));stds.append(np.std(values))
            if dx:
                means=np.asarray(means);stds=np.asarray(stds)
                ax.plot(dx,means,color='#1b6ca8',lw=2,label='DINO 19,914 PPP sweep')
                ax.fill_between(dx,means-stds,means+stds,color='#1b6ca8',alpha=.15)
            ax.axvline(gt,color='gray',ls=':',label=f'GT mean: {gt:,.2f}')
            for name,summary,marker,color in styles:
                y=value(summary,scope,key)
                if y is None: continue
                count=value(summary,'metrics','generated_count')
                ax.scatter(count,y,marker=marker,s=100 if marker=='*' else 65,
                    color=color,edgecolor='white',linewidth=.6,zorder=5,
                    label=f'{name.replace("DINO final", "DINO 19,914")}: {count:,.2f}')
            if key.startswith('ppp_'):
                note='Analytical PPP metric — no RadarGen equivalent'
            elif key in ('target_collision_count','gt_count','gt_positive_boxes','evaluated_boxes'):
                note='Diagnostic; independent of sampling threshold/intensity'
            elif 'mae_at_' in key:
                note='Fixed fields evaluated at GT detections; independent of sampled count'
            elif 'time_seconds' in key:
                note='Postprocessing timing; stages differ between models'
            else:
                note=''
            if note:
                ax.text(.02,.97,note,transform=ax.transAxes,va='top',fontsize=8,
                        bbox=dict(facecolor='white',alpha=.85,edgecolor='none'))
            ax.set(xlabel='Realized mean points per frame',ylabel=label,title=label)
            ax.grid(alpha=.2)
            ax.spines[['top','right']].set_visible(False)
            handles,labels=ax.get_legend_handles_labels()
            fig.legend(handles,labels,loc='lower center',ncol=2,fontsize=8,frameon=False,bbox_to_anchor=(.5,.01))
            fig.suptitle('SS50 · DINO and RadarGen count sweeps',fontsize=12)
            fig.tight_layout(rect=(0,.23,1,.95))
            filename=key+'.png'
            fig.savefig(plots/filename,dpi=200,bbox_inches='tight');plt.close(fig)
            catalog.append(dict(scope=scope,key=key,filename=filename,radargen_curve=bool(radar_points),dino_curve=bool(dx)))
            index.append(f'- [{label}]({filename})'+(' — '+note if note else ''))
        index.append('')
    (plots/'README.md').write_text('\n'.join(index)+'\n')
    (plots/'catalog.json').write_text(json.dumps(catalog,indent=2)+'\n')
    # Visualize the actual optimization output and original selected pixels.
    with np.load(output/'optimized_maps/00.npz') as f: amplitude=f['amplitude']
    from scripts.ss50_radargen_count_sweep import ARCHIVE
    import h5py
    scene,frame=manifest['identities'][0]
    with h5py.File(ARCHIVE/'predictions.h5','r') as h:
        pd=h[f'scenes/{scene}/frames/{frame:06d}/fields/point_density'][()]
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    for ax,array,title in zip(axes,[pd,amplitude,amplitude>.1],['Predicted density map','Optimized amplitudes','Selected cells (threshold 0.1)']):
        im=ax.imshow(array,origin='upper',cmap='magma');ax.set_title(title);fig.colorbar(im,ax=ax,shrink=.7)
    fig.tight_layout();fig.savefig(output/'recovery_example.png',dpi=180);plt.close(fig)
    print(f'Reports and individual plots saved to {output}',flush=True)
