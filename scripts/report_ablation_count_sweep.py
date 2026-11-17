"""One ablation versus RadarGen only: all numeric metrics, tables, individual PNGs."""
import csv
import json
from pathlib import Path
import numpy as np

RADAR=Path('output/paper_material/count_sweep_evaluation/ss50_radargen_count_sweep')
LABELS={
 'chamfer_distance_xy':'CD location ↓','chamfer_distance_all_norm':'CD full ↓',
 'distance_attr_f1':'Detection F1 ↑','distance_attr_recall':'Detection recall ↑','distance_attr_precision':'Detection precision ↑',
 'iou_th1':'IoU @ 1 m ↑','count_abs_error':'Point-count absolute error ↓',
 'pcl_mmd_xy':'MMD location ↓','pcl_mmd_rcs':'MMD RCS ↓','pcl_mmd_doppler':'MMD Doppler ↓',
 'rcs_mae_at_detections':'RCS MAE at GT detections ↓','doppler_mae_at_detections':'Doppler MAE at GT detections ↓',
 'per_box_chamfer_distance_xy':'Foreground CD location ↓','per_box_chamfer_distance_all_norm':'Foreground CD full ↓',
 'per_box_num_points_similarity':'Density similarity ↑','hit_rate':'Hit rate ↑',
 'ppp_expected_count':'PPP expected count per box','ppp_expected_count_abs_error':'PPP expected-count error per box ↓',
 'ppp_expected_total_count':'PPP expected total count','ppp_total_count_abs_error':'PPP expected total-count error ↓',
 'ppp_hit_probability_gt_positive':'PPP hit probability in GT-positive boxes ↑',
 'generated_count':'Mean generated points','gt_count':'Mean GT points','target_collision_count':'GT raster collisions (diagnostic)'}


def value(summary,scope,key):
    v=summary[scope].get(key)
    return v.get('mean') if isinstance(v,dict) else v


def report(output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output=Path(output);manifest=json.loads((output/'experiment.json').read_text())
    label=manifest['label']
    native=json.loads((output/'native/summary.json').read_text())
    radar=json.loads((RADAR/'native_threshold_0p1/summary.json').read_text())
    rs=[json.loads(f.read_text()) for f in sorted(RADAR.glob('count_*/summary.json'))]
    groups={}
    for file in sorted(output.glob('budget_*/summary.json')):
        budget=int(file.parent.name.split('_')[1]);s=json.loads(file.read_text())
        assert s['complete'] and s['evaluated_samples']==50
        groups.setdefault(budget,[]).append(s)
    assert len(groups)==20 and all(len(g)==3 for g in groups.values())
    gt=value(native,'metrics','gt_count');assert gt==value(radar,'metrics','gt_count')
    scopes=['metrics','foreground']
    keys={scope:sorted({k for summary in [native,radar,*rs,*[s for g in groups.values() for s in g]]
                            for k in summary[scope] if isinstance(value(summary,scope,k),(int,float))}) for scope in scopes}
    def aggregate(group):
        result={scope:{} for scope in scopes}
        for scope in scopes:
            for key in keys[scope]:
                vv=[value(s,scope,key) for s in group if value(s,scope,key) is not None]
                if vv:result[scope][key]={'mean':float(np.mean(vv)),'seed_sd':float(np.std(vv))}
        return result
    records=[(f'Target {b}',aggregate(g),False) for b,g in sorted(groups.items())]
    records += [(label+' native',native,True),('RadarGen native threshold 0.1',radar,True)]
    records.sort(key=lambda r:value(r[1],'metrics','generated_count'))
    csvrows=[]
    for name,s,_ in records:
        row={'operating_point':name}
        for scope in scopes:
            for key in keys[scope]:
                v=value(s,scope,key)
                if isinstance(v,(int,float)):
                    row[scope+'.'+key]=v
                    if isinstance(s[scope][key],dict) and 'seed_sd' in s[scope][key]:row[scope+'.'+key+'_seed_sd']=s[scope][key]['seed_sd']
        csvrows.append(row)
    columns=['operating_point']+sorted({k for r in csvrows for k in r if k!='operating_point'})
    with (output/'results.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=columns);writer.writeheader();writer.writerows(csvrows)
    lines=[f'# {label}: SS50 count sweep versus RadarGen','',f'GT mean: **{gt:,.2f} points/frame**. Twenty expected-mean count targets, 100–2000 in steps of 100, three sampling seeds each. One shared intensity multiplier preserves scene-density differences. Table sweep entries are seed means; CSV includes seed SD.','',
        'Only this model and RadarGen are included. Native model points come from the same archived maps used by the sweep. RadarGen uses its existing fixed-map threshold sweep (100–1700); maximum attainable mean with unchanged optimization is 1750.04. Its native threshold 0.1 baseline uses those same maps. No historical models or historical RadarGen reference are included.','',
        'Analytical PPP metrics have no RadarGen equivalent. GT counts, collisions and evaluated-box counts are diagnostics. Mark MAE at fixed GT detections remains constant; target-cell MAE entries are compatibility aliases. Timing entries describe different postprocessing stages and are not model-speed comparisons.','']
    for scope,tablekeys,title in [('metrics',['generated_count','count_abs_error','chamfer_distance_xy','chamfer_distance_all_norm','iou_th1','distance_attr_recall','distance_attr_precision','distance_attr_f1','pcl_mmd_xy','pcl_mmd_rcs','pcl_mmd_doppler'],'Whole area'),
        ('foreground',['per_box_chamfer_distance_xy','per_box_chamfer_distance_all_norm','per_box_num_points_similarity','hit_rate'],'Foreground')]:
        lines += [f'## {title}','','| Operating point | '+' | '.join(LABELS.get(k,k) for k in tablekeys)+' |','| --- | '+' | '.join(['---:']*len(tablekeys))+' |']
        ordered=[(value(s,'metrics','generated_count'),name,s,bold) for name,s,bold in records]
        ordered.append((gt,'Ground truth',None,True));ordered.sort(key=lambda r:r[0])
        for count,name,s,bold in ordered:
            cells=[name+(f' ({count:,.2f} points/frame)' if bold else '')]
            for key in tablekeys:
                v=gt if s is None and key=='generated_count' else None if s is None else value(s,scope,key)
                cells.append('' if v is None else f'{v:.3f}')
            if bold:cells=[f'**{c}**' if c else '' for c in cells]
            lines.append('| '+' | '.join(cells)+' |')
        lines.append('')
    lines+=['[All metrics CSV](results.csv) · [All individual plots](plots/README.md)','',f'Checkpoint: `{manifest["checkpoint"]}`.','']
    (output/'results.md').write_text('\n'.join(lines))
    plots=output/'plots';plots.mkdir(exist_ok=True)
    index=[f'# {label} versus RadarGen: all individual metrics','',
        'One metric per PNG. Blue: model PPP sweep, three-seed mean ± SD. Red: RadarGen shared-threshold sweep. Baseline markers use native archived points on the fixed fields. X is realized mean points per frame.','']
    catalog=[]
    for scope in scopes:
        index += [f'## {"Whole area" if scope=="metrics" else "Foreground"}','']
        for key in keys[scope]:
            title=LABELS.get(key,key.replace('_',' ').capitalize())
            if key.startswith('invariant_mmd_'):
                parts=key.split('_');title=f'Foreground {parts[-1]} MMD {"location" if parts[-2]=="xy" else parts[-2].upper()} ↓'
            fig,ax=plt.subplots(figsize=(7.5,6))
            rpoints=[(value(s,'metrics','generated_count'),value(s,scope,key)) for s in rs if value(s,scope,key) is not None]
            if rpoints:ax.plot(*zip(*rpoints),lw=2,color='#b2182b',label='RadarGen threshold sweep')
            xx=[];yy=[];sd=[]
            for budget,group in sorted(groups.items()):
                vv=[value(s,scope,key) for s in group if value(s,scope,key) is not None]
                if vv:
                    xx.append(np.mean([value(s,'metrics','generated_count') for s in group]));yy.append(np.mean(vv));sd.append(np.std(vv))
            if xx:
                yy=np.asarray(yy);sd=np.asarray(sd)
                ax.plot(xx,yy,lw=2,color='#1b6ca8',label=label+' sweep')
                ax.fill_between(xx,yy-sd,yy+sd,color='#1b6ca8',alpha=.15)
            ax.axvline(gt,ls=':',color='gray',label=f'GT mean: {gt:,.2f}')
            for name,s,marker,color in [(label+' native',native,'D','#1b6ca8'),('RadarGen native',radar,'*','#b2182b')]:
                v=value(s,scope,key)
                if v is not None:
                    count=value(s,'metrics','generated_count')
                    ax.scatter(count,v,marker=marker,s=90,color=color,edgecolor='white',linewidth=.6,zorder=5,label=f'{name}: {count:,.2f}')
            note=''
            if key.startswith('ppp_'):note='Analytical PPP metric — no RadarGen equivalent'
            elif key in ['gt_count','target_collision_count','gt_positive_boxes','evaluated_boxes']:note='Diagnostic; not a prediction-quality metric'
            elif 'mae_at_' in key:note='Fixed GT locations and fields; independent of sampled count'
            elif 'time_seconds' in key:note='Postprocessing timing; stages differ between models'
            if note:ax.text(.02,.97,note,transform=ax.transAxes,va='top',fontsize=8,bbox=dict(facecolor='white',alpha=.85,edgecolor='none'))
            ax.set(xlabel='Realized mean points per frame',ylabel=title,title=title);ax.grid(alpha=.2)
            ax.spines[['top','right']].set_visible(False)
            handles,legends=ax.get_legend_handles_labels()
            fig.legend(handles,legends,loc='lower center',ncol=1,fontsize=8,frameon=False,bbox_to_anchor=(.5,.01))
            fig.suptitle('SS50 · '+label+' versus RadarGen',fontsize=11)
            fig.tight_layout(rect=(0,.25,1,.95));fig.savefig(plots/(key+'.png'),dpi=180,bbox_inches='tight');plt.close(fig)
            catalog.append(dict(scope=scope,key=key,filename=key+'.png',model_curve=bool(xx),radargen_curve=bool(rpoints)))
            index.append(f'- [{title}]({key}.png)'+(' — '+note if note else ''))
        index.append('')
    (plots/'README.md').write_text('\n'.join(index)+'\n')
    (plots/'catalog.json').write_text(json.dumps(catalog,indent=2))
    print(f'Reports saved: {output}',flush=True)
