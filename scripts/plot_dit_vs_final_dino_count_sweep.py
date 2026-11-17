"""Compare only DiT no bottleneck 58,057 and final DINO 19,914 count sweeps."""
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts.report_ablation_count_sweep import LABELS, value

ROOT=Path('output/paper_material/count_sweep_evaluation')
DEST=Path('output/paper_material/count_sweep_comparisons/dit_58057_vs_dino_19914')
SPECS=[('DiT no bottleneck — step 58,057',ROOT/'ss50_dit_no_bottleneck_count_sweep','native/summary.json','#b2182b','D'),
       ('DINO final — step 19,914',ROOT/'ss50_sampling_budget','baseline_dino.json','#1b6ca8','o')]


def main():
    DEST.mkdir(parents=True,exist_ok=True)
    plots=DEST/'plots';plots.mkdir(exist_ok=True)
    models=[]
    for label,directory,native_path,color,marker in SPECS:
        manifest=json.loads((directory/'experiment.json').read_text())
        groups={}
        for file in sorted(directory.glob('budget_*/summary.json')):
            s=json.loads(file.read_text());assert s['complete'] and s['evaluated_samples']==50
            b=int(file.parent.name.split('_')[1]);groups.setdefault(b,[]).append(s)
        assert len(groups)==20 and all(len(g)==3 for g in groups.values())
        native=json.loads((directory/native_path).read_text())
        assert native['checkpoint']==manifest['checkpoint']
        models.append(dict(label=label,directory=directory,manifest=manifest,groups=groups,native=native,color=color,marker=marker))
    a,b=models
    assert '_step_58057.pth' in a['manifest']['checkpoint']
    assert '_step_19914.pth' in b['manifest']['checkpoint']
    assert a['manifest']['ss50_identities']==b['manifest']['ss50_identities']
    assert a['manifest']['budgets']==b['manifest']['budgets']
    assert a['manifest']['seeds']==b['manifest']['seeds']
    # Check frame order and per-frame GT counts, not just aggregate equality.
    def rows(model):
        return [json.loads(line) for line in (model['directory']/'budget_1000_seed_42/metrics.jsonl').read_text().splitlines()]
    aa,bb=rows(a),rows(b)
    assert [(r['scene'],r['frame'],r['metrics']['gt_count']) for r in aa]==[(r['scene'],r['frame'],r['metrics']['gt_count']) for r in bb]
    gt=value(a['native'],'metrics','gt_count');assert gt==value(b['native'],'metrics','gt_count')
    summaries=[s for model in models for s in [model['native'],*[s for group in model['groups'].values() for s in group]]]
    keys={scope:sorted({k for s in summaries for k in s[scope] if isinstance(value(s,scope,k),(int,float))}) for scope in ['metrics','foreground']}
    catalog=[];csvrows=[]
    index=['# DiT no bottleneck 58,057 versus final DINO 19,914','',
        'SS50; expected mean count targets 100–2000, three sampling seeds per target. Both curves use one shared intensity multiplier per model, preserving scene density differences. The x-axis is realized mean count. Bands show ±1 population SD across the three seed aggregates. Only these two models are included.','',
        'Native markers use original recorded sampled points; DINO is checkpoint 19,914, not 15,000. Raster collisions/GT counts are diagnostics. MAE at fixed GT detections stays constant; target-cell MAE keys are compatibility aliases. Sampling and native inference timings are not a controlled speed comparison.','']
    for scope in ['metrics','foreground']:
        index += [f'## {"Whole area" if scope=="metrics" else "Foreground"}','']
        for key in keys[scope]:
            title=LABELS.get(key,key.replace('_',' ').capitalize())
            if key.startswith('invariant_mmd_'):
                parts=key.split('_');title=f'Foreground {parts[-1]} MMD {"location" if parts[-2]=="xy" else parts[-2].upper()} ↓'
            fig,ax=plt.subplots(figsize=(7.5,6))
            curves=[]
            for model in models:
                xx=[];yy=[];std=[]
                for budget,group in sorted(model['groups'].items()):
                    vv=[value(s,scope,key) for s in group if value(s,scope,key) is not None]
                    if vv:
                        xx.append(np.mean([value(s,'metrics','generated_count') for s in group]))
                        yy.append(np.mean(vv));std.append(np.std(vv))
                if xx:
                    yy=np.asarray(yy);std=np.asarray(std)
                    ax.plot(xx,yy,color=model['color'],lw=2,label=model['label'])
                    ax.fill_between(xx,yy-std,yy+std,color=model['color'],alpha=.15)
                curves.append(bool(xx))
                native=model['native'];v=value(native,scope,key)
                if v is not None:
                    count=value(native,'metrics','generated_count')
                    ax.scatter(count,v,marker=model['marker'],s=85,color=model['color'],edgecolor='white',linewidth=.6,zorder=5,
                        label=f'{model["label"]} native: {count:,.2f}')
            ax.axvline(gt,ls=':',color='gray',label=f'GT mean: {gt:,.2f}')
            note=''
            if key in ['gt_count','target_collision_count','gt_positive_boxes','evaluated_boxes']:note='Diagnostic; not a prediction-quality metric'
            elif 'mae_at_' in key:note='Fixed GT locations and fields; independent of sampled count'
            elif 'time_seconds' in key:note='Timing diagnostic; not a controlled model-speed comparison'
            if note:ax.text(.02,.97,note,transform=ax.transAxes,va='top',fontsize=8,bbox=dict(facecolor='white',alpha=.85,edgecolor='none'))
            ax.set(xlabel='Realized mean points per frame',ylabel=title,title=title);ax.grid(alpha=.2)
            ax.spines[['top','right']].set_visible(False)
            handles,labels=ax.get_legend_handles_labels()
            fig.legend(handles,labels,loc='lower center',ncol=1,fontsize=8,frameon=False,bbox_to_anchor=(.5,.01))
            fig.suptitle('SS50 · DiT 58,057 versus final DINO 19,914',fontsize=12)
            fig.tight_layout(rect=(0,.25,1,.95));fig.savefig(plots/(key+'.png'),dpi=180,bbox_inches='tight');plt.close(fig)
            index.append(f'- [{title}]({key}.png)'+(' — '+note if note else ''))
            catalog.append(dict(scope=scope,key=key,filename=key+'.png',dit_curve=curves[0],dino_curve=curves[1]))
        index.append('')
    for model in models:
        for budget,group in sorted(model['groups'].items()):
            row={'model':model['label'],'target_expected_mean_count':budget,'operating_point':'sweep'}
            for scope in keys:
                for key in keys[scope]:
                    vv=[value(s,scope,key) for s in group if value(s,scope,key) is not None]
                    if vv:row[scope+'.'+key]=np.mean(vv);row[scope+'.'+key+'_seed_sd']=np.std(vv)
            csvrows.append(row)
        row={'model':model['label'],'target_expected_mean_count':'','operating_point':'native'}
        for scope in keys:
            for key in keys[scope]:
                v=value(model['native'],scope,key)
                if v is not None:row[scope+'.'+key]=v
        csvrows.append(row)
    columns=['model','operating_point','target_expected_mean_count']+sorted({k for row in csvrows for k in row if k not in ['model','operating_point','target_expected_mean_count']})
    with (DEST/'results.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=columns);writer.writeheader();writer.writerows(csvrows)
    (plots/'README.md').write_text('\n'.join(index))
    (plots/'catalog.json').write_text(json.dumps(catalog,indent=2))
    (DEST/'provenance.json').write_text(json.dumps(dict(models=[dict(label=m['label'],source=str(m['directory']),checkpoint=m['manifest']['checkpoint']) for m in models],frames=50,budgets=a['manifest']['budgets'],seeds=a['manifest']['seeds'],frame_order_and_gt_counts_verified=True),indent=2))
    (DEST/'README.md').write_text('# DiT no bottleneck versus final DINO\n\nDiT step **58,057** versus DINO step **19,914** on the same SS50 frames. Twenty count targets, three seeds per target. Existing sweeps reused; no new evaluation needed.\n\n[All individual comparison plots](plots/README.md) · [Complete comparison CSV](results.csv)\n\nOnly these two models are included. See the plot index for diagnostic and timing caveats.\n')
    print(f'Saved {len(catalog)} individual comparisons to {DEST}',flush=True)


if __name__=='__main__':main()
