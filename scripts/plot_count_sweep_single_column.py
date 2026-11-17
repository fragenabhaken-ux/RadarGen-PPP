"""Single-paper-column count-sweep figures, including DA-F1 and count-induced PR."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scripts.plot_final_dino_six_panel import value

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'output/paper_material/count_sweep_evaluation'
OUT=ROOT/'output/paper_material/count_sweep_comparisons/single_column'
SPECS=[('dino_final','ss50_sampling_budget','baseline_dino.json','CP3-dar-Dv3 (ours)'),
       ('sana','ss50_dit_no_bottleneck_count_sweep','native/summary.json','CP3-dar-SANA (ours)'),
       ('dino_compact','ss50_dino_compact_count_sweep','native/summary.json','CP3-dar-Dv3 (compact)'),
       ('dino_random','ss50_dino_random_count_sweep','native/summary.json','CP3-dar-Dv3 (random)'),
       ('rn50','ss50_resnet50_count_sweep','native/summary.json','CP3-dar-RN50 (ours)')]
PANELS=[('metrics','chamfer_distance_xy','Location Chamfer distance (m) ↓'),
        ('foreground','hit_rate','Object hit rate ↑'),
        ('metrics','distance_attr_f1','DA F1 ↑')]
BLUE='#2166ac';RED='#b2182b';GRAY='#777777'


def setup(ax,title):
    ax.set_title(title,loc='left',fontsize=7.5,pad=3)
    ax.grid(alpha=.22,lw=.5);ax.set_axisbelow(True)
    ax.spines[['top','right']].set_visible(False)
    ax.tick_params(labelsize=6.5,length=2.5,width=.6,pad=2)
    ax.margins(y=.12)


def save(fig,path,*,tight=False):
    options=dict(bbox_inches="tight",pad_inches=.015) if tight else {}
    for ext in ['png','pdf','svg']:fig.savefig(path.with_suffix('.'+ext),dpi=250,**options)
    plt.close(fig)


def main():
    OUT.mkdir(exist_ok=True,parents=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':7,'axes.linewidth':.6})
    rdir=BASE/'ss50_radargen_count_sweep'
    radar=[json.loads(f.read_text()) for f in sorted(rdir.glob('count_*/summary.json'))]
    rn=json.loads((rdir/'native_threshold_0p1/summary.json').read_text())
    rm=json.loads((rdir/'experiment.json').read_text())
    index=['# Count sweeps: single-column paper figures','',
        'Each figure is 3.45 inches wide. Three-panel figures use one column of three axes: location Chamfer, object hit rate, and DA-F1. '
        'F1 is the existing distance-and-attribute one-to-one matching score, averaged per frame, then across sampling seeds. '
        'Blue shading is ±1 population SD across three sampling seeds. '
        'Markers indicate original operating points. The PR plot varies generated count, not a detection confidence threshold; it does not define AP.','']
    for slug,folder,native_path,label in SPECS:
        directory=BASE/folder;destination=OUT/slug;destination.mkdir(exist_ok=True)
        manifest=json.loads((directory/'experiment.json').read_text())
        assert manifest['ss50_identities']==rm['identities']
        groups={}
        for file in sorted(directory.glob('budget_*/summary.json')):
            summary=json.loads(file.read_text());assert summary['complete'] and summary['evaluated_samples']==50
            groups.setdefault(int(file.parent.name.split('_')[1]),[]).append(summary)
        assert len(groups)==20 and all(len(g)==3 for g in groups.values())
        native=json.loads((directory/native_path).read_text());assert native['checkpoint']==manifest['checkpoint']
        gt=value(native,'metrics','gt_count');budgets=sorted(groups)
        x=np.array([np.mean([value(s,'metrics','generated_count') for s in groups[b]]) for b in budgets])
        rx=np.array([value(s,'metrics','generated_count') for s in radar])
        def series(scope,key):
            data=np.array([[value(s,scope,key) for s in groups[b]] for b in budgets])
            return data.mean(axis=1),data.std(axis=1)
        handles=[Line2D([],[],color=BLUE,lw=1.5,marker='o',markersize=4,label=label+' Count Sweep'),
                 Line2D([],[],color=RED,lw=1.5,marker='*',markersize=6,label='RadarGen Count Sweep'),
                 Line2D([],[],color=GRAY,ls=(0,(2,2)),lw=.8,label=f'GT mean ({gt:,.0f})')]
        def draw(ax,scope,key):
            mean,sd=series(scope,key)
            ax.plot(x,mean,color=BLUE,lw=1.5);ax.fill_between(x,mean-sd,mean+sd,color=BLUE,alpha=.14,lw=0)
            ax.plot(rx,[value(s,scope,key) for s in radar],color=RED,lw=1.5)
            ax.axvline(gt,color=GRAY,ls=(0,(2,2)),lw=.8)
            for s,color,marker,size in [(native,BLUE,'o',25),(rn,RED,'*',45)]:
                ax.scatter(value(s,'metrics','generated_count'),value(s,scope,key),color=color,marker=marker,s=size,edgecolor='white',lw=.4,zorder=5)
            ax.set_xlim(0,2050);ax.set_xticks([0,500,1000,1500,2000])
        fig,axes=plt.subplots(3,1,figsize=(3.45,4.35),sharex=True)
        for i,(ax,(scope,key,title)) in enumerate(zip(axes,PANELS)):
            draw(ax,scope,key);setup(ax,f'({chr(97+i)}) {title}')
        axes[-1].set_xlabel('Mean generated points per frame',fontsize=7.5,labelpad=4)
        fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.007),ncol=1,frameon=False,fontsize=6.5,labelspacing=.3)
        fig.subplots_adjust(left=.095,right=.97,top=.97,bottom=.20,hspace=.22)
        save(fig,destination/'three_metrics',tight=True)
        fig,ax=plt.subplots(figsize=(3.45,2.9));draw(ax,'metrics','distance_attr_f1');setup(ax,'Distance-and-attribute F1 ↑')
        ax.set_xlabel('Mean generated points per frame',fontsize=7.5,labelpad=4)
        fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.005),ncol=1,frameon=False,fontsize=6.5,labelspacing=.3)
        fig.subplots_adjust(left=.15,right=.98,top=.90,bottom=.32)
        save(fig,destination/'f1_count_sweep')
        fig,ax=plt.subplots(figsize=(3.45,2.9))
        recall,_=series('metrics','distance_attr_recall');precision,_=series('metrics','distance_attr_precision')
        ax.plot(recall,precision,color=BLUE,lw=1.5)
        ax.plot([value(s,'metrics','distance_attr_recall') for s in radar],
                [value(s,'metrics','distance_attr_precision') for s in radar],color=RED,lw=1.5)
        for s,color,marker,size in [(native,BLUE,'o',25),(rn,RED,'*',45)]:
            ax.scatter(value(s,'metrics','distance_attr_recall'),value(s,'metrics','distance_attr_precision'),
                       color=color,marker=marker,s=size,edgecolor='white',lw=.4,zorder=5)
        setup(ax,'Precision–recall across the count sweep')
        ax.set_xlabel('DA recall',fontsize=7.5);ax.set_ylabel('DA precision',fontsize=7.5)
        fig.legend(handles=handles[:2],loc='lower center',bbox_to_anchor=(.5,.01),frameon=False,ncol=1,fontsize=7.5,labelspacing=.3)
        fig.subplots_adjust(left=.18,right=.98,top=.90,bottom=.29)
        save(fig,destination/'precision_recall')
        mean,sd=series('metrics','distance_attr_f1');best=int(np.argmax(mean))
        report=dict(label=label,checkpoint=manifest['checkpoint'],width_inches=3.45,
            best_tested_f1=dict(target_expected_mean_count=budgets[best],realized_mean_count=float(x[best]),mean=float(mean[best]),seed_sd=float(sd[best])),
            native_f1=value(native,'metrics','distance_attr_f1'),radargen_native_f1=value(rn,'metrics','distance_attr_f1'))
        (destination/'summary.json').write_text(json.dumps(report,indent=2))
        index += [f'## {label}','']
        for name,title in [('three_metrics','Three metrics, one column'),('f1_count_sweep','F1 versus count'),('precision_recall','Precision–recall across count')]:
            index.append(f'- {title}: [PNG]({slug}/{name}.png) · [PDF]({slug}/{name}.pdf) · [SVG]({slug}/{name}.svg)')
        index.append('')
        print(label,report['best_tested_f1'],flush=True)
    (OUT/'README.md').write_text('\n'.join(index))


if __name__=='__main__':main()
