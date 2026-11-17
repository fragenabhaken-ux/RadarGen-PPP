"""Compact six-panel final-DINO versus RadarGen SS50 count-sweep figure."""
import json
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'output/paper_material/count_sweep_evaluation'
OUT=ROOT/'output/paper_material/count_sweep_comparisons/final_dino_vs_radargen_six_panel'
PANELS=[
 ('metrics','distance_attr_recall','Distance-and-attribute recall ↑'),
 ('metrics','distance_attr_precision','Distance-and-attribute precision ↑'),
 ('metrics','chamfer_distance_xy','Location Chamfer distance ↓'),
 ('foreground','per_box_num_points_similarity','Density similarity ↑'),
 ('metrics','iou_th1','BEV IoU @ 1 m ↑'),
 ('foreground','hit_rate','Object hit rate ↑'),
]


def value(s,scope,key):
    v=s[scope][key]
    return v['mean'] if isinstance(v,dict) else v


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir',type=Path,default=BASE/'ss50_sampling_budget')
    parser.add_argument('--native-summary',default='baseline_dino.json')
    parser.add_argument('--model-label',default='CP3-dar-Dv3 (ours)')
    parser.add_argument('--output',type=Path,default=OUT)
    args=parser.parse_args()
    output=args.output;output.mkdir(parents=True,exist_ok=True)
    model_label=args.model_label
    dino_dir=args.model_dir;radar_dir=BASE/'ss50_radargen_count_sweep'
    dino={}
    for file in sorted(dino_dir.glob('budget_*/summary.json')):
        summary=json.loads(file.read_text());assert summary['complete'] and summary['evaluated_samples']==50
        dino.setdefault(int(file.parent.name.split('_')[1]),[]).append(summary)
    assert len(dino)==20 and all(len(v)==3 for v in dino.values())
    radar=[json.loads(file.read_text()) for file in sorted(radar_dir.glob('count_*/summary.json'))]
    native_dino=json.loads((dino_dir/args.native_summary).read_text())
    native_radar=json.loads((radar_dir/'native_threshold_0p1/summary.json').read_text())
    dm=json.loads((dino_dir/'experiment.json').read_text())
    rm=json.loads((radar_dir/'experiment.json').read_text())
    assert dm['ss50_identities']==rm['identities']
    assert native_dino['checkpoint']==dm['checkpoint']
    gt=value(native_dino,'metrics','gt_count')
    blue='#2166ac';red='#b2182b';gray='#777777'
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,
        'axes.titlesize':9.5,'xtick.labelsize':9,'ytick.labelsize':8.5,
        'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.7,
        'grid.linewidth':.55,'savefig.facecolor':'white'})
    fig,axes=plt.subplots(3,2,figsize=(6.5,6.1),sharex=True)
    budgets=sorted(dino)
    for i,(ax,(scope,key,title)) in enumerate(zip(axes.flat,PANELS)):
        dx=[np.mean([value(s,'metrics','generated_count') for s in dino[b]]) for b in budgets]
        dy=np.array([[value(s,scope,key) for s in dino[b]] for b in budgets])
        mean,sd=dy.mean(axis=1),dy.std(axis=1)
        ax.fill_between(dx,mean-sd,mean+sd,color=blue,alpha=.14,lw=0)
        ax.plot(dx,mean,color=blue,lw=1.8,zorder=3)
        ax.plot([value(s,'metrics','generated_count') for s in radar],
                [value(s,scope,key) for s in radar],color=red,lw=1.8,zorder=3)
        ax.axvline(gt,color=gray,ls=(0,(2,2)),lw=1,zorder=1)
        for summary,color,marker,size in [(native_dino,blue,'o',40),(native_radar,red,'*',85)]:
            ax.scatter(value(summary,'metrics','generated_count'),value(summary,scope,key),
                       color=color,marker=marker,s=size,edgecolor='white',lw=.65,zorder=5)
        title=title.replace('Distance-and-attribute ', 'Distance-and-attribute\n')
        ax.set_title(f'({chr(97+i)}) {title}',loc='left',pad=4)
        ax.set_xlim(0,2050);ax.set_xticks([0,500,1000,1500,2000])
        ax.grid(alpha=.23);ax.set_axisbelow(True);ax.margins(y=.10)
        if key=='chamfer_distance_xy':ax.set_ylabel('m',labelpad=3)
        ax.tick_params(length=3,width=.7,pad=3)
    fig.supxlabel('Mean generated points per frame',fontsize=10.5,y=.111)
    dino_count=value(native_dino,'metrics','generated_count')
    radar_count=value(native_radar,'metrics','generated_count')
    # Matplotlib fills legend columns: pair each model's curve and marker by row.
    handles=[Line2D([],[],color=blue,lw=1.8,label=model_label+' Count Sweep'),
             Line2D([],[],color=red,lw=1.8,label='RadarGen Count Sweep'),
             Line2D([],[],color=blue,marker='o',ls='',markersize=5,label=model_label),
             Line2D([],[],color=red,marker='*',ls='',markersize=8,label='RadarGen'),
             Line2D([],[],color=gray,ls=(0,(2,2)),lw=1,label=f'GT mean ({gt:,.0f})')]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.008),ncol=3,
               frameon=False,fontsize=9.5,columnspacing=1.,handlelength=1.7,labelspacing=.35)
    fig.subplots_adjust(left=.085,right=.97,top=.94,bottom=.18,wspace=.17,hspace=.23)
    for extension in ['png','pdf','svg']:
        fig.savefig(output/f'count_sweep_six_panel.{extension}',dpi=220)
    plt.close(fig)
    (output/'README.md').write_text(
        f'# {model_label} versus RadarGen: six-panel count sweep\n\n'
        f'Model checkpoint: `{dm["checkpoint"]}`. Source: `{dino_dir}`. '
        'Same SS50 identities; expected mean counts 100–2000, three sampling seeds. '
        'Blue band is ±1 SD across the three model sampling seeds. '
        'RadarGen uses the deterministic shared-threshold sweep and its threshold-0.1 baseline on the same archived maps. '
        'Markers show original operating points; the historical RadarGen reference is excluded.\n\n'
        '[PNG preview](count_sweep_six_panel.png) · [PDF](count_sweep_six_panel.pdf) · [SVG](count_sweep_six_panel.svg)\n')
    (output/'provenance.json').write_text(json.dumps(dict(model_label=model_label,
        model_source=str(dino_dir),checkpoint=dm['checkpoint'],radargen_source=str(radar_dir),
        ss50_identities=dm['ss50_identities'],panels=PANELS),indent=2))
    print(output/'count_sweep_six_panel.png')



if __name__=='__main__':main()
