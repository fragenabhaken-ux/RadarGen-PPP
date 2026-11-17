"""2x2: CD, object hit rate, position-only F1, and PR with right-axis iso-F1."""
import json
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scripts.plot_final_dino_six_panel import value

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'output/paper_material/count_sweep_evaluation'
DEST=ROOT/'output/paper_material/count_sweep_comparisons/single_column/dino_final'
CACHE=DEST/'position_only'
BLUE='#2166ac';RED='#b2182b';GRAY='#888888'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir',type=Path,default=BASE/'ss50_sampling_budget')
    parser.add_argument('--native-summary',default='baseline_dino.json')
    parser.add_argument('--output',type=Path,default=DEST)
    parser.add_argument('--model-key',default='dv3')
    parser.add_argument('--model-label',default='CP3-dar-Dv3 (ours)')
    parser.add_argument('--text-scale',type=float,default=1.,help='Scale all titles, labels, tick text, annotations, and legend')
    parser.add_argument('--legend-text-scale',type=float,default=1.)
    parser.add_argument('--legend-y',type=float,default=.005)
    parser.add_argument('--legend-columnspacing',type=float,default=2.)
    args=parser.parse_args()
    if args.text_scale<=0:parser.error("text-scale must be positive")
    text_scale=args.text_scale
    output=args.output;cache=output/'position_only';model_key=args.model_key
    assert json.loads((cache/'provenance.json').read_text())['runs']==79
    assert json.loads((cache/'provenance.json').read_text())['checkpoint']==json.loads((args.model_dir/'experiment.json').read_text())['checkpoint']
    groups={}
    for f in sorted(args.model_dir.glob('budget_*/summary.json')):
        groups.setdefault(int(f.parent.name.split('_')[1]),[]).append(json.loads(f.read_text()))
    radar=[json.loads(f.read_text()) for f in sorted((BASE/'ss50_radargen_count_sweep').glob('count_*/summary.json'))]
    native=json.loads((args.model_dir/args.native_summary).read_text())
    rn=json.loads((BASE/'ss50_radargen_count_sweep/native_threshold_0p1/summary.json').read_text())
    budgets=sorted(groups);gt=value(native,'metrics','gt_count')
    dx=np.array([np.mean([value(s,'metrics','generated_count') for s in groups[b]]) for b in budgets])
    rx=np.array([value(s,'metrics','generated_count') for s in radar])
    position={}
    for name in [model_key,'radargen']:
        rows={}
        for file in cache.glob(name+'_*.json'):
            s=json.loads(file.read_text())
            if s.get('target') is not None:rows.setdefault(s['target'],[]).append(s)
        result={}
        for key in ['precision','recall','f1','count']:
            vals=np.array([[s['mean'][key] for s in rows[b]] for b in sorted(rows)])
            result[key]=vals.mean(axis=1);result[key+'_sd']=vals.std(axis=1)
        result['targets']=sorted(rows)
        result['native']=json.loads((cache/(name+'_native.json')).read_text())['mean']
        position[name]=result
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8*text_scale,'axes.linewidth':.6})
    fig,axes=plt.subplots(2,2,figsize=(6.5,3.35))
    def style(ax,title):
        ax.set_title(title,loc='left',fontsize=8.5*text_scale,pad=3)
        ax.spines[['top','right']].set_visible(False);ax.grid(alpha=.2,lw=.45);ax.set_axisbelow(True)
        ax.tick_params(labelsize=7.5*text_scale,length=2.5,width=.6,pad=2)
    for ax,(scope,key,title) in zip(axes.flat[:2],[('metrics','chamfer_distance_xy','(a) Location Chamfer (m) ↓'),
                                                ('foreground','hit_rate','(b) Object hit rate ↑')]):
        vals=np.array([[value(s,scope,key) for s in groups[b]] for b in budgets]);mean,sd=vals.mean(axis=1),vals.std(axis=1)
        ax.plot(dx,mean,color=BLUE,lw=1.5);ax.fill_between(dx,mean-sd,mean+sd,color=BLUE,alpha=.14,lw=0)
        ax.plot(rx,[value(s,scope,key) for s in radar],color=RED,lw=1.5)
        for s,color,marker,size in [(native,BLUE,'o',25),(rn,RED,'*',45)]:
            ax.scatter(value(s,'metrics','generated_count'),value(s,scope,key),color=color,marker=marker,s=size,edgecolor='white',lw=.4,zorder=5)
        style(ax,title);ax.set_xlim(0,2050);ax.set_xticks([0,500,1000,1500,2000]);ax.margins(y=.1)
        ax.axvline(gt,color=GRAY,ls=(0,(2,2)),lw=.8)
        ax.set_xlabel('Mean points/frame',fontsize=8*text_scale,labelpad=2)
    ax=axes[1,0]
    for name,color,marker,size in [(model_key,BLUE,'o',25),('radargen',RED,'*',45)]:
        d=position[name];ax.plot(d['count'],d['f1'],color=color,lw=1.5)
        if name==model_key:ax.fill_between(d['count'],d['f1']-d['f1_sd'],d['f1']+d['f1_sd'],color=color,alpha=.14,lw=0)
        ax.scatter(d['native']['count'],d['native']['f1'],color=color,marker=marker,s=size,edgecolor='white',lw=.4,zorder=5)
    style(ax,'(c) Position F1 @ 1 m ↑');ax.set_xlim(0,2050);ax.set_xticks([0,500,1000,1500,2000])
    ax.axvline(gt,color=GRAY,ls=(0,(2,2)),lw=.8);ax.margins(y=.1)
    ax.set_xlabel('Mean points/frame',fontsize=8*text_scale,labelpad=2)
    ax=axes[1,1]
    recalls=np.concatenate([d['recall'] for d in position.values()]);precisions=np.concatenate([d['precision'] for d in position.values()])
    xmax=float(np.ceil((recalls.max()+.04)/.05)*.05)
    ymin=max(0.,float(np.floor((precisions.min()-.045)/.05)*.05))
    ymax=min(1.,float(np.ceil((precisions.max()+.045)/.05)*.05))
    rr=np.linspace(.001,xmax,600);pp=np.linspace(max(.001,ymin),ymax,450)
    rg,pg=np.meshgrid(rr,pp);f1=2*rg*pg/(rg+pg)
    levels=np.arange(.05,1.,.05)
    ax.contour(rg,pg,f1,levels=levels,colors=GRAY,linestyles='--',linewidths=.7,zorder=1)
    annotations=[]
    for name,color,marker,size in [(model_key,BLUE,'o',25),('radargen',RED,'*',45)]:
        d=position[name];ax.plot(d['recall'],d['precision'],color=color,lw=1.35,zorder=3)
        ax.scatter(d['recall'],d['precision'],color=color,marker='x',s=16,lw=.75,zorder=4)
        ax.scatter(d['native']['recall'],d['native']['precision'],color=color,marker=marker,s=size,edgecolor='white',lw=.4,zorder=5)
        for target in ([100,1000,2000] if name==model_key else [100,1000,1700]):
            i=d['targets'].index(target)
            if target==100:offset=(5,6) if name==model_key else (5,5)
            elif target==1000:offset=(-7,9) if name==model_key else (-5,-12)
            else:offset=(-20,7) if name==model_key else (5,-11)
            text=f'{d["count"][i]:,.0f}'
            ax.annotate(text,(d['recall'][i],d['precision'][i]),xytext=offset,textcoords='offset points',
                fontsize=6.75*text_scale,color=color,zorder=6,bbox=dict(facecolor='white',alpha=.8,edgecolor='none',pad=.25))
            annotations.append(dict(model=name,target=target,actual_mean_count=float(d['count'][i]),label=text))
    style(ax,'(d) Position PR @ 1 m')
    ax.set_xlim(0,xmax);ax.set_ylim(ymin,ymax)
    ax.set_xlabel('Recall ↑',fontsize=8*text_scale,labelpad=2);ax.set_ylabel('Precision ↑',fontsize=8*text_scale,labelpad=2)
    # Right-edge values label the constant-F1 curves where they meet R=xmax.
    right=ax.twinx();right.set_ylim(ymin,ymax)
    right.spines[['top','left','bottom']].set_visible(False);right.spines['right'].set_linewidth(.6)
    ticks=[];labels=[]
    for level in levels:
        y=level*xmax/(2*xmax-level) if level<2*xmax else -1
        if ymin+.005<=y<=ymax-.005:ticks.append(y);labels.append(f'{level:.2f}')
    right.set_yticks(ticks,labels);right.tick_params(axis='y',labelsize=7*text_scale,colors=GRAY,length=2,width=.6,pad=3)
    right.set_ylabel('F1 score ↑',fontsize=8.5*text_scale,color=GRAY,labelpad=5);right.grid(False)
    handles=[Line2D([],[],color=BLUE,lw=1.5,marker='o',markersize=4,label=args.model_label+' Count Sweep'),
             Line2D([],[],color=RED,lw=1.5,marker='*',markersize=6,label='RadarGen Count Sweep'),
             Line2D([],[],color=GRAY,ls=(0,(2,2)),lw=.8,label=f'GT mean ({gt:,.0f}; a–c)')]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,args.legend_y),ncol=3,fontsize=7.5*text_scale*args.legend_text_scale,frameon=False,labelspacing=.3,columnspacing=args.legend_columnspacing)
    fig.subplots_adjust(left=.065,right=.895,top=.93,bottom=.18,wspace=.32,hspace=.62)
    for ext in ['png','pdf','svg']:fig.savefig(output/f'four_panel_position.{ext}',dpi=250,bbox_inches='tight',pad_inches=.02)
    plt.close(fig)
    (output/'four_panel_position_notes.json').write_text(json.dumps(dict(model_label=args.model_label,text_scale=text_scale,legend_text_scale=args.legend_text_scale,legend_y=args.legend_y,checkpoint=json.loads((cache/'provenance.json').read_text())['checkpoint'],position_metric='Hungarian one-to-one matching within <=1 m; RCS/Doppler gates disabled.',
        right_axis='F1 contour values at the right recall boundary; not a precision-to-F1 conversion for arbitrary recall.',
        aggregation='Panel c: mean per-frame F1 across frames/seeds. Panel d contours: 2*meanP*meanR/(meanP+meanR).',
        annotations=annotations),indent=2))
    print(output/'four_panel_position.png',flush=True)


if __name__=='__main__':main()
