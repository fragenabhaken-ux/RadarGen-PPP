"""DV3-only count-induced precision/recall with iso-F1; replacement third panel."""
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
OUT=ROOT/'output/paper_material/count_sweep_comparisons/single_column/dino_final'
BLUE='#2166ac';RED='#b2182b';GRAY='#888888'


def main():
    groups={}
    for f in sorted((BASE/'ss50_sampling_budget').glob('budget_*/summary.json')):
        s=json.loads(f.read_text());assert s['complete'] and s['evaluated_samples']==50
        groups.setdefault(int(f.parent.name.split('_')[1]),[]).append(s)
    budgets=sorted(groups);assert len(budgets)==20 and all(len(g)==3 for g in groups.values())
    def series(scope,key):
        data=np.array([[value(s,scope,key) for s in groups[b]] for b in budgets])
        return data.mean(axis=1),data.std(axis=1)
    recall,_=series('metrics','distance_attr_recall');precision,_=series('metrics','distance_attr_precision')
    counts,_=series('metrics','generated_count')
    native=json.loads((BASE/'ss50_sampling_budget/baseline_dino.json').read_text())
    radar=[json.loads(f.read_text()) for f in sorted((BASE/'ss50_radargen_count_sweep').glob('count_*/summary.json'))]
    rn=json.loads((BASE/'ss50_radargen_count_sweep/native_threshold_0p1/summary.json').read_text())
    assert '_step_19914.pth' in native['checkpoint']
    gt=value(native,'metrics','gt_count')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':7,'axes.linewidth':.6})
    def style(ax,title):
        ax.set_title(title,loc='left',fontsize=7.5,pad=3)
        ax.spines[['top','right']].set_visible(False)
        ax.tick_params(labelsize=6.5,length=2.5,width=.6,pad=2)
        ax.grid(alpha=.18,lw=.4);ax.set_axisbelow(True)
    def draw_pr(ax,title):
        rr=np.linspace(.001,.47,600);pp=np.linspace(.20,.43,400)
        rg,pg=np.meshgrid(rr,pp);f1=2*rg*pg/(rg+pg)
        contour=ax.contour(rg,pg,f1,levels=[.05,.10,.15,.20,.25,.30,.35,.40],
                           colors=GRAY,linewidths=.7,linestyles='--',zorder=1)
        # Place horizontal F1 labels inside the axes, avoiding clipped edge labels.
        for level in [.05,.10,.15,.20,.25,.30,.35,.40]:
            py=.414 if level<.40 else .365
            px=level*py/(2*py-level)
            ax.text(px,py,f'{level:.2f}',ha='center',va='center',fontsize=6,color=GRAY,
                    bbox=dict(facecolor='white',edgecolor='none',alpha=.9,pad=.4),zorder=2)
        ax.plot(recall,precision,color=BLUE,lw=1.35,zorder=3)
        ax.scatter(recall,precision,marker='x',color=BLUE,s=17,linewidth=.8,zorder=4)
        offsets={100:(5,-12),500:(4,6),1000:(-5,8),1500:(-24,-11),2000:(4,-9)}
        labels=[]
        for target,(ox,oy) in offsets.items():
            i=budgets.index(target)
            ax.annotate(f'{counts[i]:,.0f}',(recall[i],precision[i]),xytext=(ox,oy),
                        textcoords='offset points',fontsize=6.5,color=BLUE,
                        bbox=dict(facecolor='white',edgecolor='none',alpha=.78,pad=.3),zorder=6)
            labels.append(dict(target=target,realized_mean_count=float(counts[i]),recall=float(recall[i]),precision=float(precision[i])))
        style(ax,title)
        ax.set(xlim=(0,.47),ylim=(.20,.43),xticks=[0,.1,.2,.3,.4],yticks=[.20,.25,.30,.35,.40])
        ax.set_xlabel('Recall ↑',fontsize=7.5,labelpad=3);ax.set_ylabel('Precision ↑',fontsize=7.5,labelpad=3)
        return labels
    def save(fig,name):
        for ext in ['png','pdf','svg']:fig.savefig(OUT/f'{name}.{ext}',dpi=260,bbox_inches='tight',pad_inches=.015)
        plt.close(fig)
    fig,ax=plt.subplots(figsize=(3.45,2.65))
    annotations=draw_pr(ax,'DA precision–recall · CP3-dar-Dv3 (ours)')
    fig.subplots_adjust(left=.14,right=.98,bottom=.18,top=.91)
    save(fig,'precision_recall_f1_contours')
    fig=plt.figure(figsize=(3.45,4.8))
    grid=fig.add_gridspec(5,1,height_ratios=[1,.06,1,.36,1.35])
    axes=[fig.add_subplot(grid[i]) for i in [0,2,4]]
    axes[0].sharex(axes[1])
    for i,(scope,key,title) in enumerate([('metrics','chamfer_distance_xy','Location Chamfer distance (m) ↓'),
                                         ('foreground','hit_rate','Object hit rate ↑')]):
        ax=axes[i];mean,sd=series(scope,key)
        ax.plot(counts,mean,color=BLUE,lw=1.5);ax.fill_between(counts,mean-sd,mean+sd,color=BLUE,alpha=.14,lw=0)
        ax.plot([value(s,'metrics','generated_count') for s in radar],[value(s,scope,key) for s in radar],color=RED,lw=1.5)
        ax.axvline(gt,color=GRAY,ls=(0,(2,2)),lw=.8)
        for s,color,marker,size in [(native,BLUE,'o',25),(rn,RED,'*',45)]:
            ax.scatter(value(s,'metrics','generated_count'),value(s,scope,key),color=color,marker=marker,s=size,edgecolor='white',lw=.4,zorder=5)
        style(ax,f'({chr(97+i)}) {title}')
        ax.set_xlim(0,2050);ax.set_xticks([0,500,1000,1500,2000]);ax.margins(y=.12)
    axes[0].tick_params(labelbottom=False)
    axes[1].set_xlabel('Mean generated points per frame',fontsize=7.5,labelpad=3)
    draw_pr(axes[2],'(c) DA precision–recall with iso-F1')
    handles=[Line2D([],[],color=BLUE,lw=1.5,label='CP3-dar-Dv3 (ours) Count Sweep'),
             Line2D([],[],color=RED,lw=1.5,label='RadarGen Count Sweep (a, b)'),
             Line2D([],[],color=GRAY,ls=(0,(2,2)),lw=.8,label=f'GT mean ({gt:,.0f}; a, b)')]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.005),frameon=False,fontsize=6.5,ncol=1,labelspacing=.3)
    fig.subplots_adjust(left=.14,right=.98,top=.97,bottom=.15,hspace=.06)
    save(fig,'three_metrics_pr')
    (OUT/'pr_contour_notes.json').write_text(json.dumps(dict(checkpoint=native['checkpoint'],
        model='CP3-dar-Dv3 (ours)',contours='F1 = 2 * precision * recall / (precision + recall)',
        note='Contours are harmonic means of plotted aggregate P/R, not the evaluator mean of per-frame F1. '
             'Crosses cover all 20 count targets. Numeric annotations are realized mean counts rounded to integers. '
             'PR panel includes DV3 only. Count sweep is not a confidence-threshold PR curve and does not define AP.',
        annotations=annotations),indent=2))
    print(OUT/'three_metrics_pr.png')


if __name__=='__main__':main()
