"""Export the approved four-panel grid for DV3, SANA, compact DV3, and RN50."""
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'output/paper_material/count_sweep_evaluation'
OUT=ROOT/'output/paper_material/count_sweep_comparisons/single_column'
MODELS=[('sana','ss50_dit_no_bottleneck_count_sweep','CP3-dar-SANA (ours)'),
        ('dino_compact','ss50_dino_compact_count_sweep','CP3-dar-Dv3 (compact)'),
        ('rn50','ss50_resnet50_count_sweep','CP3-dar-RN50 (ours)')]


def main():
    for slug,folder,label in MODELS:
        destination=OUT/slug
        subprocess.run([sys.executable,str(ROOT/'scripts/position_count_sweep.py'),
            '--model-dir',str(BASE/folder),'--output',str(destination/'position_only'),
            '--model-key',slug],check=True,cwd=ROOT)
        subprocess.run([sys.executable,str(ROOT/'scripts/plot_position_count_grid.py'),
            '--model-dir',str(BASE/folder),'--native-summary','native/summary.json',
            '--output',str(destination),'--model-key',slug,'--model-label',label,
            '--text-scale','1.10','--legend-text-scale','1.15',
            '--legend-y=-0.045','--legend-columnspacing','1.1'],check=True,cwd=ROOT)
    lines=['# Four-model position-only count-sweep figures','',
        'Each 2×2 grid contains location Chamfer, object hit rate, position-only F1, and precision–recall with a right-hand iso-F1 axis. Each model is compared separately with RadarGen on the same SS50 frames. Position matching is one-to-one within 1 m, with RCS/Doppler gates disabled. Crosses show count settings, filled markers show original points, and selected count labels show realized means. Iso-F1 curves use the harmonic mean of aggregate P/R; the F1/count panel averages per-frame F1.','']
    for slug,label in [('dino_final','CP3-dar-Dv3 (ours)')]+[(slug,label) for slug,_,label in MODELS]:
        lines.append(f'- **{label}:** [PNG]({slug}/four_panel_position.png) · [PDF]({slug}/four_panel_position.pdf) · [SVG]({slug}/four_panel_position.svg)')
    (OUT/'four_panel_position_index.md').write_text('\n'.join(lines)+'\n')
    print(OUT/'four_panel_position_index.md',flush=True)


if __name__=='__main__':main()
