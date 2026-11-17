"""Export the approved six-panel layout for all other paper models."""
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'output/paper_material/count_sweep_evaluation'
OUT=ROOT/'output/paper_material/count_sweep_comparisons/other_models_six_panel'
MODELS=[
 ('sana','ss50_dit_no_bottleneck_count_sweep','CP3-dar-SANA (ours)'),
 ('dino_compact','ss50_dino_compact_count_sweep','CP3-dar-Dv3 (compact)'),
 ('dino_random','ss50_dino_random_count_sweep','CP3-dar-Dv3 (random)'),
 ('rn50','ss50_resnet50_count_sweep','CP3-dar-RN50 (ours)'),
]


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    entries=['# Other models: six-panel count sweeps versus RadarGen','',
        'Same compact two-column layout as the DINO figure, enlarged bottom text, and individual model-versus-RadarGen comparisons. All use the same SS50 identities. Labels retain the approved CP3 prefix; exact training checkpoints are recorded in each figure directory.','']
    for slug,folder,label in MODELS:
        directory=BASE/folder
        if len(list(directory.glob('budget_*/summary.json')))!=60:
            print(f'Waiting for complete sweep: {label}',flush=True)
            continue
        subprocess.run([sys.executable,str(ROOT/'scripts/plot_final_dino_six_panel.py'),
            '--model-dir',str(directory),'--native-summary','native/summary.json',
            '--model-label',label,'--output',str(OUT/slug)],check=True,cwd=ROOT)
        entries.append(f'- **{label}:** [PNG]({slug}/count_sweep_six_panel.png) · [PDF]({slug}/count_sweep_six_panel.pdf) · [SVG]({slug}/count_sweep_six_panel.svg)')
    (OUT/'README.md').write_text('\n'.join(entries)+'\n')


if __name__=='__main__':main()
