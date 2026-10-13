"""Export verified pilot curves as CSV and a standalone PGFPlots figure.

Requires only Python's standard library to export. Compile the .tex using
pdflatex on a machine with PGFPlots (the cluster does not need TeX installed).
Each line is one independent run, not the pooled best of multiple seeds.
"""
import argparse
import csv
import json
from pathlib import Path

from evaluate import verified_score
from launch import verify
from report import summarize

METHODS = ('greedy', 'cp_sat', 'rl_scratch', 'patternboost', 'cp_bc', 'patternboost_rl', 'cp_rl')
LABELS = ('Greedy', 'CP-SAT', 'RL', 'PatternBoost', 'CP cloning', 'PatternBoost to RL', 'CP data to RL')
COLORS = ('gray', 'black', 'orange', 'blue', 'cyan!60!black', 'red', 'green!50!black')


def export(run):
    run = Path(run).resolve()
    manifest = verify(run)
    summarize(run)  # Abort if witnesses or provenance do not pass the report verifier.
    curves = []
    for task in manifest['tasks']:
        folder = run / 'tasks' / f"{task['task']:02d}_{task['condition']}_seed{task['seed']}"
        for method in ('model', *task['controls']):
            path = folder / f'{method}_evaluation.json'
            if not path.exists():
                continue
            label = task['condition'] if method == 'model' else method
            for row in json.loads(path.read_text())['results']:
                points = []
                for point in row['curve']:
                    if 'positions' not in point:
                        raise ValueError('Plot requires a saved witness for every improvement')
                    if verified_score(row['sequence'], point['positions']) != point['contacts']:
                        raise ValueError('Curve witness mismatch')
                    points.append((point['seconds'], point['contacts']))
                if points:
                    covered = row.get('proof_within_budget') or row['elapsed_s'] >= row['budget_s']
                    if covered and points[-1][0] < row['budget_s']:
                        points.append((row['budget_s'], points[-1][1]))
                    curves.append((row['id'], label, row['seed'], points))
    out = run / 'plots'
    out.mkdir(exist_ok=True)
    with (out / 'curves.csv').open('w', newline='') as handle:
        writer = csv.writer(handle, lineterminator='\n')
        writer.writerow(['sequence_id', 'method', 'seed', 'cumulative_run_seconds', 'best_contacts'])
        for seq, method, seed, points in curves:
            writer.writerows((seq, method, seed, seconds, score) for seconds, score in points)
    data = json.loads((run / 'data.json').read_text())
    rows = {row['id']: row for row in data['validation']}
    lines = [r'\documentclass[tikz,border=8pt]{standalone}', r'\usepackage{pgfplots}',
             r'\usepgfplotslibrary{groupplots}', r'\pgfplotsset{compat=1.18}',
             r'\begin{document}', r'\begin{tikzpicture}',
             r'\begin{groupplot}[group style={group size=2 by 2,horizontal sep=1.7cm,vertical sep=1.8cm},',
             r'width=8cm,height=6cm,xmin=0,ymin=0,grid=major,',
             r'xlabel={Cumulative inference time per run (s)},ylabel={Best valid HH contacts},',
             r'legend style={font=\scriptsize,at={(0.02,0.98)},anchor=north west},legend columns=1]']
    for index, seq in enumerate(manifest['validation_ids']):
        row = rows[seq]
        lines.append(r'\nextgroupplot[title={Validation length ' + str(row['length']) +
                     ', H fraction ' + f"{row['sequence'].count('H')/row['length']:.2f}" +
                     '},xmax=' + str(manifest['evaluation_seconds_per_sequence_per_seed']) + ']')
        legend_seen = set()
        for name, label, color in zip(METHODS, LABELS, COLORS):
            for selected_seq, method, seed, points in curves:
                if selected_seq != seq or method != name:
                    continue
                legend = index == 0 and method not in legend_seen
                style = 'solid' if seed % 2 == 0 else 'densely dashed'
                options = f'const plot,thick,{color},{style}' + ('' if legend else ',forget plot')
                coords = ' '.join(f'({seconds:.6f},{score})' for seconds, score in points)
                lines.append(r'\addplot+[' + options + ',mark=none] coordinates {' + coords + '};')
                if legend:
                    lines.append(r'\addlegendentry{' + label + '}')
                    legend_seen.add(method)
    lines += [r'\end{groupplot}', r'\end{tikzpicture}', r'\end{document}']
    (out / 'inference_curves.tex').write_text('\n'.join(lines) + '\n')
    (out / 'README.md').write_text(
        'Validation pilot curves; engineering evidence only. Each line is one independent run. '
        'Solid/dashed lines are seed 0/1. No pooling of their compute or best scores. '
        'Curves begin at the first witnessed valid fold; absence before it is not a zero score. '
        'Plateaus extend only over established elapsed coverage or a within-budget optimality proof.\n\n'
        'Compile with `pdflatex inference_curves.tex` in this directory.\n')
    print(json.dumps({'csv': str(out / 'curves.csv'), 'figure_source': str(out / 'inference_curves.tex'),
                      'independent_run_curves': len(curves)}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory')
    export(parser.parse_args().directory)
