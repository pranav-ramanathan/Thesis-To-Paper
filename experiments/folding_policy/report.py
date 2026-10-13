"""Verify saved validation witnesses and summarize an in-progress or completed pilot."""
import argparse
import csv
import json
from pathlib import Path
import statistics
from collections import defaultdict

from evaluate import verified_score
from launch import sha, verify, write_json


def summarize(directory):
    run = Path(directory).resolve()
    manifest = verify(run)
    data = json.loads((run / 'data.json').read_text())
    validation = {row['id']: row for row in data['validation']}
    groups, statuses, training = defaultdict(list), [], defaultdict(list)
    diagnostics = []
    errors, witnesses = [], 0
    for task in manifest['tasks']:
        folder = run / 'tasks' / f"{task['task']:02d}_{task['condition']}_seed{task['seed']}"
        launch_file = folder / 'launch.json'
        launch = json.loads(launch_file.read_text()) if launch_file.exists() else {}
        statuses.append({'task': task['task'], 'condition': task['condition'], 'seed': task['seed'],
                         'status': launch.get('status', 'not_started'), 'elapsed_s': launch.get('elapsed_s'),
                         'error': launch.get('error')})
        for phase in launch.get('phases', []):
            if phase['name'] == 'training':
                training[task['condition']].append({'seed': task['seed'], **phase})
        training_result = folder / 'training' / 'result.json'
        if training_result.exists():
            trained = json.loads(training_result.read_text())
            counters = trained.get('counters', {})
            handoff = trained.get('handoff')
            handoff_valid = (bool(handoff) and handoff.get('pretraining_sha256') == handoff.get('rl_initial_sha256')
                             and handoff.get('same_model_object') is True
                             and handoff.get('optimizer_state_retained') is True)
            if trained.get('status') == 'complete':
                if (trained.get('dataset_sha256') != manifest['hashes']['data.json']
                        or trained.get('test_evaluated') is not False
                        or trained.get('selection', {}).get('validated') is not True):
                    errors.append(f'{training_result}: complete training lacks matching data/validation provenance')
                if task['condition'] in ('patternboost_rl', 'cp_rl') and not handoff_valid:
                    errors.append(f'{training_result}: same-weights stage handoff was not verified')
                if task['condition'] != 'rl_scratch' and counters.get('supervised_updates', 0) == 0:
                    errors.append(f'{training_result}: no supervised updates in an allegedly complete run')
                if task['condition'] in ('rl_scratch', 'patternboost_rl', 'cp_rl') and counters.get('rl_updates', 0) == 0:
                    errors.append(f'{training_result}: no RL updates in an allegedly complete run')
            diagnostics.append({'task': task['task'], 'condition': task['condition'], 'seed': task['seed'],
                                'status': trained.get('status'), 'counters': counters, 'timing': trained.get('timing', {}),
                                'selection': trained.get('selection'), 'handoff_valid': handoff_valid if handoff else None,
                                'elite_count': trained.get('elite_count'),
                                'elite_sequence_count': trained.get('elite_sequence_count'),
                                'training_example_coverage': trained.get('training_example_coverage', []),
                                'training_sequences': trained.get('training_sequences'),
                                'zero_advantage_group_fraction': trained.get('zero_advantage_group_fraction')})
        for method in ['model', *task['controls']]:
            path = folder / (method + '_evaluation.json')
            if not path.exists():
                continue
            payload = json.loads(path.read_text())
            if payload.get('split') != 'validation' or payload.get('final_test_access') is not False:
                errors.append(f'{path}: forbidden evaluation split')
                continue
            if payload.get('dataset_file_sha256') != manifest['hashes']['data.json']:
                errors.append(f'{path}: evaluation dataset differs from frozen data')
                continue
            if method == 'model':
                checkpoint = folder / 'training' / 'checkpoint.pt'
                if not checkpoint.exists() or payload.get('checkpoint_sha256') != sha(checkpoint):
                    errors.append(f'{path}: selected checkpoint changed or is missing')
                    continue
            for row in payload['results']:
                try:
                    if row['id'] not in manifest['validation_ids'] or row['sequence'] != validation[row['id']]['sequence']:
                        raise ValueError('Sequence differs from frozen validation selection')
                    if row['seed'] != task['seed'] or row['method'] != method:
                        raise ValueError('Method or seed provenance differs from its task')
                    if method == 'model' and row.get('checkpoint_condition') != task['condition']:
                        raise ValueError('Checkpoint training condition differs from its task')
                    if method == 'model' and row.get('checkpoint_seed') != task['seed']:
                        raise ValueError('Checkpoint training seed differs from its task')
                    if row['budget_s'] != manifest['evaluation_seconds_per_sequence_per_seed']:
                        raise ValueError('Inference budget differs from frozen pilot')
                    if row['elapsed_s'] < row['budget_s'] and not row.get('proof_within_budget'):
                        raise ValueError('Inference stopped before its budget without an optimality proof')
                    if row['best_contacts'] is not None:
                        score = verified_score(row['sequence'], row['best_positions'])
                        if score != row['best_contacts']:
                            raise ValueError('Best contact count does not match witness')
                        witnesses += 1
                    last_time, last_score = -1, -1
                    for point in row['curve']:
                        if not last_time <= point['seconds'] <= row['budget_s'] or point['contacts'] <= last_score:
                            raise ValueError('Invalid best-so-far curve')
                        if verified_score(row['sequence'], point['positions']) != point['contacts']:
                            raise ValueError('Curve contact count does not match its witness')
                        witnesses += 1
                        last_time, last_score = point['seconds'], point['contacts']
                    if row['curve'] and row['curve'][-1]['contacts'] != row['best_contacts']:
                        raise ValueError('Final curve disagrees with verified best')
                    label = task['condition'] if method == 'model' else method
                    if any(prior['seed'] == row['seed'] for prior in groups[(row['id'], label)]):
                        raise ValueError('Duplicate seed would be counted twice')
                    groups[(row['id'], label)].append(row)
                except (ValueError, KeyError, TypeError) as exc:
                    errors.append(f'{path}: {exc}')
    summary = []
    for (seq_id, method), rows in sorted(groups.items()):
        values = [row['best_contacts'] for row in rows if row['best_contacts'] is not None]
        attempts = sum(row['attempted_folds'] or 0 for row in rows)
        generated = sum(row['generated_complete_folds'] or 0 for row in rows)
        summary.append({'id': seq_id, 'length': len(validation[seq_id]['sequence']), 'method': method,
                        'runs_recorded': len(rows), 'runs_with_valid_fold': len(values),
                        'best_contacts': max(values) if values else None,
                        'mean_contacts': statistics.mean(values) if values else None,
                        'sample_sd': statistics.stdev(values) if len(values) > 1 else None,
                        'generated_complete_folds': generated if method != 'cp_sat' else None,
                        'attempted_folds': attempts if method != 'cp_sat' else None,
                        'valid_fold_rate': generated / attempts if attempts else None,
                        'total_requested_inference_s': sum(row['budget_s'] for row in rows),
                        'total_observed_inference_s': sum(row['elapsed_s'] for row in rows),
                        'proofs_within_budget': sum(row['proof_within_budget'] for row in rows),
                        'bounds': [row['bound'] for row in rows], 'seeds': [row['seed'] for row in rows]})
    result = {'schema': 1, 'kind': 'validation_engineering_pilot', 'tasks': statuses,
              'verified_witness_records': witnesses, 'errors': errors, 'summary': summary,
              'training': dict(training), 'training_diagnostics': diagnostics, 'final_test_evaluated': False}
    write_json(run / 'analysis.json', result)
    lines = ['# Variable-length policy validation pilot', '',
             'Engineering measurements on validation sequences only. The final test sets remain unevaluated.', '',
             f"Tasks: {sum(x['status'] == 'complete' for x in statuses)}/{len(statuses)} complete; "
             f"{sum(x['status'] == 'failed' for x in statuses)} failed. "
             f'Independently checked witness records: {witnesses}; verification errors: {len(errors)}. '
             'These include final-best and improvement records and are not a count of unique folds.', '',
             'Each method receives 60 seconds per sequence per seed, two seeds: 120 seconds total per sequence. '
             'Training and example generation share the 20-minute training budget per model seed. '
             'Inference includes model loading; CP-SAT performs fresh full-domain searches and never repairs model outputs.', '',
             'Only rows with a valid complete fold enter contact means; missing folds are not assigned zero. '
             'The valid-run count remains visible. Two seeds describe pilot variability and cannot establish superiority.', '',
             '| Validation sequence | Length | Method | Valid / recorded / planned | Best | Mean contacts | Sample SD | Complete folds | Valid rate | Inference seconds (sum) |',
             '|---|---:|---|---:|---:|---:|---:|---:|---:|---:|']
    def fmt(value, digits=2):
        return '—' if value is None else f'{value:.{digits}f}'
    for row in summary:
        lines.append(f"| {row['id']} | {row['length']} | {row['method']} | {row['runs_with_valid_fold']}/{row['runs_recorded']}/{manifest['evaluation_seeds']} | "
                     f"{fmt(row['best_contacts'], 0)} | {fmt(row['mean_contacts'])} | {fmt(row['sample_sd'])} | "
                     f"{fmt(row['generated_complete_folds'], 0)} | {fmt(row['valid_fold_rate'], 3)} | {fmt(row['total_observed_inference_s'])} |")
    lines += ['', 'CP-SAT generated-fold totals are unavailable: its reported witnesses are incumbents, '
              'not all search candidates. Bounds, proof status and complete improvement curves are saved in task evaluation JSON files.', '',
              '## Training cost', '', '| Method | Recorded runs | Total training seconds |', '|---|---:|---:|']
    for method, phases in sorted(training.items()):
        lines.append(f"| {method} | {len(phases)} | {sum(x['seconds'] for x in phases):.2f} |")
    lines += ['', '## Training diagnostics', '',
              '| Method | Seed | Status | SFT updates | RL updates | Valid / generated training folds | Sequences with elites | Zero-advantage groups | Validation selected | Same-weights handoff |',
              '|---|---:|---|---:|---:|---:|---:|---:|---|---|']
    for row in diagnostics:
        counters, selection = row['counters'], row['selection'] or {}
        lines.append(f"| {row['condition']} | {row['seed']} | {row['status']} | {counters.get('supervised_updates', 0)} | "
                     f"{counters.get('rl_updates', 0)} | {counters.get('valid_folds', 0)}/{counters.get('generated_folds', 0)} | "
                     f"{fmt(row['elite_sequence_count'], 0)}/{fmt(row['training_sequences'], 0)} | "
                     f"{fmt(row['zero_advantage_group_fraction'], 3)} | {selection.get('validated', False)} | "
                     f"{row['handoff_valid'] if row['handoff_valid'] is not None else '—'} |")
    lines += ['', 'Data generation, supervised updates, candidate generation, RL, initialization and validation '
              'times are retained per task in analysis.json. The training summary also records elite-pool coverage '
              'and generated/valid fold counts; inspect zero-advantage groups and empty pools before expanding the pilot.']
    lines += ['', 'For an end-to-end comparison, add each model seed’s measured training time to its inference time. '
              'For N future sequences, report amortized time as training time / N + inference time per sequence; '
              'state N explicitly. Keep the two seeds as independent trained policies.', '',
              '## Task status', '', '| Task | Condition | Seed | Status | Error |', '|---:|---|---:|---|---|']
    for row in statuses:
        lines.append(f"| {row['task']} | {row['condition']} | {row['seed']} | {row['status']} | {row['error'] or '—'} |")
    if errors:
        lines += ['', '## Verification errors', '', *['- ' + error for error in errors]]
    (run / 'report.md').write_text('\n'.join(lines) + '\n')
    with (run / 'summary.csv').open('w', newline='') as handle:
        fields = list(summary[0]) if summary else ['id', 'method', 'runs_recorded']
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        writer.writerows(summary)
    with (run / 'curves.csv').open('w', newline='') as handle:
        fields = ['id', 'length', 'method', 'seed', 'seconds', 'contacts', 'budget_s']
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        for (seq_id, method), rows in sorted(groups.items()):
            for row in sorted(rows, key=lambda item: item['seed']):
                for point in row['curve']:
                    writer.writerow({'id': seq_id, 'length': len(validation[seq_id]['sequence']),
                                     'method': method, 'seed': row['seed'], 'seconds': point['seconds'],
                                     'contacts': point['contacts'], 'budget_s': row['budget_s']})
    print(json.dumps({'report': str(run / 'report.md'), 'complete_tasks': sum(x['status'] == 'complete' for x in statuses),
                      'verified_witness_records': witnesses, 'errors': errors}))
    if errors:
        raise SystemExit(1)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory')
    summarize(parser.parse_args().directory)
