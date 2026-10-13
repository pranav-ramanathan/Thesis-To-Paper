"""Verify frozen-policy inference and explicit amortized-cost comparisons."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics


ARMS = ('patternboost', 'balanced', 'more_pretrain', 'rl_scratch', 'cp_sat', 'greedy')


def finite_nonnegative(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def verify_row(row, source, task):
    """Independently check every fold and establish time/proof coverage."""
    from evaluate import verified_score
    seconds = task['seconds']
    if (row.get('id') != source['id'] or row.get('sequence') != source['sequence']
            or row.get('length') != len(source['sequence'])):
        raise ValueError('Sequence differs from frozen validation data')
    if row.get('method') != task['method'] or row.get('seed') != task['seed']:
        raise ValueError('Sequence method or seed differs from frozen task')
    if row.get('budget_s') != seconds:
        raise ValueError('Inference budget differs from frozen task')
    elapsed = row.get('elapsed_s')
    if not finite_nonnegative(elapsed):
        raise ValueError('Invalid elapsed time')
    if not finite_nonnegative(row.get('setup_s')) or row['setup_s'] > elapsed:
        raise ValueError('Invalid setup time')
    cp = task['method'] == 'cp_sat'
    generated, attempted = row.get('generated_complete_folds'), row.get('attempted_folds')
    if cp:
        if generated is not None or attempted is not None:
            raise ValueError('CP-SAT incumbent counts cannot stand in for generated fold counts')
    elif (type(generated) is not int or type(attempted) is not int
          or not 0 <= generated <= attempted):
        raise ValueError('Invalid complete or attempted fold count')
    best, positions, curve = row.get('best_contacts'), row.get('best_positions'), row.get('curve')
    if not isinstance(curve, list):
        raise ValueError('Missing improvement curve')
    witnesses = 0
    if best is None:
        if positions is not None or curve or (not cp and generated != 0):
            raise ValueError('Missing best fold has claimed witnesses or complete folds')
    else:
        if type(best) is not int or best < 0 or not curve or (not cp and generated < 1):
            raise ValueError('Scored fold requires an integer score and recorded improvement')
        if verified_score(source['sequence'], positions) != best:
            raise ValueError('Best contact count differs from its witness')
        witnesses += 1
    previous_time, previous_score = 0, -1
    for point in curve:
        when, score = point.get('seconds'), point.get('contacts')
        if (not finite_nonnegative(when) or not previous_time <= when <= min(seconds, elapsed)
                or type(score) is not int or score <= previous_score):
            raise ValueError('Invalid improvement curve time or score')
        if verified_score(source['sequence'], point.get('positions')) != score:
            raise ValueError('Curve contact count differs from its witness')
        witnesses += 1
        previous_time, previous_score = when, score
    if curve and curve[-1]['contacts'] != best:
        raise ValueError('Final improvement differs from best contact count')
    proof_time = None
    if cp:
        bound, recorded = row.get('bound'), row.get('bound_recorded_s')
        if bound is not None:
            if (not finite_nonnegative(bound) or row.get('bound_scope') != 'full_declared_cube'
                    or not finite_nonnegative(recorded) or recorded > elapsed):
                raise ValueError('Solver bound lacks full-domain scope or valid timing')
            if best is not None and bound < best:
                raise ValueError('Solver upper bound is below a verified contact count')
        if row.get('proof_within_budget') is True:
            if (row.get('solver_status') != 'OPTIMAL' or best is None or bound != best
                    or not finite_nonnegative(recorded) or recorded > min(seconds, elapsed)
                    or row.get('bound_scope') != 'full_declared_cube'
                    or curve[-1]['seconds'] > recorded):
                raise ValueError('Invalid full-domain optimality proof within budget')
            proof_time = recorded
    elif row.get('proof_within_budget'):
        raise ValueError('A sampled model or greedy fold is not an optimality proof')
    return {'witnesses': witnesses, 'coverage_s': min(elapsed, seconds),
            'proof_s': proof_time, 'complete_budget': elapsed >= seconds or proof_time is not None}


def at_time(row, checked, seconds):
    """Later optimality proofs never fill earlier unobserved timepoints."""
    proof = checked['proof_s']
    covered = checked['coverage_s'] >= seconds or (proof is not None and proof <= seconds)
    best = next((point['contacts'] for point in reversed(row['curve']) if point['seconds'] <= seconds), None)
    bound_time = row.get('bound_recorded_s')
    bound = (row.get('bound') if row.get('bound_scope') == 'full_declared_cube'
             and finite_nonnegative(bound_time) and bound_time <= seconds else None)
    return {'covered': covered, 'contacts': best if covered else None,
            'upper_bound': bound if covered else None,
            'proven_optimal': covered and proof is not None and proof <= seconds,
            'gap_contacts': bound-best if covered and best is not None and bound is not None else None}


def validate_payload(payload, manifest, task):
    if (payload.get('schema') != 1 or payload.get('kind') != 'inference_comparison_evaluation'
            or payload.get('split') != 'validation' or payload.get('final_test_access') is not False):
        raise ValueError('Evaluation is not the declared validation-only inference comparison')
    for key in ('arm', 'method', 'seed'):
        if payload.get(key) != task[key]:
            raise ValueError(f'Evaluation {key} differs from frozen task')
    if (payload.get('threads') != manifest['threads']
            or payload.get('seconds_per_sequence') != task['seconds']
            or payload.get('ids') != manifest['validation_ids']):
        raise ValueError('Evaluation resource budget or validation IDs differ from frozen task')
    if payload.get('dataset_file_sha256') != manifest['hashes']['data.json']:
        raise ValueError('Evaluation dataset differs from frozen validation data')
    if payload.get('model_setup_charged_per_sequence') is not True:
        raise ValueError('Inference must charge setup within each sequence budget')
    if payload.get('status') not in ('running', 'complete', 'failed'):
        raise ValueError('Unrecognized evaluation status')
    if task['method'] != 'model':
        if payload.get('checkpoint_sha256') is not None or payload.get('checkpoint_metadata') is not None:
            raise ValueError('Plain baseline unexpectedly uses a model checkpoint')
        return
    if payload.get('checkpoint_sha256') != task['checkpoint_sha256']:
        raise ValueError('Evaluation checkpoint differs from frozen endpoint')
    metadata = payload.get('checkpoint_metadata')
    if metadata is None and payload.get('status') != 'complete' and not payload.get('results'):
        return
    if not isinstance(metadata, dict):
        raise ValueError('Model checkpoint metadata is missing')
    mode = task['arm'] if task['arm'] in ('patternboost', 'rl_scratch') else 'rl_branch'
    expected = {'schema': 1, 'kind': 'allocation_training_state', 'seed': task['seed'],
                'mode': mode, 'stage': 'patternboost' if mode == 'patternboost' else 'rl',
                'condition': 'patternboost_rl' if mode == 'rl_branch' else mode,
                'training_seconds': task['training_seconds'],
                'dataset_sha256': manifest['hashes']['data.json'],
                'state_sha256': task['state_sha256']}
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise ValueError('Checkpoint does not match the fixed training endpoint and lineage')
    for row in payload.get('results', []):
        for field, value in (('checkpoint_stage', metadata['stage']),
                             ('checkpoint_condition', metadata['condition']),
                             ('checkpoint_seed', task['seed']),
                             ('checkpoint_dataset_sha256', metadata['dataset_sha256'])):
            if row.get(field) != value:
                raise ValueError('Sequence checkpoint provenance differs from frozen endpoint')


def _csv(path, rows, fields):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in fields} for row in rows)


def _number(value, digits=2):
    return '—' if value is None else f'{value:.{digits}f}'


def summarize(directory):
    from inference_launch import task_directory, verify
    run = Path(directory).resolve()
    manifest = verify(run)
    data = json.loads((run / 'data.json').read_text())
    sources = {row['id']: row for row in data['validation']}
    ids, tasks = manifest['validation_ids'], manifest['tasks']
    seeds = sorted({task['seed'] for task in tasks})
    errors, statuses, valid_rows = [], [], {}
    witnesses = 0
    for task in tasks:
        folder = task_directory(run, task)
        path = folder / 'evaluation.json'
        state = {'task': task['task'], 'arm': task['arm'], 'seed': task['seed'],
                 'status': 'not_started', 'verified_sequences': 0, 'complete_sequences': 0,
                 'planned_sequences': len(ids), 'error': None}
        statuses.append(state)
        launch_path = folder / 'launch.json'
        try:
            launched = json.loads(launch_path.read_text()) if launch_path.exists() else {}
            state['launch_status'] = launched.get('status', 'not_started')
            state['error'] = launched.get('error')
            if not path.exists():
                state['status'] = state['launch_status']
                continue
            payload = json.loads(path.read_text())
            validate_payload(payload, manifest, task)
            if task['method'] == 'model':
                checkpoint = run / task['checkpoint']
                if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != task['checkpoint_sha256']:
                    raise ValueError('Frozen checkpoint file changed')
            seen, rows, row_witnesses = set(), [], 0
            for row in payload.get('results', []):
                seq_id = row.get('id')
                if seq_id not in ids or seq_id in seen:
                    raise ValueError('Duplicate or unexpected validation sequence')
                seen.add(seq_id)
                checked = verify_row(row, sources[seq_id], task)
                row_witnesses += checked['witnesses']
                rows.append((seq_id, row, checked))
            if payload['status'] == 'complete' and (seen != set(ids) or not all(c['complete_budget'] for _, _, c in rows)):
                raise ValueError('Complete evaluation lacks every sequence and full budget or timely proof')
            state['status'] = payload['status']
            state['error'] = payload.get('error') or state['error']
            state['verified_sequences'] = len(rows)
            state['complete_sequences'] = sum(c['complete_budget'] for _, _, c in rows)
            for seq_id, row, checked in rows:
                valid_rows[(task['arm'], task['seed'], seq_id)] = {
                    'row': row, 'checked': checked, 'task_complete': payload['status'] == 'complete'}
            witnesses += row_witnesses
        except (ValueError, TypeError, KeyError, OSError) as exc:
            state['status'], state['error'] = 'verification_failed', str(exc)
            errors.append(f'{path}: {exc}')

    points, curves = [], []
    inference_times = manifest['inference_timepoints_s']
    n = manifest['amortization_sequences']
    model_seconds = max(inference_times)
    lineage = manifest['training_lineage_seconds']
    charged_seconds = model_seconds + lineage / n
    for task in tasks:
        for seq_id in ids:
            stored = valid_rows.get((task['arm'], task['seed'], seq_id))
            evaluations = [('equal_inference', t, t) for t in inference_times]
            evaluations.append(('amortized_end_to_end', charged_seconds,
                                model_seconds if task['method'] == 'model' else charged_seconds))
            for comparison, charge, search in evaluations:
                result = at_time(stored['row'], stored['checked'], search) if stored else {
                    'covered': False, 'contacts': None, 'upper_bound': None,
                    'proven_optimal': False, 'gap_contacts': None}
                eligible = bool(stored and stored['task_complete'] and result['covered'])
                points.append({'id': seq_id, 'length': len(sources[seq_id]['sequence']),
                               'arm': task['arm'], 'seed': task['seed'], 'comparison': comparison,
                               'charged_seconds': charge, 'inference_seconds': search,
                               'training_charge_seconds': task['training_seconds'] / n if comparison == 'amortized_end_to_end' else 0,
                               'recorded': stored is not None, 'complete': eligible,
                               'contacts': result['contacts'], 'covered': result['covered'],
                               'upper_bound': result['upper_bound'], 'gap_contacts': result['gap_contacts'],
                               'proven_optimal': result['proven_optimal'],
                               'actual_full_search_elapsed_s': stored['row']['elapsed_s'] if stored else None,
                               'generated_complete_folds': stored['row'].get('generated_complete_folds') if stored and search == task['seconds'] else None,
                               'attempted_folds': stored['row'].get('attempted_folds') if stored and search == task['seconds'] else None})
            if stored:
                for point in stored['row']['curve']:
                    curves.append({'id': seq_id, 'arm': task['arm'], 'seed': task['seed'],
                                   'seconds': point['seconds'], 'contacts': point['contacts'],
                                   'task_complete': stored['task_complete']})

    summary = []
    for comparison, charge in [('equal_inference', t) for t in inference_times] + [('amortized_end_to_end', charged_seconds)]:
        for seq_id in ids:
            for arm in ARMS:
                group = [p for p in points if p['comparison'] == comparison and p['charged_seconds'] == charge
                         and p['id'] == seq_id and p['arm'] == arm]
                if not group:
                    continue
                complete = [p for p in group if p['complete']]
                valid = [p for p in complete if p['contacts'] is not None]
                values = [p['contacts'] for p in valid]
                summary.append({'comparison': comparison, 'charged_seconds': charge,
                                'inference_seconds': group[0]['inference_seconds'], 'id': seq_id,
                                'length': len(sources[seq_id]['sequence']), 'arm': arm,
                                'valid_runs': len(valid), 'complete_runs': len(complete),
                                'recorded_runs': sum(p['recorded'] for p in group), 'planned_runs': len(group),
                                'best_contacts': max(values) if values else None,
                                'mean_contacts': statistics.mean(values) if values else None,
                                'sd_contacts': statistics.stdev(values) if len(values) > 1 else None,
                                'proofs': sum(p['proven_optimal'] for p in complete)})

    planned_seconds = sum(task['seconds'] * len(ids) for task in tasks)
    actual_seconds = sum(value['row']['elapsed_s'] for value in valid_rows.values())
    costs = {'new_training_seconds': 0, 'planned_evaluation_wall_hours_sum': planned_seconds / 3600,
             'planned_evaluation_allocated_cpu_hours': planned_seconds * manifest['threads'] / 3600,
             'recorded_evaluation_wall_hours_sum': actual_seconds / 3600,
             'recorded_evaluation_allocated_cpu_hours': actual_seconds * manifest['threads'] / 3600,
             'reused_model_lineage_cpu_hours': sum(t['training_seconds'] for t in tasks) * manifest['threads'] / 3600,
             'amortization_sequences': n, 'model_training_charge_seconds_per_sequence': lineage / n,
             'end_to_end_seconds_per_sequence': charged_seconds}
    output = {'schema': 1, 'kind': 'validation_inference_comparison_report', 'final_test_access': False,
              'complete_tasks': sum(s['status'] == 'complete' for s in statuses), 'planned_tasks': len(tasks),
              'verified_witness_records': witnesses, 'errors': errors, 'tasks': statuses,
              'summary': summary, 'timepoints': points, 'costs': costs,
              'runs': [{'arm': arm, 'seed': seed, 'id': seq_id, **stored}
                       for (arm, seed, seq_id), stored in valid_rows.items()]}
    (run / 'analysis.json').write_text(json.dumps(output, indent=2) + '\n')
    _csv(run / 'summary.csv', summary, list(summary[0]) if summary else ['comparison', 'id', 'arm'])
    _csv(run / 'curves.csv', curves, ['id', 'arm', 'seed', 'seconds', 'contacts', 'task_complete'])
    lines = ['# Frozen-policy inference comparison', '',
             'Validation-only screening using the saved two-hour training endpoints. No retraining, checkpoint selection, final-test evaluation or model-output repair occurs.', '',
             f"Tasks: {output['complete_tasks']}/{len(tasks)} complete. Independently verified witness records: {witnesses}; verification errors: {len(errors)}.", '',
             'Contacts are each seed’s best independently verified complete fold within its inference budget. Means exclude incomplete tasks and missing folds; zero contacts remain zero. Valid, complete, recorded and planned counts remain visible. Three trained policy seeds do not become independent replications when evaluated on multiple sequences.', '',
             '## Equal inference budgets', '',
             'Setup and model loading count toward each sequence’s search budget. Scores at 1, 5 and 10 minutes exclude later improvements. A timely full-domain optimality proof permits carrying its score forward, never backward.', '']
    for seconds in inference_times:
        lines += [f'### {seconds / 60:g} minutes per sequence and seed', '',
                  '| Sequence | Length | Method | Valid / complete / recorded / planned | Best | Mean contacts | SD | CP proofs |',
                  '|---|---:|---|---:|---:|---:|---:|---:|']
        for row in summary:
            if row['comparison'] == 'equal_inference' and row['charged_seconds'] == seconds:
                lines.append(f"| {row['id']} | {row['length']} | {row['arm']} | {row['valid_runs']}/{row['complete_runs']}/{row['recorded_runs']}/{row['planned_runs']} | {_number(row['best_contacts'], 0)} | {_number(row['mean_contacts'])} | {_number(row['sd_contacts'])} | {row['proofs'] if row['arm'] == 'cp_sat' else '—'} |")
        lines.append('')
    lines += ['## Including amortized training cost', '',
              f"Deployment assumption: one policy is reused across **{n} sequences**. Charge each model seed its full {lineage / 3600:g}-hour training lineage, including example generation: {lineage / n:g} seconds per sequence, plus {model_seconds:g} seconds inference. Compare with **{charged_seconds:g} seconds ({charged_seconds / 60:g} minutes) of fresh CP-SAT or greedy search** on the same allocated hardware. Shared historical prefixes do not erase scientific training costs. This amortization assumption is explicit, not a universal break-even claim.", '',
              '| Sequence | Length | Method | Inference seconds | Valid / complete / recorded / planned | Mean contacts | SD |',
              '|---|---:|---|---:|---:|---:|---:|']
    for row in summary:
        if row['comparison'] == 'amortized_end_to_end':
            lines.append(f"| {row['id']} | {row['length']} | {row['arm']} | {row['inference_seconds']:g} | {row['valid_runs']}/{row['complete_runs']}/{row['recorded_runs']}/{row['planned_runs']} | {_number(row['mean_contacts'])} | {_number(row['sd_contacts'])} |")
    lines += ['', '## Verified inference accounting', '',
              'Complete folds and attempted folds below cover each task’s full inference budget; they are not counts at earlier timepoints. CP-SAT reports incumbents rather than every candidate, so its generated-fold counts remain unavailable. Bounds recorded after a cutoff do not enter that cutoff’s gaps or proof claims.', '',
              '| Sequence | Seed | Method | Allocated seconds | Actual seconds | Complete folds | Valid rate | Status | Upper bound within budget | Gap contacts | Proof within budget |',
              '|---|---:|---|---:|---:|---:|---:|---|---:|---:|---|']
    for task in tasks:
        for seq_id in ids:
            stored = valid_rows.get((task['arm'], task['seed'], seq_id))
            if not stored:
                continue
            row, checked = stored['row'], stored['checked']
            final = at_time(row, checked, task['seconds'])
            attempted, complete = row.get('attempted_folds'), row.get('generated_complete_folds')
            rate = complete / attempted if attempted else None
            lines.append(f"| {seq_id} | {task['seed']} | {task['arm']} | {task['seconds']:g} | {_number(row['elapsed_s'])} | {_number(complete, 0)} | {_number(rate, 3)} | {row.get('solver_status') or ('complete' if checked['complete_budget'] else 'partial')} | {_number(final['upper_bound'])} | {_number(final['gap_contacts'], 0)} | {'yes' if final['proven_optimal'] else 'no'} |")
    lines += ['', '## Compute accounting', '',
              f"New training: zero. Planned new inference allocation: **{costs['planned_evaluation_allocated_cpu_hours']:.2f} CPU-hours**, before job startup and reporting. Recorded verified inference: {costs['recorded_evaluation_wall_hours_sum']:.3f} summed wall-hours ({costs['recorded_evaluation_allocated_cpu_hours']:.3f} allocated CPU-hours). Early CP-SAT proofs can reduce actual elapsed time.", '',
              f"Historical training charged to the evaluated model lineages: {costs['reused_model_lineage_cpu_hours']:.2f} allocated CPU-hours; no such training is repeated by this campaign. Allocated core-wall products are not measured CPU utilization. Consult Slurm accounting for actual CPU use and peak RAM.", '',
              'These are validation screening results, not final-test superiority or optimal-ratio claims. Both baselines use fresh searches; neither repairs model outputs. All runs and timepoint coverage are retained in analysis.json.', '',
              '## Task status', '', '| Task | Arm | Seed | Status | Complete / verified / planned sequences | Error |',
              '|---:|---|---:|---|---:|---|']
    for state in statuses:
        error = str(state['error'] or '—').replace('|', '/').replace('\n', ' ')
        lines.append(f"| {state['task']} | {state['arm']} | {state['seed']} | {state['status']} | {state['complete_sequences']}/{state['verified_sequences']}/{state['planned_sequences']} | {error} |")
    (run / 'report.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps({'report': str(run / 'report.md'), 'complete_tasks': output['complete_tasks'],
                      'verified_witness_records': witnesses, 'errors': errors}))
    if errors:
        raise SystemExit(1)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    summarize(parser.parse_args().directory)


if __name__ == '__main__':
    main()
