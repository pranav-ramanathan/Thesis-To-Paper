"""Verify learning-rate controls and longer inference, including ordered attempts."""
import argparse
import json
import math
from pathlib import Path
import statistics
import time

from finetune_report import number, sha, validate_training, write_csv
from inference_report import at_time, finite_nonnegative, verify_row

ARMS = ('parent', 'patternboost', 'grpo', 'dr_grpo', 'reinforce', 'dpo',
        'patternboost_low_lr', 'grpo_low_lr')
STATUS_COUNTERS = {'complete': 'generated_complete_folds', 'trapped': 'trapped_folds',
                   'interrupted': 'interrupted_folds', 'after_budget': 'discarded_after_budget'}


def verify_attempts(row, source, checkpoints):
    """Reconstruct fixed-attempt results from independently checked ordered folds.

    A prefix includes traps and interrupted attempts. It is covered only when
    every one of its requested attempts finished before the inference cutoff.
    Neither later folds nor a summary's claimed coverage can fill a prefix.
    """
    from evaluate import verified_score
    if row.get('attempt_checkpoints') != checkpoints:
        raise ValueError('Attempt checkpoints differ from the frozen comparison')
    records, total = row.get('ordered_attempts'), row['attempted_folds']
    if not isinstance(records, list) or len(records) != min(max(checkpoints), total):
        raise ValueError('Ordered attempt records are incomplete or excessive')
    previous_time, witnessed = 0, 0
    counts = {name: 0 for name in STATUS_COUNTERS}
    for index, record in enumerate(records, 1):
        when, status = record.get('seconds'), record.get('status')
        if (type(record.get('attempt_index')) is not int or record['attempt_index'] != index
                or not finite_nonnegative(when) or not previous_time <= when <= row['elapsed_s']
                or status not in STATUS_COUNTERS):
            raise ValueError('Invalid ordered attempt index, status or timestamp')
        previous_time = when
        counts[status] += 1
        if status in ('complete', 'trapped') and when > row['budget_s']:
            raise ValueError('Late attempt cannot count as a timely completed or trapped fold')
        if status == 'complete':
            score = record.get('contacts')
            if type(score) is not int or score < 0 or verified_score(source['sequence'], record.get('positions')) != score:
                raise ValueError('Ordered attempt score differs from its complete witness')
            witnessed += 1
        elif record.get('contacts') is not None:
            raise ValueError('Unfinished or discarded attempts cannot carry scored contacts')
    for status, key in STATUS_COUNTERS.items():
        value = row.get(key)
        if type(value) is not int or value < counts[status]:
            raise ValueError('Full attempt counters are below their recorded prefix')
        if total <= max(checkpoints) and value != counts[status]:
            raise ValueError('Full attempt counters differ from the complete ordered log')
    if sum(row[key] for key in STATUS_COUNTERS.values()) != total:
        raise ValueError('Attempt status counters do not partition all attempts')
    if row.get('valid_fold_rate') != (row['generated_complete_folds'] / total if total else None):
        raise ValueError('Valid-fold rate differs from attempt counters')

    previous_index = 0
    for point in row['curve']:
        index = point.get('attempt_index')
        if type(index) is not int or not previous_index < index <= total:
            raise ValueError('Improvement curve has an invalid attempt index')
        previous_index = index
        if index <= len(records):
            record = records[index - 1]
            if (record['status'] != 'complete' or record['contacts'] != point['contacts']
                    or record['seconds'] != point['seconds'] or record['positions'] != point['positions']):
                raise ValueError('Improvement curve differs from its ordered attempt')
    # The curve must also include every improvement established in the raw prefix.
    best, raw_improvements = -1, []
    for record in records:
        if record['status'] == 'complete' and record['contacts'] > best:
            best = record['contacts']
            raw_improvements.append(record['attempt_index'])
    if [p['attempt_index'] for p in row['curve'] if p['attempt_index'] <= len(records)] != raw_improvements:
        raise ValueError('Improvement curve omitted or invented an ordered-prefix improvement')

    snapshots = row.get('attempt_prefixes')
    if not isinstance(snapshots, list) or [p.get('attempts') for p in snapshots] != checkpoints:
        raise ValueError('Missing or duplicate fixed-attempt snapshots')
    checked = {}
    for requested, snapshot in zip(checkpoints, snapshots):
        prefix = records[:requested]
        reached = total >= requested
        covered = reached and all(p['status'] in ('complete', 'trapped') for p in prefix)
        expected = {'attempts': requested, 'observed_attempts': min(requested, total),
                    'reached': reached, 'covered': covered,
                    'seconds': prefix[-1]['seconds'] if prefix else None,
                    'complete_folds': sum(p['status'] == 'complete' for p in prefix),
                    'trapped_folds': sum(p['status'] == 'trapped' for p in prefix),
                    'interrupted_folds': sum(p['status'] == 'interrupted' for p in prefix),
                    'discarded_after_budget': sum(p['status'] == 'after_budget' for p in prefix)}
        if any(snapshot.get(key) != value for key, value in expected.items()):
            raise ValueError('Fixed-attempt snapshot differs from its ordered prefix')
        if type(snapshot.get('covered')) is not bool or type(snapshot.get('reached')) is not bool:
            raise ValueError('Fixed-attempt coverage must be boolean')
        complete = [p for p in prefix if p['status'] == 'complete']
        score = max((p['contacts'] for p in complete), default=None) if covered else None
        if snapshot.get('best_contacts') != score:
            raise ValueError('Fixed-attempt best score used missing, late or later-attempt folds')
        positions = snapshot.get('best_positions')
        if score is None:
            if positions is not None:
                raise ValueError('Missing fixed-attempt score has an unexpected witness')
        elif not any(p['contacts'] == score and p['positions'] == positions for p in complete):
            raise ValueError('Fixed-attempt best witness is absent from its ordered prefix')
        checked[requested] = {**expected, 'contacts': score}
    return checked, witnessed


def verify_resources(row):
    resources = row.get('resource_usage')
    if resources is None:
        if row.get('status') == 'complete':
            raise ValueError('Complete follow-up inference lacks process resource measurements')
        return {}
    if not isinstance(resources, dict):
        raise ValueError('Resource usage must be a record')
    if not isinstance(resources.get('hostname'), str) or not resources['hostname']:
        raise ValueError('Inference hostname is missing')
    cpu, setup = resources.get('process_cpu_s'), resources.get('setup_process_cpu_s')
    if (not finite_nonnegative(cpu) or not finite_nonnegative(setup) or setup > cpu
            or not finite_nonnegative(resources.get('process_cpu_per_wall'))):
        raise ValueError('Invalid process CPU accounting')
    ratio = cpu / row['elapsed_s'] if row['elapsed_s'] else 0
    if not math.isclose(resources['process_cpu_per_wall'], ratio, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError('Process CPU / wall ratio differs from recorded durations')
    for name in ('verification_s', 'snapshot_write_s'):
        if name in resources and not finite_nonnegative(resources[name]):
            raise ValueError('Invalid inference verification or writing overhead')
    return resources


def training_endpoint(run, folder, manifest, task):
    """Read either a frozen historical receipt or this task's new training receipt."""
    parent = run / task['parent']
    if not parent.is_file() or sha(parent) != task['parent_sha256']:
        raise ValueError('Frozen parent is missing or changed')
    algorithm = task['algorithm']
    if algorithm == 'parent':
        if (task['checkpoint_sha256'] != task['parent_sha256']
                or task['checkpoint_state_sha256'] != task['parent_state_sha256']):
            raise ValueError('Unchanged parent endpoint differs from its parent')
        checkpoint = run / task['checkpoint']
        if not checkpoint.is_file() or sha(checkpoint) != task['checkpoint_sha256']:
            raise ValueError('Frozen parent endpoint changed')
        return {}, {'file_sha256': task['parent_sha256'], 'state_sha256': task['parent_state_sha256'],
                    'training_seconds': manifest['parent_training_seconds']}, 'reused_parent'
    new = task['new_training_seconds'] > 0
    receipt = folder / 'training/result.json' if new else run / task['training_receipt']
    source_folder = folder if new else run / task['source_task_directory']
    if not receipt.is_file():
        if new:
            return {}, None, 'not_started'
        raise ValueError('Reused endpoint is missing its frozen training receipt')
    if receipt.resolve() != (source_folder / 'training/result.json').resolve():
        raise ValueError('Training receipt is outside its declared task')
    trained = json.loads(receipt.read_text())
    algorithm_task = {**task, 'arm': algorithm}
    record = validate_training(trained, manifest, algorithm_task, source_folder)
    if record is not None:
        handoff = trained.get('handoff', {})
        if (handoff.get('learning_rate_override') != task['learning_rate']
                or handoff.get('rng_state_retained') is not True
                or handoff.get('elite_pool_retained') is not True):
            raise ValueError('Training did not retain parent RNG, elite pool and declared learning rate')
    if not new:
        if record is None:
            raise ValueError('Reused endpoint did not complete its declared training')
        checkpoint = run / task['checkpoint']
        if (not checkpoint.is_file() or sha(checkpoint) != task['checkpoint_sha256']
                or record['file_sha256'] != task['checkpoint_sha256']
                or record['state_sha256'] != task['checkpoint_state_sha256']):
            raise ValueError('Reused checkpoint differs from its complete training receipt')
    return trained, record, trained.get('status', 'not_started') if new else 'reused_complete'


def validate_payload(payload, manifest, task, record, trained):
    expected = {'schema': 1, 'kind': 'finetune_followup_endpoint_evaluation', 'split': 'validation',
                'final_test_access': False, 'selection': 'fixed_training_endpoint', 'method': 'model',
                'arm': task['arm'], 'algorithm': task['algorithm'], 'seed': task['seed'],
                'threads': manifest['threads'], 'ids': manifest['validation_ids'],
                'dataset_file_sha256': manifest['hashes']['data.json'],
                'evaluation_seconds_per_sequence_per_seed': task['evaluation_seconds'],
                'model_setup_charged_per_sequence': True}
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f'Evaluation {key} differs from the frozen follow-up task')
    if payload.get('status') not in ('running', 'complete', 'failed'):
        raise ValueError('Unrecognized evaluation status')
    if record is None:
        if (payload['status'] != 'complete' and not payload.get('results')
                and payload.get('checkpoint_sha256') is None and payload.get('checkpoint_metadata') is None):
            return
        raise ValueError('Evaluated endpoint lacks verified complete training')
    if payload.get('checkpoint_sha256') != record['file_sha256']:
        raise ValueError('Evaluation checkpoint differs from its verified endpoint')
    metadata = payload.get('checkpoint_metadata')
    if metadata is None and payload['status'] != 'complete' and not payload.get('results'):
        return
    if not isinstance(metadata, dict):
        raise ValueError('Checkpoint metadata is missing')
    algorithm = task['algorithm']
    common = {'schema': 1, 'threads': manifest['threads'], 'seed': task['seed'],
              'training_seconds': task['lineage_training_seconds'],
              'dataset_sha256': manifest['hashes']['data.json'], 'state_sha256': record['state_sha256']}
    if algorithm == 'parent':
        common.update(kind='allocation_training_state', mode='patternboost', stage='patternboost', condition='patternboost')
    else:
        common.update(kind='finetune_training_state', mode='finetune', stage=algorithm, condition=algorithm,
                      arm=algorithm, base_training_seconds=manifest['parent_training_seconds'],
                      additional_training_seconds=task['training_seconds'],
                      parent_file_sha256=task['parent_sha256'], parent_state_sha256=task['parent_state_sha256'])
        if metadata.get('segment_counters') != trained.get('segment_counters'):
            raise ValueError('Training result counters differ from evaluated checkpoint counters')
        if metadata.get('effective_training_settings', {}).get('learning_rate') != task['learning_rate']:
            raise ValueError('Checkpoint learning rate differs from its frozen control')
        if metadata.get('reference_state_sha256') != task['parent_state_sha256']:
            raise ValueError('Checkpoint reference differs from its frozen parent')
    if any(metadata.get(key) != value for key, value in common.items()):
        raise ValueError('Checkpoint metadata differs from the algorithm, seed or training lineage')
    for row in payload.get('results', []):
        for field, value in (('checkpoint_stage', metadata['stage']), ('checkpoint_condition', metadata['condition']),
                             ('checkpoint_seed', task['seed']), ('checkpoint_dataset_sha256', metadata['dataset_sha256'])):
            if row.get(field) != value:
                raise ValueError('Sequence checkpoint provenance differs from the declared endpoint')


def contrasts(arms):
    result = [(arm, 'parent') for arm in arms if arm != 'parent']
    result += [(arm, 'patternboost') for arm in arms if arm not in ('parent', 'patternboost')]
    result += [('grpo_low_lr', 'grpo')]
    result += [(arm, 'patternboost_low_lr') for arm in ('grpo_low_lr', 'dr_grpo', 'reinforce', 'dpo')]
    result += [(arm, 'grpo_low_lr') for arm in ('dr_grpo', 'reinforce', 'dpo')]
    return [(arm, ref) for arm, ref in result if arm in arms and ref in arms]


def aggregate(points, ids, arms, comparisons):
    summary, pairs, paired = [], [], []
    lookup = {(p['comparison'], p['budget'], p['id'], p['arm'], p['seed']): p for p in points}
    for comparison, budget in comparisons:
        for seq_id in ids:
            for arm in arms:
                group = [p for p in points if (p['comparison'], p['budget'], p['id'], p['arm']) ==
                         (comparison, budget, seq_id, arm)]
                complete = [p for p in group if p['complete']]
                values = [p['contacts'] for p in complete if p['contacts'] is not None]
                summary.append({'comparison': comparison, 'budget': budget, 'id': seq_id,
                                'length': group[0]['length'], 'arm': arm, 'valid_runs': len(values),
                                'complete_runs': len(complete), 'recorded_runs': sum(p['recorded'] for p in group),
                                'planned_runs': len(group), 'best_contacts': max(values) if values else None,
                                'mean_contacts': statistics.mean(values) if values else None,
                                'sd_contacts': statistics.stdev(values) if len(values) > 1 else None})
            for arm, reference in contrasts(arms):
                group = [p for p in points if (p['comparison'], p['budget'], p['id'], p['arm']) ==
                         (comparison, budget, seq_id, arm)]
                differences = []
                for point in group:
                    other = lookup.get((comparison, budget, seq_id, reference, point['seed']))
                    eligible = bool(other and other['complete'] and point['complete']
                                    and other['contacts'] is not None and point['contacts'] is not None)
                    delta = point['contacts'] - other['contacts'] if eligible else None
                    pairs.append({'comparison': comparison, 'budget': budget, 'id': seq_id, 'arm': arm,
                                  'reference': reference, 'seed': point['seed'], 'paired': eligible, 'difference': delta})
                    if eligible:
                        differences.append(delta)
                paired.append({'comparison': comparison, 'budget': budget, 'id': seq_id, 'arm': arm,
                               'reference': reference, 'paired_runs': len(differences), 'planned_runs': len(group),
                               'mean_difference': statistics.mean(differences) if differences else None,
                               'sd_difference': statistics.stdev(differences) if len(differences) > 1 else None,
                               'positive': sum(x > 0 for x in differences), 'tied': differences.count(0),
                               'negative': sum(x < 0 for x in differences)})
    return summary, pairs, paired


def summarize(directory):
    from finetune_followup_launch import task_directory, verify
    analysis_start = time.monotonic()
    run = Path(directory).resolve()
    manifest = verify(run)
    data = json.loads((run / 'data.json').read_text())
    ids, tasks, threads = manifest['validation_ids'], manifest['tasks'], manifest['threads']
    sources = {row['id']: row for row in data['validation']}
    times, attempts = manifest['inference_timepoints_s'], manifest['attempt_checkpoints']
    arms = [arm for arm in ARMS if any(t['arm'] == arm for t in tasks)]
    errors, states, stored, training = [], [], {}, []
    witnesses = 0
    for task in tasks:
        folder = task_directory(run, task)
        state = {'task': task['task'], 'arm': task['arm'], 'seed': task['seed'], 'status': 'not_started',
                 'launch_status': 'not_started', 'training_status': 'not_started', 'verified_sequences': 0,
                 'complete_sequences': 0, 'planned_sequences': len(ids), 'error': None}
        states.append(state)
        try:
            launch = json.loads((folder / 'launch.json').read_text()) if (folder / 'launch.json').exists() else {}
            state.update(launch_status=launch.get('status', 'not_started'), error=launch.get('error'))
            trained, record, training_status = training_endpoint(run, folder, manifest, task)
            state['training_status'] = training_status
            if task['new_training_seconds'] and record is not None:
                training.append({**trained, 'task': task['task'], 'arm': task['arm'], 'algorithm': task['algorithm'],
                                 'learning_rate': task['learning_rate']})
            path = folder / 'evaluation.json'
            if not path.exists():
                state['status'] = state['launch_status']
                if state['status'] == 'complete':
                    raise ValueError('Complete launch has no endpoint evaluation')
                continue
            payload = json.loads(path.read_text())
            validate_payload(payload, manifest, task, record, trained)
            if payload['status'] == 'complete':
                if (launch.get('status') != 'complete' or launch.get('software') != manifest['software_at_freeze']
                        or launch.get('task') != task):
                    raise ValueError('Complete endpoint lacks complete launch and frozen software receipts')
            seen, rows, local_witnesses = set(), [], 0
            for row in payload.get('results', []):
                seq_id = row.get('id')
                if seq_id not in ids or seq_id in seen:
                    raise ValueError('Duplicate or unexpected validation sequence')
                seen.add(seq_id)
                if row.get('status') not in ('running', 'complete', 'failed'):
                    raise ValueError('Unknown per-sequence evaluation status')
                if payload['status'] == 'complete' and row['status'] != 'complete':
                    raise ValueError('Complete endpoint includes an unfinished sequence')
                checked = verify_row(row, sources[seq_id], {'method': 'model', 'seed': task['seed'],
                                                           'seconds': task['evaluation_seconds']})
                prefixes, count = verify_attempts(row, sources[seq_id], attempts)
                resource = verify_resources(row)
                if resource and resource.get('threads') != threads:
                    raise ValueError('Inference resource threads differ from frozen task')
                rows.append((seq_id, row, checked, prefixes, resource))
                local_witnesses += checked['witnesses'] + count
            if payload['status'] == 'complete' and (seen != set(ids) or not all(c['complete_budget'] for _, _, c, _, _ in rows)):
                raise ValueError('Complete endpoint lacks all sequences or full time coverage')
            state.update(status=payload['status'], verified_sequences=len(rows),
                         complete_sequences=sum(c['complete_budget'] for _, _, c, _, _ in rows),
                         error=payload.get('error') or state['error'])
            for seq_id, row, checked, prefixes, resource in rows:
                stored[task['arm'], task['seed'], seq_id] = {'row': row, 'checked': checked,
                    'prefixes': prefixes, 'resources': resource, 'complete': payload['status'] == 'complete'}
            witnesses += local_witnesses
            if (len(states) % 6 == 0):
                print(json.dumps({'verified_tasks_so_far': len(states), 'planned_tasks': len(tasks),
                                  'verified_witness_records': witnesses}), flush=True)
        except (ValueError, TypeError, KeyError, OSError) as exc:
            state.update(status='verification_failed', error=str(exc))
            errors.append(f'{folder}: {exc}')

    comparisons = [('time', t) for t in times] + [('attempts', n) for n in attempts]
    points, throughput = [], []
    for task in tasks:
        for seq_id in ids:
            value = stored.get((task['arm'], task['seed'], seq_id))
            for comparison, budget in comparisons:
                if value:
                    point = at_time(value['row'], value['checked'], budget) if comparison == 'time' else value['prefixes'][budget]
                else:
                    point = {'covered': False, 'contacts': None}
                points.append({'comparison': comparison, 'budget': budget, 'id': seq_id,
                               'length': len(sources[seq_id]['sequence']), 'arm': task['arm'], 'seed': task['seed'],
                               'recorded': value is not None, 'complete': bool(value and value['complete'] and point['covered']),
                               'covered': point['covered'], 'contacts': point['contacts'],
                               'prefix_details': point if comparison == 'attempts' else None})
            if value:
                row, resource = value['row'], value['resources']
                wall = row['elapsed_s']
                throughput.append({'arm': task['arm'], 'seed': task['seed'], 'id': seq_id,
                    'complete': value['complete'], 'elapsed_s': wall, 'setup_s': row['setup_s'],
                    'attempted_folds': row['attempted_folds'], 'complete_folds': row['generated_complete_folds'],
                    'valid_rate': row['valid_fold_rate'],
                    'complete_folds_per_wall_s': row['generated_complete_folds'] / wall if wall else None,
                    'process_cpu_s': resource.get('process_cpu_s'),
                    'process_cpu_per_wall': resource.get('process_cpu_per_wall'),
                    'complete_folds_per_cpu_s': (row['generated_complete_folds'] / resource['process_cpu_s']
                                               if resource.get('process_cpu_s') else None),
                    'hostname': resource.get('hostname'), 'cpu_affinity': resource.get('cpu_affinity')})
                throughput[-1].update(verification_s=resource.get('verification_s'),
                                      snapshot_write_s=resource.get('snapshot_write_s'))
    summary, pairs, paired = aggregate(points, ids, arms, comparisons)
    actual = sum(v['row']['elapsed_s'] for v in stored.values())
    new_training = sum(t['new_training_seconds'] for t in tasks)
    costs = {'planned_new_training_allocated_cpu_hours': new_training * threads / 3600,
             'planned_inference_allocated_cpu_hours': sum(t['evaluation_seconds'] * len(ids) for t in tasks) * threads / 3600,
             'verified_new_training_allocated_cpu_hours': sum(t['compute_elapsed_s'] for t in training) * threads / 3600,
             'verified_inference_allocated_cpu_hours': actual * threads / 3600,
             'training_lineage_allocated_cpu_hours': sum(t['lineage_training_seconds'] for t in tasks) * threads / 3600,
             'report_verification_and_analysis_s': time.monotonic() - analysis_start,
             'historical_training_repeated': False}
    output = {'schema': 1, 'kind': 'validation_finetune_followup_report', 'final_test_access': False,
              'complete_tasks': sum(s['status'] == 'complete' for s in states), 'planned_tasks': len(tasks),
              'verified_witness_records': witnesses, 'errors': errors, 'tasks': states,
              'summary': summary, 'paired': paired, 'pairs': pairs, 'points': points,
              'throughput': throughput, 'training': training, 'costs': costs,
              'runs': [{'arm': arm, 'seed': seed, 'id': seq_id, **value} for (arm, seed, seq_id), value in stored.items()]}
    with (run / 'analysis.json').open('w') as handle:
        json.dump(output, handle, separators=(',', ':'))
        handle.write('\n')
    for filename, rows in [('summary.csv', summary), ('paired.csv', paired), ('throughput.csv', throughput)]:
        write_csv(run / filename, rows, list(rows[0]) if rows else ['id', 'arm'])
    (run / 'report.md').write_text(render(output, manifest, arms, comparisons) + '\n')
    print(json.dumps({'report': str(run / 'report.md'), 'complete_tasks': output['complete_tasks'],
                      'verified_witness_records': witnesses, 'errors': errors}))
    if errors:
        raise SystemExit(1)
    return output


def render(output, manifest, arms, comparisons):
    lines = ['# PatternBoost fine-tuning controls and inference follow-up', '',
             'Validation-only screening. Final test sets remain unevaluated. Fixed endpoints are evaluated without checkpoint selection or solver repair.', '',
             f"Tasks: {output['complete_tasks']}/{output['planned_tasks']} complete. Independently verified witness records: {output['verified_witness_records']}; verification errors: {len(output['errors'])}.", '',
             'The original two-hour PatternBoost parents and saved 30-minute branches are reused. Only the low-learning-rate PatternBoost and GRPO controls receive new training: 30 minutes from the same original parent, at 3e-5. Parent endpoints have two hours of training; all other endpoints have 2.5 hours including example generation.', '',
             'Timed best-of-search is the primary comparison. Ordered fixed-attempt prefixes diagnose sampling quality separately from throughput; traps consume attempts, and incomplete prefixes do not count as results. Neither diagnostic is a new training seed.', '',
             'Cells show mean best contacts ± sample SD across independent trained seeds. All planned seeds must have complete evaluation and a valid fold for an unannotated cell; brackets give valid/complete/planned otherwise. Missing results remain missing; zero contacts remain zero. Full counts, individual seed results and negative paired changes are retained in CSV and analysis.json.', '']
    lines += ['During execution, this report reads sequence entries already collected in evaluation.json. In-flight or failed-sequence snapshots can also remain under tasks/*/sequences/; they do not create completed results here.', '']
    by_key = {(r['comparison'], r['budget'], r['id'], r['arm']): r for r in output['summary']}
    for comparison, budget in comparisons:
        title = f'{budget / 60:g} minutes per sequence and seed' if comparison == 'time' else f'First {budget} attempted folds (diagnostic)'
        lines += [f'## {title}', '', '| Sequence | Length | ' + ' | '.join(arms) + ' |',
                  '|---|---:|' + '---:|' * len(arms)]
        for seq_id in manifest['validation_ids']:
            rows = [by_key[comparison, budget, seq_id, arm] for arm in arms]
            cells = []
            for row in rows:
                cell = number(row['mean_contacts'])
                if row['sd_contacts'] is not None:
                    cell += ' ± ' + number(row['sd_contacts'])
                if row['valid_runs'] != row['planned_runs'] or row['complete_runs'] != row['planned_runs']:
                    cell += f" [{row['valid_runs']}/{row['complete_runs']}/{row['planned_runs']}]"
                cells.append(cell)
            lines.append(f"| {seq_id} | {rows[0]['length']} | " + ' | '.join(cells) + ' |')
        lines.append('')
    lines += ['## Paired comparison summary', '',
              'Descriptive equal-sequence averages of same-seed paired differences, using only sequences with every planned pair present. Better/tied/worse counts compare sequence means, not individual training runs. Three trained policy seeds remain three independent policy replications across the validation set. Full per-sequence differences and their SD are in paired.csv. Learning-rate controls are not an equal hyperparameter-tuning competition.', '',
              '| Budget | Method | Reference | Fully paired sequences | Mean contact difference | Better / tied / worse sequences |',
              '|---|---|---|---:|---:|---:|']
    endpoints = [('time', max(manifest['inference_timepoints_s'])), ('attempts', max(manifest['attempt_checkpoints']))]
    for comparison, budget in endpoints:
        for arm, reference in contrasts(arms):
            rows = [r for r in output['paired'] if (r['comparison'], r['budget'], r['arm'], r['reference']) ==
                    (comparison, budget, arm, reference) and r['paired_runs'] == r['planned_runs'] and r['paired_runs'] > 0]
            differences = [r['mean_difference'] for r in rows]
            mean = statistics.mean(differences) if differences else None
            label = f'{budget:g}s' if comparison == 'time' else f'{budget} attempts'
            lines.append(f"| {label} | {arm} | {reference} | {len(rows)}/{len(manifest['validation_ids'])} | {number(mean)} | {sum(x > 0 for x in differences)}/{differences.count(0)}/{sum(x < 0 for x in differences)} |")
    lines += ['', '## Inference throughput', '',
              'Totals use complete evaluations. Fold rate divides total valid folds by total elapsed time, including setup. Process CPU / wall is measured process CPU time divided by elapsed time, not allocated cores. Host and affinity records and per-sequence rates are in throughput.csv and analysis.json. Fixed-attempt results help assess whether timed differences persist when sampling counts match.', '',
              '| Method | Complete sequence evaluations | Complete folds | Valid rate | Complete folds / wall second | Process CPU / wall |',
              '|---|---:|---:|---:|---:|---:|']
    for arm in arms:
        rows = [r for r in output['throughput'] if r['arm'] == arm and r['complete']]
        folds, tries = sum(r['complete_folds'] for r in rows), sum(r['attempted_folds'] for r in rows)
        wall = sum(r['elapsed_s'] for r in rows)
        cpu = sum(r['process_cpu_s'] for r in rows) if rows and all(r['process_cpu_s'] is not None for r in rows) else None
        lines.append(f"| {arm} | {len(rows)} | {folds} | {number(folds / tries if tries else None, 3)} | {number(folds / wall if wall else None)} | {number(cpu / wall if cpu is not None and wall else None)} |")
    costs = output['costs']
    lines += ['', '## Compute accounting and interpretation', '',
              f"Planned new training: {costs['planned_new_training_allocated_cpu_hours']:.2f} allocated CPU-hours. Planned inference: {costs['planned_inference_allocated_cpu_hours']:.2f} allocated CPU-hours. Verified new training: {costs['verified_new_training_allocated_cpu_hours']:.3f}; recorded verified inference: {costs['verified_inference_allocated_cpu_hours']:.3f} allocated CPU-hours.", '',
              f"Summed scientific training lineages across evaluated endpoints: {costs['training_lineage_allocated_cpu_hours']:.2f} allocated CPU-hours, including reused parents and branches. This is not new cluster consumption. Shared historical training is not rerun. Allocated core-wall products are not measured CPU utilization; inspect accounting.txt for Slurm utilization and peak RAM.", '',
              f"Report verification and analysis took {costs['report_verification_and_analysis_s']:.2f} seconds, outside inference budgets. In-worker geometry verification and intermediate snapshot writes are charged to inference; their durations are retained separately in throughput.csv.", '',
              'Previous validation results informed this follow-up. These are exploratory learning-rate controls and inference diagnostics, not final-test superiority claims. No new CP-SAT or greedy searches occur, so this report makes no new end-to-end superiority comparison with those solvers.', '',
              '## Saved task status', '',
              '| Task | Method | Seed | Launch | Training | Evaluation | Verified / complete / planned sequences | Error |',
              '|---:|---|---:|---|---|---|---:|---|']
    for state in output['tasks']:
        error = str(state['error'] or '—').replace('|', '/').replace('\n', ' ')
        lines.append(f"| {state['task']} | {state['arm']} | {state['seed']} | {state['launch_status']} | {state['training_status']} | {state['status']} | {state['verified_sequences']}/{state['complete_sequences']}/{state['planned_sequences']} | {error} |")
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    summarize(parser.parse_args().directory)


if __name__ == '__main__':
    main()
