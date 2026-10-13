"""Verify fixed training endpoints and paired allocation-pilot comparisons."""
import argparse
import csv
import json
import math
from pathlib import Path
import statistics

from evaluate import verified_score


PRIMARY_ARMS = ('patternboost', 'balanced', 'more_pretrain', 'rl_scratch')
CONTRASTS = (
    ('balanced_minus_its_pretraining', ('balanced', 'final'), ('patternboost', 'stage_3600')),
    ('more_pretrain_minus_its_pretraining', ('more_pretrain', 'final'), ('patternboost', 'stage_5400')),
    ('balanced_minus_continued_patternboost', ('balanced', 'final'), ('patternboost', 'stage_7200')),
    ('more_pretrain_minus_continued_patternboost', ('more_pretrain', 'final'), ('patternboost', 'stage_7200')),
    ('balanced_minus_rl_scratch', ('balanced', 'final'), ('rl_scratch', 'final')),
    ('more_pretrain_minus_rl_scratch', ('more_pretrain', 'final'), ('rl_scratch', 'final')),
    ('more_pretrain_minus_balanced', ('more_pretrain', 'final'), ('balanced', 'final')),
)


def _finite_nonnegative(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def verify_row(row, source, seed, seconds):
    """Check all witnesses, including earlier improvements, independently."""
    if row.get('id') != source['id'] or row.get('sequence') != source['sequence']:
        raise ValueError('Sequence differs from frozen validation data')
    if row.get('seed') != seed or row.get('method') != 'model':
        raise ValueError('Method or seed differs from its endpoint')
    if row.get('budget_s') != seconds:
        raise ValueError('Inference budget differs from frozen allocation pilot')
    if not _finite_nonnegative(row.get('elapsed_s')) or row['elapsed_s'] < seconds:
        raise ValueError('Inference stopped before its full budget')
    attempted, generated = row.get('attempted_folds'), row.get('generated_complete_folds')
    if (type(attempted) is not int or type(generated) is not int
            or not 0 <= generated <= attempted):
        raise ValueError('Invalid generated or attempted fold count')
    best, positions = row.get('best_contacts'), row.get('best_positions')
    curve = row.get('curve')
    if not isinstance(curve, list):
        raise ValueError('Missing improvement curve')
    witnesses = 0
    if best is None:
        if positions is not None or curve:
            raise ValueError('Missing best score has a claimed witness')
    else:
        if type(best) is not int or best < 0 or not curve or generated < 1:
            raise ValueError('A scored fold needs an integer score and improvement witness')
        if verified_score(source['sequence'], positions) != best:
            raise ValueError('Best contact count does not match its witness')
        witnesses += 1
    previous_time, previous_score = 0, -1
    for point in curve:
        elapsed, score = point.get('seconds'), point.get('contacts')
        if (not _finite_nonnegative(elapsed) or not previous_time <= elapsed <= seconds
                or type(score) is not int or score <= previous_score):
            raise ValueError('Invalid best-so-far curve')
        if verified_score(source['sequence'], point['positions']) != score:
            raise ValueError('Curve contact count does not match its witness')
        witnesses += 1
        previous_time, previous_score = elapsed, score
    if curve and curve[-1]['contacts'] != best:
        raise ValueError('Final curve disagrees with verified best')
    return witnesses


def _endpoint_role(arm, stage):
    return 'primary' if stage == ('stage_7200' if arm == 'patternboost' else 'final') else 'before_rl'


def _write_csv(path, rows, fields):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def _record(trained, filename):
    matches = [row for row in trained.get('checkpoint_records', []) if row.get('filename') == filename]
    if len(matches) != 1:
        raise ValueError('Endpoint lacks one unique training checkpoint record')
    return matches[0]


def _training_errors(task, trained, manifest):
    errors = []
    if trained.get('status') != 'complete':
        return errors
    if trained.get('seed') != task['seed'] or trained.get('dataset_sha256') != manifest['hashes']['data.json']:
        errors.append('Complete training has different seed or dataset provenance')
    expected_mode = task['arm'] if task['arm'] in ('patternboost', 'rl_scratch') else 'rl_branch'
    if trained.get('mode') != expected_mode or trained.get('test_evaluated') is not False:
        errors.append('Complete training has different mode or test-set provenance')
    if trained.get('budget_s') != task['training_seconds']:
        errors.append('Training budget differs from frozen allocation')
    if (not _finite_nonnegative(trained.get('compute_elapsed_s'))
            or trained['compute_elapsed_s'] < task['training_seconds']):
        errors.append('Complete training lacks full elapsed-budget coverage')
    counters = trained.get('counters', {})
    if task['arm'] == 'patternboost' and counters.get('supervised_updates', 0) < 1:
        errors.append('PatternBoost completed without supervised updates')
    if task['arm'] != 'patternboost' and trained.get('segment_counters', counters).get('rl_updates', 0) < 1:
        errors.append('RL completed without policy updates')
    return errors


def summarize(directory):
    from allocation_launch import sha, verify, write_json

    run = Path(directory).resolve()
    manifest = verify(run)
    data = json.loads((run / 'data.json').read_text())
    validation = {row['id']: row for row in data['validation']}
    ids = manifest['validation_ids']
    seeds = sorted({task['seed'] for task in manifest['tasks']})
    seconds = manifest['evaluation_seconds_per_sequence_per_seed']
    task_by_key = {(task['arm'], task['seed']): task for task in manifest['tasks']}
    trained_by_key, launch_by_key, statuses, errors = {}, {}, [], []
    for key, task in task_by_key.items():
        folder = run / 'tasks' / f"{task['arm']}_seed{task['seed']}"
        launch_path, training_path = folder / 'launch.json', folder / 'training' / 'result.json'
        launched = json.loads(launch_path.read_text()) if launch_path.exists() else {}
        trained = json.loads(training_path.read_text()) if training_path.exists() else {}
        launch_by_key[key], trained_by_key[key] = launched, trained
        problems = _training_errors(task, trained, manifest)
        errors.extend(f'{training_path}: {problem}' for problem in problems)
        statuses.append({'arm': task['arm'], 'seed': task['seed'],
                         'launch_status': launched.get('status', 'not_started'),
                         'training_status': trained.get('status', 'not_started'),
                         'training_elapsed_s': trained.get('elapsed_s'),
                         'error': launched.get('error'), 'verification_errors': problems})

    endpoints, all_rows, witnesses = [], {}, 0
    for key, task in task_by_key.items():
        arm, seed = key
        folder = run / 'tasks' / f'{arm}_seed{seed}'
        trained = trained_by_key[key]
        for stage in task['evaluation_stages']:
            path = folder / 'evaluations' / f'{stage}.json'
            endpoint = {'arm': arm, 'seed': seed, 'stage': stage,
                        'role': _endpoint_role(arm, stage), 'status': 'not_recorded',
                        'verified_sequences': 0, 'expected_sequences': len(ids), 'errors': []}
            endpoints.append(endpoint)
            if not path.exists():
                continue
            payload = json.loads(path.read_text())
            try:
                if (payload.get('kind') != 'allocation_endpoint_evaluation'
                        or payload.get('split') != 'validation'
                        or payload.get('final_test_access') is not False
                        or payload.get('selection') != 'fixed_training_endpoint'):
                    raise ValueError('Evaluation is not a fixed endpoint on validation only')
                if payload.get('dataset_file_sha256') != manifest['hashes']['data.json']:
                    raise ValueError('Evaluation dataset differs from frozen data')
                if payload.get('seed') != seed:
                    raise ValueError('Evaluation seed differs from task')
                if (payload.get('threads') != manifest['threads']
                        or payload.get('evaluation_seconds_per_sequence_per_seed') != seconds):
                    raise ValueError('Evaluation resource request differs from frozen allocation')
                if payload.get('status') != 'complete' and payload.get('checkpoint_metadata') is None:
                    endpoint['status'] = payload.get('status', 'partial')
                    endpoint['error'] = payload.get('error')
                    continue
                checkpoint = folder / 'training' / f'{stage}.pt'
                if not checkpoint.is_file() or payload.get('checkpoint_sha256') != sha(checkpoint):
                    raise ValueError('Checkpoint changed or is missing')
                metadata = payload.get('checkpoint_metadata', {})
                record = _record(trained, checkpoint.name)
                expected_mode = arm if arm in ('patternboost', 'rl_scratch') else 'rl_branch'
                expected_stage = 'patternboost' if arm == 'patternboost' else 'rl'
                expected_condition = arm if arm in ('patternboost', 'rl_scratch') else 'patternboost_rl'
                if (payload.get('endpoint') != stage or metadata.get('seed') != seed
                        or metadata.get('stage') != expected_stage or metadata.get('mode') != expected_mode
                        or metadata.get('condition') != expected_condition
                        or metadata.get('dataset_sha256') != manifest['hashes']['data.json']):
                    raise ValueError('Checkpoint stage, seed or dataset differs from endpoint')
                if record.get('file_sha256') != payload['checkpoint_sha256']:
                    raise ValueError('Evaluated checkpoint differs from training checkpoint record')
                if (not record.get('state_sha256') or record['state_sha256'] != metadata.get('state_sha256')):
                    raise ValueError('Checkpoint model state differs from training record')
                lineage = task['stage_lineage_seconds'][stage]
                if record.get('training_seconds') != lineage or metadata.get('training_seconds') != lineage:
                    raise ValueError('Checkpoint cumulative training budget differs from endpoint')
                if task.get('parent_arm'):
                    parent_key = (task['parent_arm'], seed)
                    parent_folder = run / 'tasks' / f'{parent_key[0]}_seed{seed}' / 'training'
                    parent_name = f"{task['resume_stage']}.pt"
                    parent_training = trained_by_key.get(parent_key, {})
                    if (parent_training.get('status') != 'complete'
                            or _training_errors(task_by_key[parent_key], parent_training, manifest)):
                        raise ValueError('Shared parent training did not complete its frozen budget')
                    parent = _record(parent_training, parent_name)
                    handoff = trained.get('handoff') or {}
                    if (not (parent_folder / parent_name).is_file()
                            or sha(parent_folder / parent_name) != parent.get('file_sha256')):
                        raise ValueError('Shared parent checkpoint changed or is missing')
                    lineage_info = trained.get('lineage', {})
                    if (lineage_info.get('parent_file_sha256') != parent['file_sha256']
                            or lineage_info.get('parent_state_sha256') != parent.get('state_sha256')
                            or trained.get('initial_state_sha256') != parent.get('state_sha256')
                            or handoff.get('pretraining_sha256') != parent.get('state_sha256')
                            or handoff.get('rl_initial_sha256') != parent.get('state_sha256')):
                        raise ValueError('Branch does not start from its declared shared parent weights')
                    if (not parent.get('optimizer_sha256')
                            or lineage_info.get('parent_optimizer_sha256') != parent['optimizer_sha256']
                            or handoff.get('pretraining_optimizer_sha256') != parent['optimizer_sha256']
                            or handoff.get('rl_initial_optimizer_sha256') != parent['optimizer_sha256']
                            or any(handoff.get(flag) is not True for flag in
                                   ('optimizer_state_retained', 'rng_state_retained', 'elite_pool_retained'))):
                        raise ValueError('Branch does not retain parent optimizer, RNG and elite-pool state')
                seen, rows, endpoint_witnesses = set(), {}, 0
                for row in payload.get('results', []):
                    seq_id = row.get('id')
                    if seq_id not in ids or seq_id in seen:
                        raise ValueError('Unexpected or duplicate validation sequence')
                    seen.add(seq_id)
                    if (row.get('checkpoint_stage') != metadata['stage']
                            or row.get('checkpoint_condition') != metadata['condition']
                            or row.get('checkpoint_seed') != seed
                            or row.get('checkpoint_dataset_sha256') != metadata['dataset_sha256']):
                        raise ValueError('Sequence checkpoint provenance differs from its endpoint')
                    endpoint_witnesses += verify_row(row, validation[seq_id], seed, seconds)
                    rows[(seq_id, arm, stage, seed)] = row
                witnesses += endpoint_witnesses
                endpoint['verified_sequences'] = len(rows)
                endpoint['checkpoint_sha256'] = payload['checkpoint_sha256']
                endpoint['lineage_training_seconds'] = lineage
                endpoint['recorded_status'] = payload.get('status')
                if payload.get('status') == 'complete' and seen != set(ids):
                    raise ValueError('Complete evaluation does not cover every planned validation sequence')
                training_valid = trained.get('status') == 'complete' and not _training_errors(task, trained, manifest)
                complete = payload.get('status') == 'complete' and seen == set(ids) and training_valid
                endpoint['status'] = 'complete' if complete else 'failed' if payload.get('status') == 'failed' else 'partial'
                endpoint['error'] = payload.get('error')
                endpoint['progress'] = [{'id': seq_id, 'best_contacts': row['best_contacts']}
                                        for (seq_id, _, _, _), row in rows.items()]
                if complete:
                    all_rows.update(rows)
            except (ValueError, KeyError, TypeError) as exc:
                endpoint['status'] = 'verification_failed'
                endpoint['errors'].append(str(exc))
                errors.append(f'{path}: {exc}')

    summary = []
    for task in manifest['tasks']:
        arm = task['arm']
        if task['seed'] != seeds[0]:
            continue
        for stage in task['evaluation_stages']:
            for seq_id in ids:
                rows = [all_rows[(seq_id, arm, stage, seed)] for seed in seeds
                        if (seq_id, arm, stage, seed) in all_rows]
                values = [row['best_contacts'] for row in rows if row['best_contacts'] is not None]
                generated = sum(row['generated_complete_folds'] for row in rows)
                attempted = sum(row['attempted_folds'] for row in rows)
                summary.append({'id': seq_id, 'length': len(validation[seq_id]['sequence']),
                                'arm': arm, 'stage': stage, 'role': _endpoint_role(arm, stage),
                                'valid_runs': len(values), 'complete_runs': len(rows), 'planned_runs': len(seeds),
                                'best_contacts': max(values) if values else None,
                                'mean_contacts': statistics.mean(values) if values else None,
                                'sample_sd': statistics.stdev(values) if len(values) > 1 else None,
                                'generated_complete_folds': generated, 'attempted_folds': attempted,
                                'valid_fold_rate': generated / attempted if attempted else None,
                                'observed_inference_s': sum(row['elapsed_s'] for row in rows)})
    contrasts, contrast_summary = [], []
    for name, after, before in CONTRASTS:
        for seq_id in ids:
            differences = []
            for seed in seeds:
                post = all_rows.get((seq_id, *after, seed))
                pre = all_rows.get((seq_id, *before, seed))
                a = post.get('best_contacts') if post else None
                b = pre.get('best_contacts') if pre else None
                delta = a - b if a is not None and b is not None else None
                if delta is not None:
                    differences.append(delta)
                contrasts.append({'id': seq_id, 'contrast': name, 'seed': seed,
                                  'after_contacts': a, 'before_contacts': b,
                                  'difference': delta, 'paired_complete': delta is not None})
            contrast_summary.append({'id': seq_id, 'contrast': name, 'paired_runs': len(differences),
                                     'planned_runs': len(seeds),
                                     'mean_difference': statistics.mean(differences) if differences else None,
                                     'sample_sd': statistics.stdev(differences) if len(differences) > 1 else None,
                                     'positive': sum(value > 0 for value in differences),
                                     'zero': sum(value == 0 for value in differences),
                                     'negative': sum(value < 0 for value in differences)})
    curves = []
    for (seq_id, arm, stage, seed), row in sorted(all_rows.items()):
        for point in row['curve']:
            curves.append({'id': seq_id, 'arm': arm, 'stage': stage, 'seed': seed,
                           'seconds': point['seconds'], 'contacts': point['contacts'],
                           'budget_s': row['budget_s']})
    threads = manifest['threads']
    measured_training = sum(row['elapsed_s'] for row in trained_by_key.values()
                            if _finite_nonnegative(row.get('elapsed_s')))
    measured_inference = sum(row['elapsed_s'] for row in all_rows.values())
    physical_planned = sum(task['training_seconds'] for task in manifest['tasks'])
    independent_planned = sum(task['lineage_training_seconds'] for task in manifest['tasks'])
    inference_planned = sum(len(task['evaluation_stages']) for task in manifest['tasks']) * len(ids) * seconds
    costs = {'threads': threads, 'planned_shared_training_cpu_hours': physical_planned * threads / 3600,
             'planned_independent_lineage_training_cpu_hours': independent_planned * threads / 3600,
             'planned_endpoint_inference_cpu_hours': inference_planned * threads / 3600,
             'observed_training_wall_seconds': measured_training,
             'observed_verified_complete_inference_wall_seconds': measured_inference,
             'observed_training_allocated_cpu_hours': measured_training * threads / 3600,
             'observed_verified_complete_inference_allocated_cpu_hours': measured_inference * threads / 3600,
             'note': 'Allocated CPU-hours are wall time times requested cores, not measured CPU utilization. '
                     'Saved partial/failed phase timing is retained separately in launch_records.'}
    result = {'schema': 1, 'kind': 'validation_allocation_pilot', 'tasks': statuses,
              'endpoints': endpoints, 'summary': summary, 'contrasts': contrasts,
              'contrast_summary': contrast_summary, 'costs': costs,
              'launch_records': [{**value, 'arm': key[0], 'seed': key[1]} for key, value in launch_by_key.items()],
              'training_records': [dict(arm=key[0], seed=key[1], result=value) for key, value in trained_by_key.items()],
              'verified_witness_records': witnesses, 'errors': errors, 'final_test_evaluated': False}
    write_json(run / 'analysis.json', result)
    _write_csv(run / 'summary.csv', summary, list(summary[0]) if summary else ['id', 'arm', 'stage'])
    _write_csv(run / 'contrast.csv', contrasts, list(contrasts[0]) if contrasts else ['id', 'contrast', 'seed'])
    _write_csv(run / 'curves.csv', curves, ['id', 'arm', 'stage', 'seed', 'seconds', 'contacts', 'budget_s'])
    _write_report(run, manifest, result)
    print(json.dumps({'report': str(run / 'report.md'),
                      'complete_endpoints': sum(row['status'] == 'complete' for row in endpoints),
                      'verified_witness_records': witnesses, 'errors': errors}))
    if errors:
        raise SystemExit(1)
    return result


def _write_report(run, manifest, result):
    def fmt(value, digits=2):
        return '—' if value is None else f'{value:.{digits}f}'

    lines = ['# PatternBoost / RL training-allocation pilot', '',
             'Validation-only engineering comparison. The final test sets remain unevaluated.', '',
             f"Verified witness records: {result['verified_witness_records']}; verification errors: {len(result['errors'])}. "
             'Witness records can repeat the same fold.', '',
             'Primary results use the actual fixed training endpoints, including any deterioration during RL. '
             'No checkpoint is selected by validation score for these comparisons. '
             'Any later validation-selected deployment checkpoint is a separate secondary result.', '',
             f"Each endpoint receives {manifest['evaluation_seconds_per_sequence_per_seed']:g} seconds per validation sequence per seed, "
             'including model loading. Means below average each seed’s best valid fold. '
             'Only complete training and complete endpoint evaluations enter means; missing folds are not zero. '
             'Three seeds support allocation screening, not a final superiority or optimal-ratio claim.', '',
             '## Actual final endpoints', '',
             '| Sequence | Length | Method | Valid / complete / planned | Best | Mean contacts | SD | Complete folds | Valid rate |',
             '|---|---:|---|---:|---:|---:|---:|---:|---:|']
    for row in result['summary']:
        if row['role'] == 'primary':
            lines.append(f"| {row['id']} | {row['length']} | {row['arm']} | {row['valid_runs']}/{row['complete_runs']}/{row['planned_runs']} | "
                         f"{fmt(row['best_contacts'], 0)} | {fmt(row['mean_contacts'])} | {fmt(row['sample_sd'])} | "
                         f"{row['generated_complete_folds']} | {fmt(row['valid_fold_rate'], 3)} |")
    lines += ['', '## Before-RL checkpoints', '',
              'These are the same shared parent weights used to initialize the corresponding RL branches.', '',
              '| Sequence | Parent checkpoint | Valid / complete / planned | Mean contacts | SD |',
              '|---|---|---:|---:|---:|']
    for row in result['summary']:
        if row['role'] == 'before_rl':
            lines.append(f"| {row['id']} | {row['stage']} | {row['valid_runs']}/{row['complete_runs']}/{row['planned_runs']} | "
                         f"{fmt(row['mean_contacts'])} | {fmt(row['sample_sd'])} |")
    lines += ['', '## Paired differences', '',
              'Positive means the first endpoint obtained more contacts. Negative changes remain negative. '
              'Pairs use the same training seed and validation sequence. '
              'Only pairs with two complete endpoints and two valid folds enter the mean.', '',
              '| Sequence | Contrast | Paired / planned | Mean difference | SD | Positive / tied / negative |',
              '|---|---|---:|---:|---:|---:|']
    for row in result['contrast_summary']:
        lines.append(f"| {row['id']} | {row['contrast']} | {row['paired_runs']}/{row['planned_runs']} | "
                     f"{fmt(row['mean_difference'])} | {fmt(row['sample_sd'])} | "
                     f"{row['positive']}/{row['zero']}/{row['negative']} |")
    cost = result['costs']
    lines += ['', '## Compute accounting', '',
              f"Planned shared physical training: {cost['planned_shared_training_cpu_hours']:g} allocated CPU-hours. "
              f"Planned endpoint inference: {cost['planned_endpoint_inference_cpu_hours']:g} allocated CPU-hours. "
              f"Training charged separately to all method lineages: {cost['planned_independent_lineage_training_cpu_hours']:g} allocated CPU-hours.", '',
              f"Saved measured training: {cost['observed_training_wall_seconds'] / 3600:.3f} summed wall-hours "
              f"({cost['observed_training_allocated_cpu_hours']:.3f} allocated CPU-hours). "
              f"Verified complete inference: {cost['observed_verified_complete_inference_wall_seconds'] / 3600:.3f} summed wall-hours "
              f"({cost['observed_verified_complete_inference_allocated_cpu_hours']:.3f} allocated CPU-hours).", '',
              'Shared prefixes reduce physical execution cost. Each final method still receives the full two-hour '
              'training lineage in scientific cost comparisons. Example generation is charged to training. '
              'Slurm accounting supplies actual CPU utilization and peak memory; these products of wall time and cores do not.', '',
              '## Saved task status', '',
              'Status is from saved files, not a live Slurm query. Partial endpoint progress and full training diagnostics are in analysis.json.', '',
              '| Arm | Seed | Launch | Training | Training seconds | Error |', '|---|---:|---|---|---:|---|']
    for row in result['tasks']:
        lines.append(f"| {row['arm']} | {row['seed']} | {row['launch_status']} | {row['training_status']} | "
                     f"{fmt(row['training_elapsed_s'])} | {row['error'] or '—'} |")
    lines += ['', '| Arm | Seed | Endpoint | Status | Verified / planned sequences |', '|---|---:|---|---|---:|']
    for row in result['endpoints']:
        lines.append(f"| {row['arm']} | {row['seed']} | {row['stage']} | {row['status']} | "
                     f"{row['verified_sequences']}/{row['expected_sequences']} |")
    if result['errors']:
        lines += ['', '## Verification errors', '', *['- ' + error for error in result['errors']]]
    (run / 'report.md').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory')
    summarize(parser.parse_args().directory)
