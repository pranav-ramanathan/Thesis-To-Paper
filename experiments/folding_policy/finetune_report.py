"""Verify fixed fine-tuning endpoints against their shared PatternBoost parents."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics

from inference_report import at_time, finite_nonnegative, verify_row

ARMS = ('parent', 'patternboost', 'grpo', 'dr_grpo', 'reinforce', 'dpo')
UPDATE_KEYS = {'patternboost': 'supervised_updates', 'grpo': 'rl_updates',
               'dr_grpo': 'rl_updates', 'reinforce': 'rl_updates', 'dpo': 'preference_updates'}


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1048576), b''):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path, rows, fields):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in fields} for row in rows)


def number(value, digits=2):
    return '—' if value is None else f'{value:.{digits}f}'


def validate_training(trained, manifest, task, folder):
    """A successful process is insufficient: require the full fixed training lineage."""
    if trained.get('status') != 'complete':
        return None
    lineage = manifest['parent_training_seconds'] + task['training_seconds']
    expected = {'arm': task['arm'], 'seed': task['seed'], 'threads': manifest['threads'],
                'budget_s': task['training_seconds'],
                'base_training_seconds': manifest['parent_training_seconds'],
                'additional_training_seconds': task['training_seconds'],
                'training_seconds': lineage, 'dataset_sha256': manifest['hashes']['data.json'],
                'initial_state_sha256': task['parent_state_sha256'],
                'parent_file_sha256': task['parent_sha256'],
                'parent_state_sha256': task['parent_state_sha256'],
                'reference_state_sha256': task['parent_state_sha256']}
    for key, value in expected.items():
        if trained.get(key) != value:
            raise ValueError(f'Training {key} differs from the frozen task or parent')
    if (trained.get('test_evaluated') is not False
            or trained.get('final_test_enabled') is not False
            or trained.get('selection', {}).get('selected') is not False
            or not finite_nonnegative(trained.get('compute_elapsed_s'))
            or trained['compute_elapsed_s'] < task['training_seconds']):
        raise ValueError('Complete training lacks full budget coverage or validation-only provenance')
    if trained.get('training_settings', {}).get('learning_rate') != task['learning_rate']:
        raise ValueError('Training learning rate differs from the frozen task')
    handoff = trained.get('handoff', {})
    if (handoff.get('pretraining_sha256') != task['parent_state_sha256']
            or handoff.get('finetune_initial_sha256') != task['parent_state_sha256']):
        raise ValueError('Training handoff differs from its frozen parent')
    if 'parent_optimizer_sha256' in task:
        if (handoff.get('pretraining_optimizer_sha256') != task['parent_optimizer_sha256']
                or handoff.get('optimizer_moments_retained') is not True):
            raise ValueError('Training optimizer handoff differs from the frozen parent')
    counters = trained.get('segment_counters')
    if not isinstance(counters, dict) or any(type(value) is not int or value < 0 for value in counters.values()):
        raise ValueError('Training segment counters must be nonnegative integers')
    for smaller, larger in (('valid_folds', 'generated_folds'), ('zero_advantage_groups', 'rl_groups')):
        if smaller in counters and larger in counters and counters[smaller] > counters[larger]:
            raise ValueError(f'Training {smaller} exceeds {larger}')
    if counters.get(UPDATE_KEYS[task['arm']], 0) < 1:
        raise ValueError('Complete training has no updates for the declared method')
    records = [record for record in trained.get('checkpoint_records', []) if record.get('filename') == 'final.pt']
    if len(records) != 1:
        raise ValueError('Training lacks one unique final checkpoint record')
    record = records[0]
    checkpoint = folder / 'training' / 'final.pt'
    if (not checkpoint.is_file() or sha(checkpoint) != record.get('file_sha256')
            or record.get('selected') is not False or record.get('training_seconds') != lineage
            or not record.get('state_sha256')
            or record['state_sha256'] != trained.get('final_state_sha256')):
        raise ValueError('Training checkpoint changed or differs from its final state or lineage')
    return record


def validate_payload(payload, manifest, task, record, trained):
    seconds = manifest['evaluation_seconds_per_sequence_per_seed']
    expected = {'schema': 1, 'kind': 'finetune_endpoint_evaluation', 'split': 'validation',
                'final_test_access': False, 'selection': 'fixed_training_endpoint', 'method': 'model',
                'arm': task['arm'], 'seed': task['seed'], 'threads': manifest['threads'],
                'ids': manifest['validation_ids'],
                'dataset_file_sha256': manifest['hashes']['data.json'],
                'evaluation_seconds_per_sequence_per_seed': seconds,
                'model_setup_charged_per_sequence': True}
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f'Evaluation {key} differs from the frozen validation endpoint')
    if payload.get('status') not in ('running', 'complete', 'failed'):
        raise ValueError('Unrecognized evaluation status')
    if record is None:
        if (payload['status'] != 'complete' and not payload.get('results')
                and payload.get('checkpoint_sha256') is None and payload.get('checkpoint_metadata') is None):
            return
        raise ValueError('Evaluated endpoint lacks verified complete training')
    if payload.get('checkpoint_sha256') != record['file_sha256']:
        raise ValueError('Evaluation checkpoint differs from the verified endpoint')
    metadata = payload.get('checkpoint_metadata')
    if metadata is None and payload['status'] != 'complete' and not payload.get('results'):
        return
    if not isinstance(metadata, dict):
        raise ValueError('Checkpoint metadata is missing')
    common = {'schema': 1, 'threads': manifest['threads'], 'seed': task['seed'], 'training_seconds': record['training_seconds'],
              'dataset_sha256': manifest['hashes']['data.json'], 'state_sha256': record['state_sha256']}
    if task['arm'] == 'parent':
        common.update(kind='allocation_training_state', mode='patternboost', stage='patternboost', condition='patternboost')
    else:
        common.update(kind='finetune_training_state', mode='finetune', stage=task['arm'], condition=task['arm'],
                      arm=task['arm'], base_training_seconds=manifest['parent_training_seconds'],
                      additional_training_seconds=task['training_seconds'],
                      parent_file_sha256=task['parent_sha256'], parent_state_sha256=task['parent_state_sha256'])
        if ('segment_counters' in metadata
                and metadata['segment_counters'] != trained.get('segment_counters')):
            raise ValueError('Result training counters differ from the evaluated checkpoint')
    if any(metadata.get(key) != value for key, value in common.items()):
        raise ValueError('Checkpoint metadata differs from the declared training lineage')
    for row in payload.get('results', []):
        for field, expected_value in (('checkpoint_stage', metadata.get('stage')),
                                      ('checkpoint_condition', metadata.get('condition')),
                                      ('checkpoint_seed', task['seed']),
                                      ('checkpoint_dataset_sha256', manifest['hashes']['data.json'])):
            if row.get(field) != expected_value:
                raise ValueError('Sequence checkpoint provenance differs from the declared endpoint')


def summarize(directory):
    from finetune_launch import task_directory, verify
    run = Path(directory).resolve()
    manifest = verify(run)
    data = json.loads((run / 'data.json').read_text())
    ids, tasks = manifest['validation_ids'], manifest['tasks']
    sources = {row['id']: row for row in data['validation']}
    seconds, threads = manifest['evaluation_seconds_per_sequence_per_seed'], manifest['threads']
    errors, states, stored_rows, training = [], [], {}, []
    witnesses = 0
    for task in tasks:
        folder = task_directory(run, task)
        state = {'task': task['task'], 'arm': task['arm'], 'seed': task['seed'],
                 'status': 'not_started', 'training_status': 'not_started', 'launch_status': 'not_started',
                 'verified_sequences': 0, 'complete_sequences': 0, 'planned_sequences': len(ids), 'error': None}
        states.append(state)
        path = folder / 'evaluation.json'
        try:
            launch = json.loads((folder / 'launch.json').read_text()) if (folder / 'launch.json').exists() else {}
            state['launch_status'], state['error'] = launch.get('status', 'not_started'), launch.get('error')
            parent = run / task['parent']
            if not parent.is_file() or sha(parent) != task['parent_sha256']:
                raise ValueError('Frozen parent checkpoint changed or is missing')
            trained, record = {}, None
            if task['arm'] == 'parent':
                state['training_status'] = 'reused_parent'
                record = {'file_sha256': task['parent_sha256'], 'state_sha256': task['parent_state_sha256'],
                          'training_seconds': manifest['parent_training_seconds']}
            else:
                result_path = folder / 'training' / 'result.json'
                if result_path.exists():
                    trained = json.loads(result_path.read_text())
                    state['training_status'] = trained.get('status', 'not_started')
                    record = validate_training(trained, manifest, task, folder)
                    if record is not None:
                        training.append({'task': task['task'], 'arm': task['arm'], 'seed': task['seed'],
                                         'learning_rate': task['learning_rate'], **trained})
            if not path.exists():
                state['status'] = state['launch_status']
                if state['status'] == 'complete':
                    raise ValueError('Complete launch is missing its endpoint evaluation')
                continue
            payload = json.loads(path.read_text())
            validate_payload(payload, manifest, task, record, trained)
            if payload['status'] == 'complete':
                if launch.get('status') != 'complete':
                    raise ValueError('Complete evaluation lacks a complete launch receipt')
                if launch.get('software') != manifest['software_at_freeze']:
                    raise ValueError('Completed task software differs from the frozen environment')
            seen, rows, verified = set(), [], 0
            for row in payload.get('results', []):
                seq_id = row.get('id')
                if seq_id not in ids or seq_id in seen:
                    raise ValueError('Duplicate or unexpected validation sequence')
                seen.add(seq_id)
                checked = verify_row(row, sources[seq_id], {'method': 'model', 'seed': task['seed'], 'seconds': seconds})
                if ('valid_fold_rate' in row and row['valid_fold_rate'] !=
                        (row['generated_complete_folds'] / row['attempted_folds'] if row['attempted_folds'] else None)):
                    raise ValueError('Valid-fold rate differs from completed and attempted counts')
                rows.append((seq_id, row, checked))
                verified += checked['witnesses']
            if payload['status'] == 'complete' and (seen != set(ids) or not all(c['complete_budget'] for _, _, c in rows)):
                raise ValueError('Complete evaluation lacks all validation sequences or full inference coverage')
            state.update(status=payload['status'], verified_sequences=len(rows),
                         complete_sequences=sum(c['complete_budget'] for _, _, c in rows),
                         error=payload.get('error') or state['error'])
            for seq_id, row, checked in rows:
                stored_rows[(task['arm'], task['seed'], seq_id)] = {
                    'row': row, 'checked': checked, 'complete': payload['status'] == 'complete'}
            witnesses += verified
        except (ValueError, TypeError, KeyError, OSError) as exc:
            state.update(status='verification_failed', error=str(exc))
            errors.append(f'{folder}: {exc}')
    points = []
    for task in tasks:
        for seq_id in ids:
            stored = stored_rows.get((task['arm'], task['seed'], seq_id))
            point = at_time(stored['row'], stored['checked'], seconds) if stored else {'contacts': None, 'covered': False}
            points.append({'id': seq_id, 'length': len(sources[seq_id]['sequence']), 'arm': task['arm'],
                           'seed': task['seed'], 'recorded': stored is not None,
                           'complete': bool(stored and stored['complete'] and point['covered']),
                           'contacts': point['contacts'], 'covered': point['covered'],
                           'complete_folds': stored['row']['generated_complete_folds'] if stored else None,
                           'attempted_folds': stored['row']['attempted_folds'] if stored else None})
    summary, pairs, paired = [], [], []
    for seq_id in ids:
        for arm in ARMS:
            group = [p for p in points if p['id'] == seq_id and p['arm'] == arm]
            if not group:
                continue
            complete = [p for p in group if p['complete']]
            valid = [p for p in complete if p['contacts'] is not None]
            values = [p['contacts'] for p in valid]
            attempts = sum(p['attempted_folds'] for p in complete)
            folds = sum(p['complete_folds'] for p in complete)
            summary.append({'id': seq_id, 'length': len(sources[seq_id]['sequence']), 'arm': arm,
                            'valid_runs': len(valid), 'complete_runs': len(complete),
                            'recorded_runs': sum(p['recorded'] for p in group), 'planned_runs': len(group),
                            'best_contacts': max(values) if values else None,
                            'mean_contacts': statistics.mean(values) if values else None,
                            'sd_contacts': statistics.stdev(values) if len(values) > 1 else None,
                            'complete_folds': folds, 'valid_rate': folds / attempts if attempts else None})
        for arm in ARMS[1:]:
            for reference in ('parent', 'patternboost'):
                if arm == reference:
                    continue
                group = [p for p in points if p['id'] == seq_id and p['arm'] == arm]
                differences = []
                for point in group:
                    other = next((p for p in points if p['id'] == seq_id and p['arm'] == reference
                                  and p['seed'] == point['seed']), None)
                    eligible = bool(other and other['complete'] and point['complete']
                                    and other['contacts'] is not None and point['contacts'] is not None)
                    delta = point['contacts'] - other['contacts'] if eligible else None
                    pairs.append({'id': seq_id, 'arm': arm, 'reference': reference, 'seed': point['seed'],
                                  'paired': eligible, 'difference': delta})
                    if eligible:
                        differences.append(delta)
                paired.append({'id': seq_id, 'arm': arm, 'reference': reference,
                               'paired_runs': len(differences), 'planned_runs': len(group),
                               'mean_difference': statistics.mean(differences) if differences else None,
                               'sd_difference': statistics.stdev(differences) if len(differences) > 1 else None,
                               'positive': sum(d > 0 for d in differences), 'tied': differences.count(0),
                               'negative': sum(d < 0 for d in differences)})
    planned_training = sum(task['training_seconds'] for task in tasks)
    measured_training = sum(t['compute_elapsed_s'] for t in training)
    planned_inference = len(tasks) * len(ids) * seconds
    actual_inference = sum(row['row']['elapsed_s'] for row in stored_rows.values())
    trained_tasks = [t for t in tasks if t['arm'] != 'parent']
    costs = {'planned_new_training_seconds': planned_training,
             'planned_new_training_allocated_cpu_hours': planned_training * threads / 3600,
             'verified_complete_training_seconds': measured_training,
             'verified_complete_training_allocated_cpu_hours': measured_training * threads / 3600,
             'planned_inference_allocated_cpu_hours': planned_inference * threads / 3600,
             'recorded_verified_inference_seconds': actual_inference,
             'recorded_verified_inference_allocated_cpu_hours': actual_inference * threads / 3600,
             'historical_parent_training_seconds_per_seed': manifest['parent_training_seconds'],
             'training_lineage_allocated_cpu_hours': sum(manifest['parent_training_seconds'] + t['training_seconds']
                                                       for t in trained_tasks) * threads / 3600}
    output = {'schema': 1, 'kind': 'validation_finetune_report', 'final_test_access': False,
              'complete_tasks': sum(s['status'] == 'complete' for s in states), 'planned_tasks': len(tasks),
              'verified_witness_records': witnesses, 'errors': errors, 'tasks': states,
              'summary': summary, 'paired': paired, 'pairs': pairs, 'points': points,
              'training': training, 'costs': costs,
              'runs': [{'arm': arm, 'seed': seed, 'id': seq_id, **stored}
                       for (arm, seed, seq_id), stored in stored_rows.items()]}
    (run / 'analysis.json').write_text(json.dumps(output, indent=2) + '\n')
    write_csv(run / 'summary.csv', summary, list(summary[0]) if summary else ['id', 'arm'])
    write_csv(run / 'paired.csv', paired, list(paired[0]) if paired else ['id', 'arm', 'reference'])
    lines = ['# PatternBoost fine-tuning algorithm pilot', '',
             'Validation-only screening from the same saved two-hour PatternBoost parent for each seed. Final test sets remain unevaluated. Primary results use fixed final endpoints, without checkpoint selection or solver repair.', '',
             f"Tasks: {output['complete_tasks']}/{len(tasks)} complete. Independently verified witness records: {witnesses}; verification errors: {len(errors)}.", '',
             f'Each endpoint receives {seconds:g} seconds per validation sequence per seed, including setup and model loading. Contacts are each seed’s best complete fold within that budget. Only verified complete training and complete endpoint evaluations enter means; missing folds remain missing and zero contacts remain zero.', '',
             'The trained seeds are the independent policy replications; evaluating one policy on multiple sequences does not create additional trained seeds. This fixed-configuration screen uses method-specific learning rates and cannot isolate an algorithm’s superiority from its settings.', '',
             '## Fixed endpoint results', '',
             '| Sequence | Length | Method | Valid / complete / recorded / planned | Best | Mean contacts | SD | Complete folds | Valid rate |',
             '|---|---:|---|---:|---:|---:|---:|---:|---:|']
    for row in summary:
        lines.append(f"| {row['id']} | {row['length']} | {row['arm']} | {row['valid_runs']}/{row['complete_runs']}/{row['recorded_runs']}/{row['planned_runs']} | {number(row['best_contacts'], 0)} | {number(row['mean_contacts'])} | {number(row['sd_contacts'])} | {row['complete_folds']} | {number(row['valid_rate'], 3)} |")
    lines += ['', '## Paired changes', '',
              'Positive means the first method obtained more contacts. Same-seed comparisons against the unchanged parent measure the effect of additional training; comparisons against continued PatternBoost match the additional training budget. Negative changes remain negative.', '',
              '| Sequence | Method | Reference | Paired / planned | Mean difference | SD | Positive / tied / negative |',
              '|---|---|---|---:|---:|---:|---:|']
    for row in paired:
        lines.append(f"| {row['id']} | {row['arm']} | {row['reference']} | {row['paired_runs']}/{row['planned_runs']} | {number(row['mean_difference'])} | {number(row['sd_difference'])} | {row['positive']}/{row['tied']}/{row['negative']} |")
    lines += ['', '## Training diagnostics', '',
              'Counters below cover only the additional training segment. Full diagnostic records and verified progress for unfinished evaluations are retained in analysis.json.', '',
              '| Method | Seed | Learning rate | Updates | Generated / valid folds | Zero-advantage groups / groups | Additional compute seconds |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for trained in training:
        counters = trained['segment_counters']
        lines.append(f"| {trained['arm']} | {trained['seed']} | {trained['learning_rate']:g} | {counters.get(UPDATE_KEYS[trained['arm']], 0)} | {counters.get('generated_folds', 0)}/{counters.get('valid_folds', 0)} | {counters.get('zero_advantage_groups', 0)}/{counters.get('rl_groups', 0)} | {trained['compute_elapsed_s']:.2f} |")
    lines += ['', 'These are the last recorded diagnostic snapshots, not averages over all training. Sampled current-minus-parent action log probability is not an exact KL divergence. PatternBoost snapshots can mix input sequences while policy-gradient snapshots usually contain one input, so their unique-fold fractions are not directly comparable. Sampled and completed sequence counts provide this context.', '',
              '| Method | Seed | Diagnostic records | Sampled / complete sequences | Reward min / max / SD | Unique complete fold fraction | Sampled log-probability drift | Gradient norm after clipping |',
              '|---|---:|---:|---:|---:|---:|---:|---:|']
    for trained in training:
        diagnostic = trained.get('diagnostics', {})
        lines.append(f"| {trained['arm']} | {trained['seed']} | {diagnostic.get('records', 0)} | {number(diagnostic.get('sampled_sequence_count'), 0)}/{number(diagnostic.get('complete_sequence_count'), 0)} | {number(diagnostic.get('reward_min'))}/{number(diagnostic.get('reward_max'))}/{number(diagnostic.get('reward_sd'))} | {number(diagnostic.get('unique_complete_fold_fraction'), 3)} | {number(diagnostic.get('sampled_logp_drift_mean'), 4)} | {number(diagnostic.get('gradient_norm_after_clip'), 4)} |")
    lines += ['', '## Compute accounting', '',
              f"Planned new training: {costs['planned_new_training_allocated_cpu_hours']:.2f} allocated CPU-hours. Planned endpoint inference: {costs['planned_inference_allocated_cpu_hours']:.2f} allocated CPU-hours. Verified completed training: {costs['verified_complete_training_allocated_cpu_hours']:.3f} allocated CPU-hours; recorded verified inference: {costs['recorded_verified_inference_allocated_cpu_hours']:.3f} allocated CPU-hours.", '',
              f"The historical {manifest['parent_training_seconds'] / 3600:g}-hour parents are reused, not retrained. Each new method’s scientific training cost includes that entire parent plus its additional training; the summed method-lineage charge is {costs['training_lineage_allocated_cpu_hours']:.2f} allocated CPU-hours. Physical shared-prefix savings do not erase lineage costs. Core-wall products are not measured CPU utilization; use Slurm accounting for utilization and peak memory.", '',
              'This pilot compares fine-tuning configurations. Earlier CP-SAT and greedy runs do not automatically match the new end-to-end training cost; no new solver superiority comparison is claimed.', '',
              '## Saved task status', '',
              '| Task | Arm | Seed | Launch | Training | Evaluation | Verified / complete / planned sequences | Error |',
              '|---:|---|---:|---|---|---|---:|---|']
    for state in states:
        error = str(state['error'] or '—').replace('|', '/').replace('\n', ' ')
        lines.append(f"| {state['task']} | {state['arm']} | {state['seed']} | {state['launch_status']} | {state['training_status']} | {state['status']} | {state['verified_sequences']}/{state['complete_sequences']}/{state['planned_sequences']} | {error} |")
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
