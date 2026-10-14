"""Independently verify the frozen final policy comparison and retain missingness.

Synthetic held-out tests and the previously inspected public benchmarks are
reported separately. Sequence evaluations do not become extra trained seeds.
"""
import argparse
import json
from pathlib import Path
import statistics
import time

from finetune_report import number, write_csv
from finetune_followup_report import verify_attempts, verify_resources
from inference_report import at_time, verify_row


def validate_launch(launch, payload, manifest, task):
    if launch.get('task') != task:
        raise ValueError('Evaluation launch task differs from the frozen final task')
    if launch.get('software') != manifest['software_at_freeze']:
        raise ValueError('Evaluation runtime software differs from the frozen campaign')
    if launch.get('status') not in ('running', 'complete', 'failed'):
        raise ValueError('Unrecognized evaluation launch status')
    if payload.get('status') == 'complete' and launch['status'] != 'complete':
        raise ValueError('Completed evaluation lacks a completed launch record')


def validate_payload(payload, manifest, task, endpoint):
    expected = {
        'schema': 1, 'kind': 'final_policy_evaluation', 'split': task['split'],
        'final_test_access': True, 'selection': 'fixed_training_endpoint',
        'arm': task['arm'], 'method': task['method'], 'seed': task['seed'],
        'threads': manifest['threads'], 'ids': task['sequence_ids'],
        'seconds_per_sequence': task['seconds'],
        'dataset_file_sha256': manifest['hashes']['data.json'],
        'model_setup_charged_per_sequence': True,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f'Evaluation {key} differs from its frozen final task')
    if payload.get('status') not in ('running', 'complete', 'failed'):
        raise ValueError('Unrecognized final evaluation status')
    if not isinstance(payload.get('results'), list):
        raise ValueError('Evaluation results must be a list')
    if task['method'] != 'model':
        if endpoint is not None or payload.get('checkpoint_sha256') is not None or payload.get('checkpoint_metadata') is not None:
            raise ValueError('Plain solver or greedy baseline unexpectedly uses a checkpoint')
        return
    if endpoint is None:
        raise ValueError('Model evaluation lacks a verified complete training endpoint')
    if payload.get('checkpoint_sha256') != endpoint['file_sha256']:
        raise ValueError('Evaluated checkpoint differs from its verified training endpoint')
    metadata = payload.get('checkpoint_metadata')
    if not isinstance(metadata, dict) or metadata != endpoint['metadata']:
        raise ValueError('Evaluated metadata differs from the verified training lineage')
    required = {'schema': 1, 'kind': 'final_policy_training_state', 'arm': task['arm'],
                'seed': task['seed'], 'threads': manifest['threads'],
                'training_seconds': manifest['training_lineage_seconds'],
                'dataset_sha256': manifest['hashes']['data.json'],
                'state_sha256': endpoint['state_sha256']}
    if any(metadata.get(key) != value for key, value in required.items()):
        raise ValueError('Verified endpoint does not match this method, seed or training budget')
    for row in payload['results']:
        for key, value in {
            'checkpoint_stage': metadata.get('stage'),
            'checkpoint_condition': metadata.get('condition'),
            'checkpoint_seed': task['seed'],
            'checkpoint_dataset_sha256': manifest['hashes']['data.json'],
        }.items():
            if row.get(key) != value:
                raise ValueError(f'Sequence {key} differs from its verified checkpoint')


def summarize_cells(points, sources, arms, seeds):
    """Retain all denominators; a verified zero is an observation, not missing."""
    groups = {}
    for point in points:
        key = (point['split'], point['id'], point['comparison'], point['budget'], point['arm'])
        groups.setdefault(key, []).append(point)
    rows = []
    for (split, seq_id, comparison, budget, arm), group in groups.items():
        if len({p['seed'] for p in group}) != len(group):
            raise ValueError('A seed is counted twice for one sequence and method')
        valid = [p['contacts'] for p in group if p['complete'] and p['contacts'] is not None]
        source = sources[(split, seq_id)]
        rows.append({
            'split': split, 'id': seq_id, 'length': source['length'], 'arm': arm,
            'hydrophobic_fraction': source['sequence'].count('H') / source['length'],
            'comparison': comparison, 'budget': budget,
            'planned_runs': len(seeds), 'recorded_runs': sum(p['recorded'] for p in group),
            'complete_runs': sum(p['complete'] for p in group), 'valid_runs': len(valid),
            'best_contacts': max(valid) if valid else None,
            'mean_contacts': statistics.mean(valid) if valid else None,
            'sample_sd': statistics.stdev(valid) if len(valid) > 1 else None,
            'cp_proofs': sum(p['complete'] and p.get('proven_optimal', False) for p in group),
            'best_known_contacts': source.get('best_known_contacts'),
            'reference': source.get('reference'), 'optimum_status': source.get('optimum_status'),
        })
    return rows


def seed_macros(points, sources, manifest):
    """Average sequences within each seed first; incomplete panels stay missing."""
    ids = {split: [seq_id for (which, seq_id) in sources if which == split]
           for split in ('test', 'benchmark')}
    held_out = set(manifest.get('held_out_training_lengths', []))
    groups = {}
    for point in points:
        key = (point['split'], point['comparison'], point['budget'], point['arm'], point['seed'])
        groups.setdefault(key, {})[point['id']] = point
    rows = []
    for (split, comparison, budget, arm, seed), group in groups.items():
        strata = {'all': ids[split]}
        if held_out:
            strata['held_out_lengths'] = [seq_id for seq_id in ids[split]
                                          if sources[(split, seq_id)]['length'] in held_out]
            strata['seen_lengths'] = [seq_id for seq_id in ids[split]
                                      if sources[(split, seq_id)]['length'] not in held_out]
        for stratum, planned in strata.items():
            if not planned:
                continue
            values = [group[seq_id]['contacts'] for seq_id in planned
                      if seq_id in group and group[seq_id]['complete'] and group[seq_id]['contacts'] is not None]
            complete = len(values) == len(planned)
            rows.append({'split': split, 'stratum': stratum, 'comparison': comparison,
                         'budget': budget, 'arm': arm, 'seed': seed,
                         'valid_sequences': len(values), 'planned_sequences': len(planned),
                         'complete': complete, 'mean_contacts': statistics.mean(values) if complete else None})
    return rows


def macro_summaries(macros, seeds):
    groups = {}
    for row in macros:
        key = (row['split'], row['stratum'], row['comparison'], row['budget'], row['arm'])
        groups.setdefault(key, []).append(row)
    results = []
    for (split, stratum, comparison, budget, arm), rows in groups.items():
        values = [row['mean_contacts'] for row in rows if row['complete']]
        results.append({'split': split, 'stratum': stratum, 'comparison': comparison,
                        'budget': budget, 'arm': arm, 'complete_seeds': len(values),
                        'planned_seeds': len(seeds),
                        'mean_contacts': statistics.mean(values) if values else None,
                        'sample_sd': statistics.stdev(values) if len(values) > 1 else None})
    return results


def paired_macros(macros, contrasts, seeds):
    """Each difference is one seed's complete-panel average, never a fold replica."""
    indexed = {(r['split'], r['stratum'], r['comparison'], r['budget'], r['arm'], r['seed']): r
               for r in macros}
    panels = sorted({(r['split'], r['stratum'], r['comparison'], r['budget']) for r in macros})
    results, individual = [], []
    for split, stratum, comparison, budget in panels:
        for arm, reference in contrasts:
            differences = []
            for seed in seeds:
                left = indexed.get((split, stratum, comparison, budget, arm, seed))
                right = indexed.get((split, stratum, comparison, budget, reference, seed))
                difference = (left['mean_contacts'] - right['mean_contacts']
                              if left and right and left['complete'] and right['complete'] else None)
                individual.append({'split': split, 'stratum': stratum, 'comparison': comparison,
                                   'budget': budget, 'arm': arm, 'reference': reference,
                                   'seed': seed, 'mean_difference': difference})
                if difference is not None:
                    differences.append(difference)
            results.append({'split': split, 'stratum': stratum, 'comparison': comparison,
                            'budget': budget, 'arm': arm, 'reference': reference,
                            'paired_seeds': len(differences), 'planned_seeds': len(seeds),
                            'mean_difference': statistics.mean(differences) if differences else None,
                            'sample_sd': statistics.stdev(differences) if len(differences) > 1 else None,
                            'positive_seeds': sum(d > 0 for d in differences),
                            'tied_seeds': sum(d == 0 for d in differences),
                            'negative_seeds': sum(d < 0 for d in differences)})
    return results, individual


def cell(row):
    if row is None:
        return '—'
    value = number(row['mean_contacts'])
    if row['sample_sd'] is not None:
        value += ' ± ' + number(row['sample_sd'])
    if row['valid_runs'] != row['planned_runs'] or row['complete_runs'] != row['planned_runs']:
        value += f" [{row['valid_runs']}/{row['complete_runs']}/{row['planned_runs']}]"
    return value


def training_accounting(run, manifest):
    """Validate each physical training segment once, including shared parents."""
    from final_launch import training_directory, verified_training
    rows, errors = [], []
    for task in manifest['tasks']:
        if not task.get('training_seconds'):
            continue
        folder = training_directory(run, task)
        row = {'task': task['task'], 'arm': task['arm'], 'seed': task['seed'],
               'planned_seconds': task['training_seconds'], 'status': 'not_started',
               'verified_complete_seconds': None, 'checkpoint_io_seconds': None, 'error': None}
        rows.append(row)
        receipt_path = folder / 'result.json'
        if not receipt_path.exists():
            continue
        try:
            receipt = json.loads(receipt_path.read_text())
            row.update(status=receipt.get('status'), error=receipt.get('error'))
            if receipt.get('status') == 'complete':
                endpoint = verified_training(run, task, manifest)
                receipt = endpoint['receipt']
                row.update(verified_complete_seconds=receipt['compute_elapsed_s'],
                           checkpoint_io_seconds=receipt.get('checkpoint_io_s'))
            elif row['status'] not in ('running', 'failed', 'interrupted', 'insufficient_training'):
                raise ValueError('Unrecognized training segment status')
        except (ValueError, TypeError, KeyError, OSError) as exc:
            row.update(status='verification_failed', error=str(exc))
            errors.append(f'{receipt_path}: {exc}')
    return rows, errors


def summarize(directory):
    from final_launch import task_directory, verify, verified_endpoint
    start = time.monotonic()
    run = Path(directory).resolve()
    manifest = verify(run)
    data = json.loads((run / 'data.json').read_text())
    sources = {(split, row['id']): row for split in ('test', 'benchmark') for row in data[split]}
    tasks = manifest['evaluation_tasks']
    seeds = manifest['seeds']
    arms = manifest['arms']
    training, errors = training_accounting(run, manifest)
    states, verified, endpoints = [], {}, {}
    witnesses = 0
    last_progress = start
    for task in tasks:
        folder = task_directory(run, task)
        state = {'task': task['task'], 'arm': task['arm'], 'seed': task['seed'], 'split': task['split'],
                 'status': 'not_started', 'verified_sequences': 0, 'complete_sequences': 0,
                 'planned_sequences': len(task['sequence_ids']), 'error': None}
        states.append(state)
        path = folder / 'evaluation.json'
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text())
            launch = json.loads((folder / 'launch.json').read_text())
            validate_launch(launch, payload, manifest, task)
            state['launch_status'] = launch['status']
            endpoint = None
            if task['method'] == 'model':
                key = task['training_task']
                if key not in endpoints:
                    endpoints[key] = verified_endpoint(run, task)
                endpoint = endpoints[key]
            validate_payload(payload, manifest, task, endpoint)
            seen, staged = set(), []
            for row in payload['results']:
                seq_id = row.get('id')
                if seq_id not in task['sequence_ids'] or seq_id in seen:
                    raise ValueError('Unexpected or duplicate final sequence')
                seen.add(seq_id)
                source = sources[(task['split'], seq_id)]
                checked = verify_row(row, source, task)
                prefixes, prefix_witnesses = {}, 0
                if task['method'] == 'model':
                    if payload['status'] == 'complete' and row.get('status') != 'complete':
                        raise ValueError('Complete model evaluation contains an unfinished sequence')
                    prefixes, prefix_witnesses = verify_attempts(row, source, manifest['attempt_checkpoints'])
                    verify_resources(row)
                staged.append((seq_id, row, checked, prefixes))
                witnesses += checked['witnesses'] + prefix_witnesses
            if payload['status'] == 'complete' and (seen != set(task['sequence_ids'])
                    or any(not checked['complete_budget'] for _, _, checked, _ in staged)):
                raise ValueError('Complete evaluation lacks every requested sequence and full budget or timely proof')
            state.update(status=payload['status'], error=payload.get('error'),
                         verified_sequences=len(staged),
                         complete_sequences=sum(checked['complete_budget'] for _, _, checked, _ in staged))
            for seq_id, row, checked, prefixes in staged:
                key = (task['split'], task['arm'], task['seed'], seq_id)
                if key in verified:
                    raise ValueError('A final sequence/seed is evaluated by duplicate tasks')
                verified[key] = {'row': row, 'checked': checked, 'prefixes': prefixes,
                                 'task_complete': payload['status'] == 'complete'}
        except (ValueError, KeyError, TypeError, OSError) as exc:
            state.update(status='verification_failed', error=str(exc))
            errors.append(f'{path}: {exc}')
        if time.monotonic() - last_progress >= 30:
            print(json.dumps({'verified_evaluation_tasks_so_far': len(states),
                              'planned_evaluation_tasks': len(tasks),
                              'verified_witness_records': witnesses}), flush=True)
            last_progress = time.monotonic()

    points, curves, throughput = [], [], []
    for task in tasks:
        for seq_id in task['sequence_ids']:
            saved = verified.get((task['split'], task['arm'], task['seed'], seq_id))
            budgets = [('time', t) for t in manifest['inference_timepoints_s']]
            if task['method'] == 'model':
                budgets += [('attempts', n) for n in manifest['attempt_checkpoints']]
            for comparison, budget in budgets:
                measurement = (at_time(saved['row'], saved['checked'], budget) if comparison == 'time'
                               else saved['prefixes'][budget]) if saved else {}
                complete = bool(saved and saved['task_complete'] and measurement.get('covered', False))
                points.append({'split': task['split'], 'id': seq_id, 'arm': task['arm'], 'seed': task['seed'],
                               'comparison': comparison, 'budget': budget, 'recorded': saved is not None,
                               'complete': complete, 'covered': measurement.get('covered', False),
                               'contacts': measurement.get('contacts'),
                               'upper_bound': measurement.get('upper_bound'),
                               'gap_contacts': measurement.get('gap_contacts'),
                               'proven_optimal': measurement.get('proven_optimal', False)})
            if saved:
                row, checked = saved['row'], saved['checked']
                for point in row['curve']:
                    curves.append({'split': task['split'], 'id': seq_id, 'arm': task['arm'],
                                   'seed': task['seed'], **point})
                generated, attempted = row.get('generated_complete_folds'), row.get('attempted_folds')
                resources = row.get('resource_usage', {})
                throughput.append({'split': task['split'], 'id': seq_id, 'arm': task['arm'],
                                   'seed': task['seed'], 'complete': saved['task_complete'] and checked['complete_budget'],
                                   'allocated_seconds': task['seconds'], 'actual_seconds': row['elapsed_s'],
                                   'generated_complete_folds': generated, 'attempted_folds': attempted,
                                   'valid_rate': generated / attempted if attempted else None,
                                   'complete_folds_per_second': generated / row['elapsed_s'] if generated is not None and row['elapsed_s'] else None,
                                   'hostname': resources.get('hostname'),
                                   'process_cpu_s': resources.get('process_cpu_s'),
                                   'process_cpu_per_wall': resources.get('process_cpu_per_wall'),
                                   'solver_status': row.get('solver_status'),
                                   'bound': row.get('bound'), 'bound_recorded_s': row.get('bound_recorded_s'),
                                   'bound_scope': row.get('bound_scope'),
                                   'verification_s': resources.get('verification_s'),
                                   'snapshot_write_s': resources.get('snapshot_write_s')})

    summary = summarize_cells(points, sources, arms, seeds)
    macros = seed_macros(points, sources, {**manifest, 'held_out_training_lengths': data.get('held_out_training_lengths', [])})
    macro_summary = macro_summaries(macros, seeds)
    contrasts = manifest.get('contrasts', [(arm, 'patternboost') for arm in arms if arm != 'patternboost'])
    paired, seed_pairs = paired_macros(macros, contrasts, seeds)
    model_lineages = {(task['arm'], task['seed']) for task in tasks if task['method'] == 'model'}
    planned_training = sum(task.get('training_seconds', 0) for task in manifest['tasks'])
    planned_inference = sum(task['seconds'] * len(task['sequence_ids']) for task in tasks)
    lineage_seconds = manifest['training_lineage_seconds']
    costs = {
        'planned_physical_training_allocated_cpu_hours': planned_training * manifest['threads'] / 3600,
        'verified_complete_physical_training_allocated_cpu_hours': sum(row['verified_complete_seconds'] or 0 for row in training) * manifest['threads'] / 3600,
        'planned_inference_allocated_cpu_hours': planned_inference * manifest['threads'] / 3600,
        'scientific_model_training_lineage_allocated_cpu_hours': len(model_lineages) * lineage_seconds * manifest['threads'] / 3600,
        'verified_complete_inference_allocated_cpu_hours': sum(r['actual_seconds'] for r in throughput if r['complete']) * manifest['threads'] / 3600,
        'model_training_seconds_per_seed': lineage_seconds,
        'amortization_scenarios': [{'future_sequences': n, 'training_seconds_per_sequence': lineage_seconds / n,
                                   'inference_seconds': max(manifest['inference_timepoints_s']),
                                   'total_seconds_per_sequence': lineage_seconds / n + max(manifest['inference_timepoints_s'])}
                                  for n in (1, 8, 80, 88, 1000)],
    }
    output = {'schema': 1, 'kind': 'final_policy_analysis', 'complete_tasks': sum(s['status'] == 'complete' for s in states),
              'planned_tasks': len(tasks), 'verified_witness_records': witnesses, 'errors': errors,
              'tasks': states, 'training': training, 'timepoints': points, 'summary': summary, 'seed_macros': macros,
              'macro_summary': macro_summary, 'paired': paired, 'seed_pairs': seed_pairs,
              'throughput': throughput, 'costs': costs,
              'report_verification_seconds': time.monotonic() - start}
    (run / 'analysis.json').write_text(json.dumps(output, indent=2, allow_nan=False) + '\n')
    for name, rows in [('summary', summary), ('training', training), ('timepoints', points), ('seed_macros', macros),
                       ('macro_summary', macro_summary), ('paired', paired), ('seed_pairs', seed_pairs),
                       ('throughput', throughput)]:
        write_csv(run / f'{name}.csv', rows, list(rows[0]) if rows else [])
    write_csv(run / 'curves.csv', curves, ['split', 'id', 'arm', 'seed', 'seconds', 'contacts', 'attempt_index'])
    _write_markdown(run, manifest, data, output)
    print(json.dumps({'report': str(run / 'report.md'), 'complete_tasks': output['complete_tasks'],
                      'planned_tasks': len(tasks), 'verified_witness_records': witnesses, 'errors': errors}))
    if errors:
        raise SystemExit(1)
    return output


def _write_markdown(run, manifest, data, result):
    arms, seeds = manifest['arms'], manifest['seeds']
    primary = max(manifest['inference_timepoints_s'])
    lookup = {(r['split'], r['id'], r['comparison'], r['budget'], r['arm']): r for r in result['summary']}
    lines = ['# Final standalone folding-policy comparison', '',
             f"Evaluation tasks: {result['complete_tasks']}/{result['planned_tasks']} complete. Independently verified witness records: {result['verified_witness_records']}; verification errors: {len(result['errors'])}.", '',
             'All learned endpoints use the same architecture and 2.5-hour training lineage. PatternBoost branches share each seed\'s two-hour parent; scratch policies train for 2.5 hours. Fixed endpoints are evaluated without solver repair or test-based checkpoint selection.', '',
             f'Primary results are best valid contacts within {primary:g} seconds per sequence and seed, including setup and model loading. Means and sample SD use {len(seeds)} planned independent trained seeds. Brackets show valid/complete/planned counts when incomplete; missing folds remain missing and genuine zero scores remain zero.', '',
             'The independent training replication is the seed. Multiple sequences, sampled folds and inference timepoints do not create extra training replications. All 1/5/10-minute observations, fixed-attempt diagnostics, per-seed values, bounds and incomplete progress are retained in analysis.json and the CSV files.', '']
    for split, title in [('test', 'Held-out synthetic test sequences'), ('benchmark', 'Previously inspected public benchmarks')]:
        lines += [f'## {title}', '']
        if split == 'benchmark':
            lines += ['These public sequences were inspected in earlier research; they are an external benchmark, not an untouched research test set. Published contact counts are reference values, with their source and proof qualifications retained below.', '']
        headers = ['Sequence', 'Length'] + (['Published reference'] if split == 'benchmark' else []) + list(arms)
        lines += ['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |']
        for source in data[split]:
            row = [source['id'], str(source['length'])]
            if split == 'benchmark':
                row.append(str(source.get('best_known_contacts', '—')))
            row += [cell(lookup.get((split, source['id'], 'time', primary, arm))) for arm in arms]
            lines.append('| ' + ' | '.join(row) + ' |')
        lines.append('')
        if split == 'benchmark':
            for source in data[split]:
                if source.get('reference'):
                    lines.append(f"- {source['id']}: {source.get('optimum_status', 'proof status unspecified')}; [reference]({source['reference']}).")
            lines.append('')
    lines += ['## Aggregate scores across complete seed panels', '',
              'Average sequence scores within each seed first, then summarize those seed averages. A seed contributes only when every planned sequence in that split/stratum is complete and has a valid fold. Longer or H-rich strings can have more contacts; these raw macro averages are descriptive, not normalized measures of difficulty.', '',
              'Per-sequence actual H fractions are retained in summary.csv. The dataset has no separate nominal generation-fraction field, so no additional composition strata are inferred.', '',
              '| Split | Length stratum | Method | Complete seeds | Mean of seed averages | SD across seeds |',
              '|---|---|---|---:|---:|---:|']
    for row in result['macro_summary']:
        if row['comparison'] == 'time' and row['budget'] == primary:
            lines.append(f"| {row['split']} | {row['stratum']} | {row['arm']} | {row['complete_seeds']}/{row['planned_seeds']} | {number(row['mean_contacts'])} | {number(row['sample_sd'])} |")
    lines += ['', '## Paired changes at the primary inference budget', '',
              'One observation is each seed\'s difference in mean contacts over a complete split. Shared-parent branch comparisons match the training seed; matching a classical solver seed is an indexing convention, not a claim of common random-number streams. Negative differences remain negative. No automatic significance or superiority claim is made.', '',
              '| Split | Method | Reference | Paired seeds | Mean difference | SD | Positive / tied / negative seeds |',
              '|---|---|---|---:|---:|---:|---:|']
    for row in result['paired']:
        if row['stratum'] == 'all' and row['comparison'] == 'time' and row['budget'] == primary:
            lines.append(f"| {row['split']} | {row['arm']} | {row['reference']} | {row['paired_seeds']}/{row['planned_seeds']} | {number(row['mean_difference'])} | {number(row['sample_sd'])} | {row['positive_seeds']}/{row['tied_seeds']}/{row['negative_seeds']} |")
    costs = result['costs']
    lines += ['', '## Compute accounting', '',
              f"Planned physical training: {costs['planned_physical_training_allocated_cpu_hours']:.2f} allocated CPU-hours. Planned inference: {costs['planned_inference_allocated_cpu_hours']:.2f} allocated CPU-hours. Scientific model-training lineage charge: {costs['scientific_model_training_lineage_allocated_cpu_hours']:.2f} allocated CPU-hours. Shared physical parents do not erase any method's training cost.", '',
              f"Verified complete physical training: {costs['verified_complete_physical_training_allocated_cpu_hours']:.3f} allocated CPU-hours. Verified complete inference: {costs['verified_complete_inference_allocated_cpu_hours']:.3f} allocated CPU-hours. Report verification took {result['report_verification_seconds']:.2f} seconds, outside inference budgets. Core-wall products are allocations, not measured CPU use; consult Slurm accounting for actual CPU utilization and peak memory.", '',
              'The main solver comparison uses equal inference time and reports offline training separately. Amortized deployment cost is training time / N + inference time per sequence, for an explicitly stated N. The scenarios below are cost arithmetic only: CP-SAT/greedy were not run for these longer totals, and no scores are invented at those budgets.', '',
              '| Future sequences N | Training seconds / sequence | Inference seconds | Total seconds / sequence |',
              '|---:|---:|---:|---:|']
    for row in costs['amortization_scenarios']:
        lines.append(f"| {row['future_sequences']} | {row['training_seconds_per_sequence']:.2f} | {row['inference_seconds']:g} | {row['total_seconds_per_sequence']:.2f} |")
    lines += ['', '## Physical training segment status', '',
              'Each shared parent appears once. Incomplete or interrupted training is not charged as verified completed training; Slurm accounting still records any resources it consumed.', '',
              '| Task | Method | Seed | Status | Planned seconds | Verified completed seconds | Error |',
              '|---:|---|---:|---|---:|---:|---|']
    for row in result['training']:
        error = str(row['error'] or '—').replace('|', '\\|').replace('\n', ' ')
        lines.append(f"| {row['task']} | {row['arm']} | {row['seed']} | {row['status']} | {row['planned_seconds']:g} | {number(row['verified_complete_seconds'])} | {error} |")
    lines += ['', '## Evaluation status', '', '| Task | Split | Method | Seed | Status | Complete / planned sequences | Error |',
              '|---:|---|---|---:|---|---:|---|']
    for row in result['tasks']:
        error = str(row['error'] or '—').replace('|', '\\|').replace('\n', ' ')
        lines.append(f"| {row['task']} | {row['split']} | {row['arm']} | {row['seed']} | {row['status']} | {row['complete_sequences']}/{row['planned_sequences']} | {error} |")
    if result['errors']:
        lines += ['', 'Verification errors invalidate the affected task results; review analysis.json before using this report.']
    (run / 'report.md').write_text('\n'.join(lines) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    summarize(parser.parse_args().directory)


if __name__ == '__main__':
    main()
