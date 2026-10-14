"""Frozen-policy inference with timed curves and ordered attempt-count prefixes.

The policy, batched sampler, legal masks and seeded CPU generator are unchanged.
An attempt is a returned batch entry, in batch order, including traps. Prefix
comparisons require every attempt in that prefix to resolve within the deadline.
"""
import copy
import math
import os
import socket
import time

from evaluate import verified_score


def prefix_snapshot(records, target, seconds):
    """A reached but interrupted prefix must never become a count-budget result."""
    rows = records[:target]
    reached = len(rows) == target
    covered = reached and all(row['status'] in ('complete', 'trapped')
                              and row['seconds'] <= seconds for row in rows)
    valid = [row for row in rows if row['status'] == 'complete']
    best = max(valid, key=lambda row: row['contacts']) if valid and covered else None
    return {'attempts': target, 'reached': reached, 'covered': covered,
            'observed_attempts': len(rows), 'seconds': rows[-1]['seconds'] if rows else None,
            'complete_folds': len(valid),
            'trapped_folds': sum(row['status'] == 'trapped' for row in rows),
            'interrupted_folds': sum(row['status'] == 'interrupted' for row in rows),
            'discarded_after_budget': sum(row['status'] == 'after_budget' for row in rows),
            'best_contacts': best['contacts'] if best else None,
            'best_positions': copy.deepcopy(best['positions']) if best else None}


def evaluate_sequence(row, method, seed, seconds, threads, checkpoint, batch_size=8,
                      data_hash=None, checkpoint_validator=None,
                      attempt_checkpoints=(128, 512, 1024), progress=None):
    if method != 'model':
        raise ValueError('Follow-up inference evaluates standalone frozen policies only')
    if (not math.isfinite(seconds) or seconds <= 0 or type(threads) is not int or threads < 1
            or type(batch_size) is not int or batch_size != 8):
        raise ValueError('Positive budget and CPU threads, with the preserved batch size eight, required')
    targets = list(attempt_checkpoints)
    if (not targets or any(type(value) is not int or value < 1 for value in targets)
            or targets != sorted(set(targets))):
        raise ValueError('Attempt checkpoints must be unique increasing positive integers')
    if checkpoint_validator is None:
        raise ValueError('An explicit frozen-endpoint validator is required')
    start, cpu_start = time.monotonic(), time.process_time()
    deadline = start + seconds
    sequence = row['sequence']
    result = {'id': row['id'], 'sequence': sequence, 'length': len(sequence), 'seed': seed,
              'method': method, 'budget_s': seconds, 'best_contacts': None, 'best_positions': None,
              'curve': [], 'attempted_folds': 0, 'generated_complete_folds': 0,
              'verified_witnesses': 0, 'trapped_folds': 0, 'interrupted_folds': 0,
              'discarded_after_budget': 0, 'solver_status': None, 'bound': None,
              'proof_within_budget': False, 'bound_recorded_s': None,
              'gap_contacts': None, 'relative_gap': None, 'bound_recorded_after_budget': None,
              'status': 'running', 'batch_size': batch_size,
              'attempt_checkpoints': targets, 'ordered_attempts': [], 'attempt_prefixes': [],
              'attempt_order': 'batch_return_order_including_traps_and_interruptions',
              'prefix_record_limit': targets[-1]}
    resources = {'hostname': socket.gethostname(), 'pid': os.getpid(), 'threads': threads,
                 'slurm_cpus_per_task': os.environ.get('SLURM_CPUS_PER_TASK'),
                 'cpu_affinity': sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None,
                 'verification_s': 0.0, 'snapshot_write_s': 0.0,
                 'overhead_note': 'Verification and intermediate writes consume inference budget; final write duration is excluded from its own saved snapshot.'}
    result['resource_usage'] = resources

    def snapshot(status=None):
        if status is not None:
            result['status'] = status
        elapsed = time.monotonic() - start
        result['elapsed_s'] = elapsed
        result['valid_fold_rate'] = (result['generated_complete_folds'] / result['attempted_folds']
                                     if result['attempted_folds'] else None)
        result['attempt_prefixes'] = [prefix_snapshot(result['ordered_attempts'], n, seconds) for n in targets]
        resources['process_cpu_s'] = time.process_time() - cpu_start
        resources['process_cpu_per_wall'] = resources['process_cpu_s'] / elapsed if elapsed else None
        if progress:
            write_start = time.monotonic()
            progress(result)
            if status is None:  # Only intermediate writes consume the measured search budget.
                resources['snapshot_write_s'] += time.monotonic() - write_start

    try:
        import torch
        from model import Policy, sample
        torch.set_num_threads(threads)
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        checkpoint_validator(saved, data_hash)
        policy = Policy(saved['model_config'])
        policy.load_state_dict(saved['model_state'])
        policy.eval()
        generator = torch.Generator(device='cpu').manual_seed(seed)
        result.update(checkpoint_stage=saved.get('stage'), checkpoint_condition=saved.get('condition'),
                      checkpoint_seed=saved.get('seed'), checkpoint_selection=saved.get('selection'),
                      checkpoint_dataset_sha256=saved.get('dataset_sha256'))
        result['setup_s'] = time.monotonic() - start
        resources['setup_process_cpu_s'] = time.process_time() - cpu_start
        last_snapshot = result['setup_s']
        while time.monotonic() < deadline:
            trajectories = sample(policy, [sequence] * batch_size, generator=generator, deadline=deadline)
            if len(trajectories) != batch_size:
                raise ValueError('Preserved batched sampler did not return eight ordered attempts')
            before = result['attempted_folds']
            for trajectory in trajectories:
                result['attempted_folds'] += 1
                index = result['attempted_folds']
                record = {'attempt_index': index, 'status': None, 'seconds': None,
                          'contacts': None, 'positions': None}
                if trajectory and trajectory.get('complete'):
                    verify_start = time.monotonic()
                    try:
                        score = verified_score(sequence, trajectory['positions'])
                    finally:
                        resources['verification_s'] += time.monotonic() - verify_start
                    if score != trajectory.get('contacts'):
                        raise ValueError('Reported contacts disagree with independent verifier')
                    elapsed = time.monotonic() - start
                    record['seconds'] = elapsed
                    if elapsed > seconds:
                        record['status'] = 'after_budget'
                        result['discarded_after_budget'] += 1
                    else:
                        record.update(status='complete', contacts=score,
                                      positions=copy.deepcopy(trajectory['positions']) if index <= targets[-1] else None)
                        result['generated_complete_folds'] += 1
                        result['verified_witnesses'] += 1
                        if result['best_contacts'] is None or score > result['best_contacts']:
                            result['best_contacts'] = score
                            result['best_positions'] = copy.deepcopy(trajectory['positions'])
                            result['curve'].append({'seconds': elapsed, 'contacts': score,
                                                    'positions': copy.deepcopy(trajectory['positions']),
                                                    'attempt_index': index})
                else:
                    elapsed = time.monotonic() - start
                    record['seconds'] = elapsed
                    if (trajectory and trajectory.get('interrupted')) or elapsed >= seconds:
                        record['status'] = 'interrupted'
                        result['interrupted_folds'] += 1
                    else:
                        record['status'] = 'trapped'
                        result['trapped_folds'] += 1
                if index <= targets[-1]:
                    result['ordered_attempts'].append(record)
            elapsed = time.monotonic() - start
            if any(before < n <= result['attempted_folds'] for n in targets) or elapsed - last_snapshot >= 30:
                snapshot()
                last_snapshot = time.monotonic() - start
        snapshot('complete')
    except Exception as exc:
        result['error'] = str(exc)
        snapshot('failed')
        raise
    return result
