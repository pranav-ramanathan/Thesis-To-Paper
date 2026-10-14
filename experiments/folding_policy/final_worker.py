"""Run one final-study lineage and its fixed test evaluations without retries."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

from final_launch import (find_job, job_directory, task_directory, training_directory,
                          verify, verified_training)
from launch import sha, software, write_json


def budget_available(manifest, seconds, reserve=60):
    remaining = (datetime.fromisoformat(manifest['deadline_utc']) - datetime.now(timezone.utc)).total_seconds()
    if remaining < seconds + reserve:
        raise RuntimeError('Campaign deadline cannot accommodate this full operation; no shortened result is substituted')
    return remaining


def runtime(manifest):
    if software() != manifest['software_at_freeze']:
        raise ValueError('Python or package versions differ from the frozen campaign')
    if int(os.environ.get('SLURM_CPUS_PER_TASK', manifest['threads'])) != manifest['threads']:
        raise ValueError('CPU allocation differs from the frozen campaign')


def evaluate_one(run, evaluation_id, sequence_id, output):
    """A fresh process per input gives every search the same cold-start accounting."""
    manifest = verify(run)
    runtime(manifest)
    task = manifest['evaluation_tasks'][evaluation_id]
    if task['task'] != evaluation_id or sequence_id not in task['sequence_ids']:
        raise ValueError('Input is outside the frozen evaluation task')
    data = json.loads((run / 'data.json').read_text())
    row = next(row for row in data[task['split']] if row['id'] == sequence_id)
    budget_available(manifest, task['seconds'])
    if output.exists():
        raise ValueError('Refusing to overwrite an existing sequence evaluation')
    output.parent.mkdir(parents=True, exist_ok=True)
    if task['method'] == 'model':
        from final_train import validate_final_checkpoint
        from finetune_followup_evaluate import evaluate_sequence
        endpoint = json.loads((task_directory(run, task) / 'endpoint.json').read_text())
        checkpoint = Path(endpoint['checkpoint'])
        if checkpoint != training_directory(run, manifest['tasks'][task['training_task']]) / 'final.pt':
            raise ValueError('Endpoint path does not match frozen task')
        # Full weight/optimizer checks occur inside the inference setup clock.
        def validate(saved, data_hash):
            validate_final_checkpoint(saved, data_hash, seed=task['seed'], arm=task['arm'],
                                      parent_sha=endpoint['metadata']['parent_file_sha256'])
            for key, value in endpoint['metadata'].items():
                if saved.get(key) != value:
                    raise ValueError(f'Inference endpoint changed: {key}')
        result = evaluate_sequence(row, 'model', task['seed'], task['seconds'], manifest['threads'],
                                   checkpoint, data_hash=manifest['hashes']['data.json'],
                                   checkpoint_validator=validate, attempt_checkpoints=manifest['attempt_checkpoints'],
                                   progress=lambda result: compact_write(output, result))
        if sha(checkpoint) != endpoint['file_sha256']:
            raise ValueError('Checkpoint changed during evaluation')
        if result['status'] != 'complete':
            raise RuntimeError('Model evaluation failed: ' + str(result.get('error')))
    else:
        from evaluate import evaluate_sequence
        result = evaluate_sequence(row, task['method'], task['seed'], task['seconds'], manifest['threads'])
        result['status'] = 'complete'
        compact_write(output, result)
    return result


def compact_write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, separators=(',', ':')) + '\n')
    temporary.replace(path)


def evaluate_split(run, manifest, task, endpoint):
    folder = task_directory(run, task)
    folder.mkdir(parents=True, exist_ok=False)
    if endpoint:
        # Receipts can be large; the immutable identity and hashes suffice here.
        write_json(folder / 'endpoint.json', {k: v for k, v in endpoint.items() if k != 'receipt'})
    payload = dict(schema=1, kind='final_policy_evaluation', split=task['split'],
                   final_test_access=True, selection='fixed_training_endpoint',
                   arm=task['arm'], method=task['method'], seed=task['seed'], threads=manifest['threads'],
                   ids=task['sequence_ids'], seconds_per_sequence=task['seconds'],
                   dataset_file_sha256=manifest['hashes']['data.json'],
                   model_setup_charged_per_sequence=True,
                   checkpoint_sha256=endpoint['file_sha256'] if endpoint else None,
                   checkpoint_metadata=endpoint['metadata'] if endpoint else None,
                   status='running', results=[])
    write_json(folder / 'launch.json', dict(task=task, software=software(), status='running',
                                          hostname=socket.gethostname(), started_utc=datetime.now(timezone.utc).isoformat()))
    compact_write(folder / 'evaluation.json', payload)
    try:
        for sequence_id in task['sequence_ids']:
            remaining = budget_available(manifest, task['seconds'])
            output = folder / 'sequences' / f'{sequence_id}.json'
            command = [manifest['python_executable'], str(run / 'bundle/folding_policy/final_worker.py'),
                       'evaluate', str(run), '--evaluation', str(task['task']), '--sequence', sequence_id,
                       '--output', str(output)]
            with (folder / 'worker.log').open('a') as log:
                subprocess.run(command, check=True, stdout=log, stderr=log,
                               timeout=min(task['seconds'] + 120, remaining - 5))
            row = json.loads(output.read_text())
            from inference_report import verify_row
            source = next(r for r in json.loads((run / 'data.json').read_text())[task['split']] if r['id'] == sequence_id)
            checked = verify_row(row, source, task)
            if not checked['complete_budget']:
                raise ValueError('Search returned before its budget without a timely optimality proof')
            payload['results'].append(row)
            compact_write(folder / 'evaluation.json', payload)
            print(json.dumps(dict(arm=task['arm'], seed=task['seed'], split=task['split'],
                                  completed=len(payload['results']), planned=len(task['sequence_ids']),
                                  sequence=sequence_id, contacts=row['best_contacts'])), flush=True)
        payload['status'] = 'complete'
    except Exception as exc:
        payload.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        compact_write(folder / 'evaluation.json', payload)
        launched = json.loads((folder / 'launch.json').read_text())
        launched.update(status=payload['status'], ended_utc=datetime.now(timezone.utc).isoformat())
        if payload.get('error'):
            launched['error'] = payload['error']
        write_json(folder / 'launch.json', launched)


def run_job(run, group, index):
    run = Path(run).resolve()
    manifest = verify(run)
    runtime(manifest)
    task = find_job(manifest, group, index)
    folder = job_directory(run, task)
    folder.mkdir(parents=True, exist_ok=False)
    record = dict(task=task, status='running', software=software(), hostname=socket.gethostname(),
                  slurm_job_id=os.environ.get('SLURM_JOB_ID'),
                  cpu_affinity=sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None,
                  started_utc=datetime.now(timezone.utc).isoformat())
    write_json(folder / 'launch.json', record)
    start = time.monotonic()
    try:
        total_search = sum(len(manifest['evaluation_tasks'][i]['sequence_ids']) * 600 for i in task['evaluation_tasks'])
        budget_available(manifest, task['training_seconds'] + total_search, reserve=300)
        endpoint = None
        if task['training_seconds']:
            from final_train import train_parent, train_endpoint
            config = json.loads((run / 'training_config.json').read_text())
            parent = (verified_training(run, manifest['tasks'][task['parent_task']], manifest)
                      if task['parent_task'] is not None else None)
            if task['job_kind'] == 'parent':
                train_parent(run / 'data.json', config, training_directory(run, task), task['seed'],
                             task['training_seconds'], manifest['threads'])
            else:
                train_endpoint(run / 'data.json', config, training_directory(run, task), task['seed'],
                               task['arm'], task['training_seconds'], manifest['threads'],
                               parent_checkpoint=parent['checkpoint'] if parent else None)
            endpoint = verified_training(run, task, manifest)
        for evaluation_id in task['evaluation_tasks']:
            evaluate_split(run, manifest, manifest['evaluation_tasks'][evaluation_id], endpoint)
        record['status'] = 'complete'
    except Exception as exc:
        record.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        record.update(elapsed_s=time.monotonic()-start, ended_utc=datetime.now(timezone.utc).isoformat())
        write_json(folder / 'launch.json', record)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['evaluate'])
    parser.add_argument('directory', type=Path)
    parser.add_argument('--evaluation', type=int, required=True)
    parser.add_argument('--sequence', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    evaluate_one(args.directory.resolve(), args.evaluation, args.sequence, args.output)


if __name__ == '__main__':
    main()
