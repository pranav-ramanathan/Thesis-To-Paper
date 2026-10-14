"""Freeze two learning-rate controls and longer inference of all saved endpoints."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys

from launch import sha, software, write_json

HERE = Path(__file__).resolve().parent
REUSED_ARMS = ('parent', 'patternboost', 'grpo', 'dr_grpo', 'reinforce', 'dpo')
CONTROL_ARMS = ('patternboost_low_lr', 'grpo_low_lr')
ARMS = REUSED_ARMS + CONTROL_ARMS
SEEDS = (0, 1, 2)
PROTOCOL = {
    'stage': 'validation-only-finetuning-followup', 'arms': list(ARMS),
    'seeds': list(SEEDS), 'threads': 4, 'memory_gib': 16,
    'model': {'width': 64, 'heads': 4, 'layers': 2, 'dropout': 0.0},
    'parent_training_seconds': 7200, 'additional_training_seconds': 1800,
    'control_learning_rate': 0.00003,
    'evaluation_seconds_per_sequence_per_seed': 600,
    'inference_timepoints_s': [60, 300, 600], 'attempt_checkpoints': [128, 512, 1024],
    'array': '0-23%24', 'slurm_minutes': 240,
    'checkpoint_selection': 'fixed_training_endpoints',
    'comparison': 'learning_rate_controls_and_equal_time_and_attempted_fold_prefixes',
    'final_test_enabled': False, 'automatic_extension': False,
}
NEW_FILES = ('finetune_followup_launch.py', 'finetune_followup_worker.py',
             'finetune_followup_evaluate.py', 'finetune_followup_report.py',
             'finetune_followup_protocol.json', 'finetune_followup.sbatch',
             'finetune_followup_pilot.sh', 'finetune_followup_report.sh')


def task_directory(run, task):
    return Path(run) / 'tasks' / f"{task['task']:03d}_{task['arm']}_seed{task['seed']}"


def tasks(endpoints):
    """Reused training is charged to lineages, never scheduled a second time."""
    result = []
    for arm in ARMS:
        algorithm = arm.removesuffix('_low_lr')
        control = arm in CONTROL_ARMS
        for seed in SEEDS:
            info = endpoints[(algorithm, seed)]
            result.append({
                'task': len(result), 'arm': arm, 'algorithm': algorithm, 'seed': seed,
                'parent': f'source_finetune/parents/patternboost_seed{seed}.pt',
                'parent_sha256': info['parent_sha256'],
                'parent_state_sha256': info['parent_state_sha256'],
                'parent_optimizer_sha256': info['parent_optimizer_sha256'],
                'checkpoint': None if control else 'source_finetune/' + info['checkpoint'],
                'checkpoint_sha256': None if control else info['file_sha256'],
                'checkpoint_state_sha256': None if control else info['state_sha256'],
                'source_task': None if control else info['source_task'],
                'source_task_directory': None if control else 'source_finetune/' + info['source_task_directory'],
                'training_receipt': None if control or algorithm == 'parent' else
                    'source_finetune/' + info['source_task_directory'] + '/training/result.json',
                'new_training_seconds': 1800 if control else 0,
                'training_seconds': 0 if algorithm == 'parent' else 1800,
                'lineage_training_seconds': 7200 if algorithm == 'parent' else 9000,
                'learning_rate': 0.00003 if control else info['learning_rate'],
                'evaluation_seconds': 600, 'slurm_seconds': 14400,
            })
    return result


def validate_protocol(config):
    if config != PROTOCOL:
        raise ValueError('Protocol differs from the bounded fine-tuning follow-up')


def validate_source(directory):
    """Validate the completed old pilot without writing into it or loading models."""
    from finetune_launch import verify as verify_finetune, task_directory as old_directory
    from finetune_report import validate_payload, validate_training
    from finetune_worker import training_receipt
    from inference_report import verify_row
    source = Path(directory).resolve()
    manifest = verify_finetune(source)
    data = json.loads((source / 'data.json').read_text())
    sequences = {row['id']: row for row in data['validation']}
    endpoints, paths = {}, {source / name for name in manifest['hashes']}
    paths.update((source / 'finetune.json', source / 'finetune.sha256'))
    for task in manifest['tasks']:
        folder = old_directory(source, task)
        launched = json.loads((folder / 'launch.json').read_text())
        payload = json.loads((folder / 'evaluation.json').read_text())
        if (launched.get('status') != 'complete' or payload.get('status') != 'complete'
                or launched.get('software') != manifest['software_at_freeze']
                or launched.get('task') != task):
            raise ValueError('Source fine-tuning task is incomplete or its runtime provenance changed')
        trained = {}
        checkpoint = source / task['parent']
        record = {'file_sha256': task['parent_sha256'], 'state_sha256': task['parent_state_sha256'],
                  'training_seconds': 7200}
        if task['arm'] != 'parent':
            trained, record = training_receipt(source, task, manifest)
            if validate_training(trained, manifest, task, folder) != record:
                raise ValueError('Source training validators disagree')
            checkpoint = folder / 'training/final.pt'
            paths.update((checkpoint, folder / 'training/result.json'))
        validate_payload(payload, manifest, task, record, trained)
        seen = set()
        for row in payload['results']:
            sequence_id = row.get('id')
            if sequence_id not in manifest['validation_ids'] or sequence_id in seen:
                raise ValueError('Source evaluation contains duplicate or unexpected validation sequences')
            seen.add(sequence_id)
            checked = verify_row(row, sequences[sequence_id],
                                 {'method': 'model', 'seed': task['seed'], 'seconds': 60})
            if not checked['complete_budget']:
                raise ValueError('Source evaluation did not cover its complete inference budget')
            expected_rate = (row['generated_complete_folds'] / row['attempted_folds']
                             if row['attempted_folds'] else None)
            if row.get('valid_fold_rate') != expected_rate:
                raise ValueError('Source evaluation valid-fold rate differs from its counters')
        if seen != set(manifest['validation_ids']):
            raise ValueError('Source evaluation did not complete every validation sequence')
        paths.update((folder / 'launch.json', folder / 'evaluation.json'))
        endpoints[(task['arm'], task['seed'])] = {
            **record, 'checkpoint': str(checkpoint.relative_to(source)),
            'source_task': task['task'], 'source_task_directory': str(folder.relative_to(source)),
            **{key: task[key] for key in ('parent_sha256', 'parent_state_sha256',
                                         'parent_optimizer_sha256', 'learning_rate')},
        }
    if set(endpoints) != {(arm, seed) for arm in REUSED_ARMS for seed in SEEDS}:
        raise ValueError('Source must contain all eighteen fixed endpoints')
    for path in paths:
        if not path.resolve().is_relative_to(source):
            raise ValueError('Source artifact escapes its campaign directory')
    return manifest, endpoints, sorted(paths)


def submission_command(run):
    run = Path(run)
    command = ['sbatch', '--parsable', f'--chdir={run}', '--job-name=hp-ft-followup',
               '--array=0-23%24', '--time=240', f'--output={run}/logs/%A_%a.out',
               f'--error={run}/logs/%A_%a.err']
    if os.environ.get('HP_MAIL_USER'):
        command.append('--mail-user=' + os.environ['HP_MAIL_USER'])
    return command + [str(run / 'bundle/folding_policy/finetune_followup.sbatch'), str(run)]


def submit_array(run, dry_run=False):
    """Never silently retry an accepted or ambiguous scheduler submission."""
    run = Path(run)
    path = run / 'submission.json'
    if path.exists():
        raise ValueError('Submission was already attempted; do not resubmit this directory')
    command = submission_command(run)
    record = {'status': 'prepared', 'dry_run': dry_run, 'command': command,
              'submitted': False, 'automatic_extension': False}
    write_json(path, record)
    if dry_run:
        print('Prepared only: ' + shlex.join(command))
        return record
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        response = result.stdout.strip()
        record.update(sbatch_response=response, sbatch_stderr=result.stderr)
        match = re.fullmatch(r'(\d+)(?:;[^;\s]+)?', response)
        if not match:
            record['status'] = 'submission_uncertain'
            raise ValueError(f'Unexpected sbatch response: {response!r}')
        record.update(status='submitted', submitted=True, job_id=match.group(1))
        write_json(run / 'job_ids.json', {'finetune_followup': record['job_id']})
        temporary = run / 'job_ids.txt.tmp'
        temporary.write_text(record['job_id'] + '\n')
        temporary.replace(run / 'job_ids.txt')
        print(f"Submitted fine-tuning follow-up array {record['job_id']} (24 tasks).")
    except Exception as exc:
        record['error'] = str(exc)
        if isinstance(exc, subprocess.CalledProcessError):
            record.update(status='failed', sbatch_output=exc.output, sbatch_stderr=exc.stderr)
        elif record['status'] == 'prepared':
            record['status'] = 'submission_uncertain'
        write_json(path, record)
        raise RuntimeError(f'Submission did not finish cleanly. Inspect {path}; check Slurm '
                           'before any new submission. This directory cannot be resubmitted.') from exc
    write_json(path, record)
    return record


def prepare(args):
    run = Path(args.directory).expanduser().resolve()
    source = Path(args.source_finetune).expanduser().resolve()
    if run.exists():
        raise ValueError(f'Refusing to overwrite existing run: {run}')
    if run.is_relative_to(source):
        raise ValueError('Follow-up output must be outside the completed source pilot')
    missing = [name for name in NEW_FILES if not (HERE / name).is_file()]
    if missing:
        raise ValueError(f'Incomplete follow-up implementation: {missing}')
    validate_protocol(json.loads((HERE / 'finetune_followup_protocol.json').read_text()))
    source_manifest, endpoints, paths = validate_source(source)
    versions = software()
    if versions != source_manifest['software_at_freeze']:
        raise ValueError('Python or package versions differ from source model training')
    python = Path(args.python).expanduser().absolute()
    if not python.is_file():
        raise ValueError(f'Python executable missing: {python}')
    if not args.dry_run and shutil.which('sbatch') is None:
        raise ValueError('sbatch is unavailable; submit on Apocrita or use --dry-run')
    if not args.dry_run and any(versions['packages'][name] is None for name in ('torch', 'numpy', 'ortools')):
        raise ValueError('The cluster virtual environment requires Torch, NumPy and OR-Tools')
    run.mkdir(parents=True)
    for name in ('logs', 'tasks', 'source_finetune'):
        (run / name).mkdir()
    # Preserve an independently verifiable historical campaign, including the entire
    # original frozen source graph, but omit mutable logs and unrelated run outputs.
    for path in paths:
        target = run / 'source_finetune' / path.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    shutil.copytree(source / 'bundle', run / 'bundle')
    for name in NEW_FILES:
        shutil.copy2(HERE / name, run / 'bundle/folding_policy' / name)
    shutil.copy2(source / 'data.json', run / 'data.json')
    second_manifest, second_endpoints, second_paths = validate_source(source)
    if (second_manifest != source_manifest or second_endpoints != endpoints or second_paths != paths
            or any(sha(path) != sha(run / 'source_finetune' / path.relative_to(source)) for path in paths)):
        raise ValueError('Source pilot changed during follow-up preparation')
    git = subprocess.run(['git', '-C', str(HERE), 'rev-parse', 'HEAD'], capture_output=True, text=True)
    files = sorted(path for name in ('bundle', 'source_finetune')
                   for path in (run / name).rglob('*') if path.is_file()) + [run / 'data.json']
    manifest = {
        'schema': 1, 'kind': 'validation_finetune_followup',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'source_revision': git.stdout.strip() if git.returncode == 0 else None,
        'source_finetune': str(source), 'source_finetune_sha256': sha(source / 'finetune.json'),
        'source_training_revision': source_manifest.get('source_revision'),
        'python_executable': str(python), 'software_at_freeze': versions,
        'hashes': {str(path.relative_to(run)): sha(path) for path in files},
        'threads': 4, 'memory_gib': 16, 'seeds': list(SEEDS), 'arms': list(ARMS),
        'tasks': tasks(endpoints), 'validation_ids': source_manifest['validation_ids'],
        'parent_training_seconds': 7200, 'additional_training_seconds': 1800,
        'evaluation_seconds_per_sequence_per_seed': 600,
        'inference_timepoints_s': [60, 300, 600], 'attempt_checkpoints': [128, 512, 1024],
        'planned_new_training_cpu_hours': 12.0, 'planned_evaluation_cpu_hours': 256.0,
        'planned_new_cpu_hours': 268.0, 'reserved_cpu_hours_ceiling': 384.0,
        'historical_parent_training_repeated': False,
        'checkpoint_selection': 'fixed_training_endpoints',
        'comparison': PROTOCOL['comparison'], 'final_test_access': False, 'no_automatic_followup': True,
    }
    write_json(run / 'finetune_followup.json', manifest)
    (run / 'finetune_followup.sha256').write_text(sha(run / 'finetune_followup.json') + '\n')
    verify(run)
    print(json.dumps({'directory': str(run), 'tasks': 24, 'new_training_tasks': 6,
                      'validation_sequences': 16, 'planned_new_cpu_hours': 268.0,
                      'reserved_cpu_hours_ceiling': 384.0, 'final_test_access': False}))
    submit_array(run, args.dry_run)
    return manifest


def verify(directory):
    from allocation_launch import validation_ids
    run = Path(directory).resolve()
    if (run / 'finetune_followup.sha256').read_text().strip() != sha(run / 'finetune_followup.json'):
        raise ValueError('Frozen follow-up manifest changed')
    manifest = json.loads((run / 'finetune_followup.json').read_text())
    expected = {'schema': 1, 'kind': 'validation_finetune_followup', 'threads': 4, 'memory_gib': 16,
                'seeds': list(SEEDS), 'arms': list(ARMS), 'parent_training_seconds': 7200,
                'additional_training_seconds': 1800, 'evaluation_seconds_per_sequence_per_seed': 600,
                'inference_timepoints_s': [60, 300, 600], 'attempt_checkpoints': [128, 512, 1024],
                'planned_new_training_cpu_hours': 12.0, 'planned_evaluation_cpu_hours': 256.0,
                'planned_new_cpu_hours': 268.0, 'reserved_cpu_hours_ceiling': 384.0,
                'historical_parent_training_repeated': False,
                'checkpoint_selection': 'fixed_training_endpoints', 'comparison': PROTOCOL['comparison'],
                'final_test_access': False, 'no_automatic_followup': True}
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise ValueError('Follow-up plan differs from its frozen validation-only budgets')
    for relative, digest in manifest['hashes'].items():
        path = (run / relative).resolve()
        if not path.is_relative_to(run) or not path.is_file() or sha(path) != digest:
            raise ValueError(f'Frozen artifact changed: {relative}')
    for name in NEW_FILES:
        if f'bundle/folding_policy/{name}' not in manifest['hashes']:
            raise ValueError(f'Frozen implementation digest is missing: {name}')
    validate_protocol(json.loads((run / 'bundle/folding_policy/finetune_followup_protocol.json').read_text()))
    source_manifest, endpoints, source_paths = validate_source(run / 'source_finetune')
    if sha(run / 'source_finetune/finetune.json') != manifest['source_finetune_sha256']:
        raise ValueError('Source fine-tuning manifest differs from its frozen provenance')
    if any(manifest['hashes'].get(str(path.relative_to(run))) != sha(path) for path in source_paths):
        raise ValueError('Historical source provenance is missing from frozen digests')
    if (source_manifest['hashes']['data.json'] != manifest['hashes'].get('data.json')
            or source_manifest['software_at_freeze'] != manifest['software_at_freeze']):
        raise ValueError('Dataset or software differs from source training')
    ids = validation_ids(json.loads((run / 'data.json').read_text()))
    if manifest.get('validation_ids') != ids or source_manifest['validation_ids'] != ids:
        raise ValueError('Frozen validation selection changed')
    overlay = {f'bundle/folding_policy/{name}' for name in NEW_FILES}
    for relative, digest in source_manifest['hashes'].items():
        if relative.startswith('bundle/') and relative not in overlay and manifest['hashes'].get(relative) != digest:
            raise ValueError(f'Source model or training implementation changed: {relative}')
    if manifest.get('tasks') != tasks(endpoints):
        raise ValueError('Frozen follow-up task plan or endpoint provenance changed')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare')
    p.add_argument('directory')
    p.add_argument('--source-finetune', required=True)
    p.add_argument('--python', default=sys.executable)
    p.add_argument('--dry-run', action='store_true')
    p = sub.add_parser('run')
    p.add_argument('directory')
    p.add_argument('--task', type=int, required=True)
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare(args)
    else:
        from finetune_followup_worker import execute
        execute(args)


if __name__ == '__main__':
    main()
