"""Recover the declared 13 missing CP-SAT seed-4 cells without changing the source run."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import socket
import subprocess
import sys
import time


FAILURE = 'ValueError: Search returned before its budget without a timely optimality proof'
RULE = 'Use every replacement even if worse; never pool with the excluded original attempt'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def imports(source):
    # The copied runner uses the campaign's original verification and runtime code.
    sys.path.insert(0, str(Path(source) / 'bundle/folding_policy'))


def source_state(source):
    imports(source)
    from final_launch import job_directory, task_directory, verify
    from inference_report import verify_row
    manifest = verify(source)
    task, benchmark = manifest['evaluation_tasks'][88:90]
    if (task['arm'], task['method'], task['seed'], task['split'], task['seconds'],
            benchmark['arm'], benchmark['method'], benchmark['seed'], benchmark['split'], benchmark['seconds']) != (
            'cp_sat', 'cp_sat', 4, 'test', 600, 'cp_sat', 'cp_sat', 4, 'benchmark', 600):
        raise ValueError('Recovery requires the original CP-SAT seed-4 tasks')
    if len(task['sequence_ids']) != 80 or len(benchmark['sequence_ids']) != 8:
        raise ValueError('Recovery requires the original 80/8 sequence panels')
    folder = task_directory(source, task)
    paths = [folder / 'evaluation.json', folder / 'launch.json',
             folder / 'sequences' / (task['sequence_ids'][75] + '.json'),
             job_directory(source, manifest['tasks'][task['job_task']]) / 'launch.json']
    payload, launched, short, job = map(read, paths)
    if payload.get('status') != 'failed' or payload.get('error') != FAILURE:
        raise ValueError('Source evaluation is not the known short-budget failure')
    expected = dict(schema=1, kind='final_policy_evaluation', split='test', final_test_access=True,
                    selection='fixed_training_endpoint', arm='cp_sat', method='cp_sat', seed=4,
                    threads=4, ids=task['sequence_ids'], seconds_per_sequence=600,
                    dataset_file_sha256=manifest['hashes']['data.json'], model_setup_charged_per_sequence=True,
                    checkpoint_sha256=None, checkpoint_metadata=None)
    if any(payload.get(k) != v for k, v in expected.items()):
        raise ValueError('Source evaluation provenance differs from its frozen task')
    if (launched.get('task') != task or launched.get('status') != 'failed'
            or launched.get('software') != manifest['software_at_freeze']
            or job.get('task') != manifest['tasks'][task['job_task']] or job.get('status') != 'failed'
            or job.get('software') != manifest['software_at_freeze']):
        raise ValueError('Source failure launch provenance changed')
    if task_directory(source, benchmark).exists():
        raise ValueError('The source benchmark task must remain unstarted')
    if [r['id'] for r in payload['results']] != task['sequence_ids'][:75]:
        raise ValueError('Source must contain exactly the first 75 accepted sequence rows')
    data = read(source / 'data.json')
    sources = {row['id']: row for row in data['test']}
    for row in payload['results']:
        if not verify_row(row, sources[row['id']], task)['complete_budget']:
            raise ValueError('Source accepted row lacks its original full budget or timely proof')
    checked = verify_row(short, sources[task['sequence_ids'][75]], task)
    if (short['id'] != task['sequence_ids'][75] or checked['complete_budget']
            or short.get('solver_status') != 'FEASIBLE' or not 599 <= short['elapsed_s'] < 600):
        raise ValueError('Original orphan is not the declared sub-second short CP-SAT attempt')
    return manifest, payload, launched, short, paths


def expected_cells(manifest):
    cells = []
    for evaluation_id, skip in ((88, 75), (89, 0)):
        task = manifest['evaluation_tasks'][evaluation_id]
        for sequence_id in task['sequence_ids'][skip:]:
            cells.append(dict(task=len(cells), evaluation_task=evaluation_id,
                              sequence_id=sequence_id, split=task['split']))
    return cells


def deadline_available(plan):
    remaining = (datetime.fromisoformat(plan['deadline_utc']) - datetime.now(timezone.utc)).total_seconds()
    if remaining < 660:
        raise ValueError('Original campaign deadline cannot accommodate a full recovery cell')
    return remaining


def prepare(source_run, recovery_dir):
    source, recovery = Path(source_run).resolve(), Path(recovery_dir).resolve()
    if recovery.exists() or recovery.is_relative_to(source) or source.is_relative_to(recovery):
        raise ValueError('Recovery must be a new separate directory outside the original campaign')
    manifest, _, _, _, paths = source_state(source)
    deadline_available(manifest)
    plan = dict(schema=1, kind='final_cp_seed4_recovery', source_run=str(source),
                source_campaign_sha256=sha(source / 'campaign.json'),
                source_hashes={str(p.relative_to(source)): sha(p) for p in paths},
                deadline_utc=manifest['deadline_utc'], python_executable=manifest['python_executable'],
                created_utc=datetime.now(timezone.utc).isoformat(), threads=4, memory_gib=16,
                seconds_per_cell=600, slurm_minutes=20, cells=expected_cells(manifest),
                replacement_rule=RULE, automatic_retry=False,
                planned_recovery_allocated_cpu_hours=13 * 600 * 4 / 3600,
                reserved_cpu_hours_ceiling=13 * 1200 * 4 / 3600, hashes={})
    recovery.mkdir(parents=True)
    (recovery / 'bundle').mkdir()
    (recovery / 'logs').mkdir()
    for name in ('final_recovery.py', 'final_recovery.sbatch'):
        target = recovery / 'bundle' / name
        shutil.copy2(Path(__file__).parent / name, target)
        plan['hashes'][str(target.relative_to(recovery))] = sha(target)
    write(recovery / 'recovery.json', plan)
    (recovery / 'recovery.sha256').write_text(sha(recovery / 'recovery.json') + '\n')
    return plan


def verify_recovery_plan(recovery_dir):
    recovery = Path(recovery_dir).resolve()
    if (recovery / 'recovery.sha256').read_text().strip() != sha(recovery / 'recovery.json'):
        raise ValueError('Frozen recovery plan changed')
    plan = read(recovery / 'recovery.json')
    source = Path(plan['source_run']).resolve()
    if recovery.is_relative_to(source) or source.is_relative_to(recovery):
        raise ValueError('Recovery and source directories must be separate')
    manifest, _, _, _, paths = source_state(source)
    required = dict(schema=1, kind='final_cp_seed4_recovery',
                    source_campaign_sha256=sha(source / 'campaign.json'),
                    deadline_utc=manifest['deadline_utc'], python_executable=manifest['python_executable'],
                    threads=4, memory_gib=16, seconds_per_cell=600, slurm_minutes=20,
                    cells=expected_cells(manifest), replacement_rule=RULE, automatic_retry=False,
                    planned_recovery_allocated_cpu_hours=13 * 600 * 4 / 3600,
                    reserved_cpu_hours_ceiling=13 * 1200 * 4 / 3600,
                    source_hashes={str(p.relative_to(source)): sha(p) for p in paths})
    if any(plan.get(k) != v for k, v in required.items()):
        raise ValueError('Recovery provenance, source artifacts or configuration changed')
    if set(plan['hashes']) != {'bundle/final_recovery.py', 'bundle/final_recovery.sbatch'}:
        raise ValueError('Missing frozen recovery implementation')
    for relative, expected in plan['hashes'].items():
        path = (recovery / relative).resolve()
        if not path.is_relative_to(recovery) or sha(path) != expected:
            raise ValueError('Frozen recovery implementation changed: ' + relative)
    return plan


def submit(recovery_dir, dry_run=False):
    recovery = Path(recovery_dir).resolve()
    plan = verify_recovery_plan(recovery)
    deadline_available(plan)
    command = ['sbatch', '--parsable', f'--chdir={recovery}', '--array=0-12%13', '--time=20',
               f'--output={recovery}/logs/%A_%a.out', f'--error={recovery}/logs/%A_%a.err',
               str(recovery / 'bundle/final_recovery.sbatch'), str(recovery)]
    if dry_run:
        print(shlex.join(command))
        return dict(status='dry_run', command=command)
    path = recovery / 'submission.json'
    if path.exists():
        raise ValueError('Recovery submission was already attempted; inspect saved job IDs')
    state = dict(status='submitting', command=command, automatic_retry=False)
    write(path, state)
    try:
        result = subprocess.run(command, text=True, capture_output=True, check=True)
        state.update(stdout=result.stdout, stderr=result.stderr)
        match = re.fullmatch(r'(\d+)(?:;[^;\s]+)?', result.stdout.strip())
        if not match:
            raise ValueError('Unexpected sbatch response; inspect Slurm before further submission')
        state.update(status='submitted', job_id=match.group(1))
        (recovery / 'job_ids.txt').write_text(match.group(1) + '\n')
    except Exception as exc:
        state.update(status='submission_uncertain', error=str(exc))
        raise
    finally:
        write(path, state)
    print(json.dumps(state), flush=True)
    return state


def run_cell(recovery_dir, index):
    recovery = Path(recovery_dir).resolve()
    plan = verify_recovery_plan(recovery)
    if type(index) is not int or not 0 <= index < len(plan['cells']):
        raise ValueError('Recovery cell is outside the frozen array')
    source = Path(plan['source_run'])
    from final_launch import verify
    from final_worker import runtime
    from launch import software
    from inference_report import verify_row
    manifest = verify(source)
    runtime(manifest)
    remaining = deadline_available(plan)
    cell = plan['cells'][index]
    folder = recovery / 'cells' / f'{index:03d}'
    folder.mkdir(parents=True, exist_ok=False)
    record = dict(cell=cell, status='running', software=software(), hostname=socket.gethostname(),
                  slurm_job_id=os.environ.get('SLURM_JOB_ID'),
                  started_utc=datetime.now(timezone.utc).isoformat())
    write(folder / 'launch.json', record)
    start = time.monotonic()
    try:
        output = folder / 'result.json'
        command = [plan['python_executable'], str(source / 'bundle/folding_policy/final_worker.py'),
                   'evaluate', str(source), '--evaluation', str(cell['evaluation_task']),
                   '--sequence', cell['sequence_id'], '--output', str(output)]
        with (folder / 'worker.log').open('w') as log:
            subprocess.run(command, stdout=log, stderr=log, check=True, timeout=min(720, remaining - 5))
        row = read(output)
        task = manifest['evaluation_tasks'][cell['evaluation_task']]
        sequence = next(r for r in read(source / 'data.json')[cell['split']] if r['id'] == cell['sequence_id'])
        if row.get('status') != 'complete' or not verify_row(row, sequence, task)['complete_budget']:
            raise ValueError('Recovery search lacks full original budget or timely proof; no automatic retry')
        record.update(status='complete', result_sha256=sha(output))
    except Exception as exc:
        record.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        record.update(elapsed_s=time.monotonic() - start, ended_utc=datetime.now(timezone.utc).isoformat())
        write(folder / 'launch.json', record)


def load_recovery_overlay(source_run, recovery_dir, manifest=None):
    recovery = Path(recovery_dir).resolve()
    plan = verify_recovery_plan(recovery)
    source = Path(source_run).resolve()
    if str(source) != plan['source_run']:
        raise ValueError('Recovery belongs to a different source campaign')
    verified_manifest, original, original_launch, short, _ = source_state(source)
    if manifest is not None and manifest != verified_manifest:
        raise ValueError('Recovery source manifest differs from the report')
    manifest = verified_manifest
    from inference_report import verify_row
    data = read(source / 'data.json')
    results, receipts, seconds = {88: list(original['results']), 89: []}, [], 0
    for cell in plan['cells']:
        folder = recovery / 'cells' / f"{cell['task']:03d}"
        record, row = read(folder / 'launch.json'), read(folder / 'result.json')
        if (record.get('cell') != cell or record.get('status') != 'complete'
                or record.get('software') != manifest['software_at_freeze']
                or record.get('result_sha256') != sha(folder / 'result.json')):
            raise ValueError('Recovery cell lacks a verified complete launch or result hash')
        task = manifest['evaluation_tasks'][cell['evaluation_task']]
        sequence = next(r for r in data[cell['split']] if r['id'] == cell['sequence_id'])
        if row.get('status') != 'complete' or not verify_row(row, sequence, task)['complete_budget']:
            raise ValueError('Recovery cell lacks original full-budget coverage or a timely proof')
        results[cell['evaluation_task']].append(row)
        seconds += row['elapsed_s']
        receipts.append(dict(cell=cell, result_sha256=record['result_sha256'],
                             launch_sha256=sha(folder / 'launch.json')))
    payloads, launches = {}, {}
    for evaluation_id, rows in results.items():
        task = manifest['evaluation_tasks'][evaluation_id]
        if [r['id'] for r in rows] != task['sequence_ids']:
            raise ValueError('Recovery does not cover the exact original evaluation panel')
        payload = {k: v for k, v in original.items() if k not in ('error', 'results')}
        payload.update(split=task['split'], ids=task['sequence_ids'], status='complete', results=rows)
        payloads[evaluation_id] = payload
        launches[evaluation_id] = dict(task=task, status='complete', software=manifest['software_at_freeze'],
                                      recovery_directory=str(recovery), original_status=(
                                          original_launch['status'] if evaluation_id == 88 else 'not_started'))
    provenance = dict(recovery_directory=str(recovery), recovery_plan_sha256=sha(recovery / 'recovery.json'),
                      source_campaign_sha256=plan['source_campaign_sha256'], source_hashes=plan['source_hashes'],
                      reused_sequence_count=75, fresh_sequence_count=13,
                      replaced_sequence_ids=[manifest['evaluation_tasks'][88]['sequence_ids'][75]],
                      replacement_rule=RULE, original_failure=FAILURE,
                      original_short_elapsed_s=short['elapsed_s'],
                      planned_recovery_allocated_cpu_hours=plan['planned_recovery_allocated_cpu_hours'],
                      verified_recovery_allocated_cpu_hours=seconds * 4 / 3600,
                      excluded_original_attempt_allocated_cpu_hours=short['elapsed_s'] * 4 / 3600,
                      reserved_cpu_hours_ceiling=plan['reserved_cpu_hours_ceiling'], cells=receipts)
    return dict(payloads=payloads, launches=launches, provenance=provenance,
                output_directory=recovery / 'report')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    prepare_parser = sub.add_parser('prepare')
    prepare_parser.add_argument('source', type=Path)
    prepare_parser.add_argument('directory', type=Path)
    prepare_parser.add_argument('--dry-run', action='store_true')
    run_parser = sub.add_parser('run')
    run_parser.add_argument('directory', type=Path)
    run_parser.add_argument('--index', type=int, required=True)
    args = parser.parse_args()
    if args.command == 'prepare':
        plan = prepare(args.source, args.directory)
        print(json.dumps(dict(directory=str(args.directory.resolve()), cells=len(plan['cells']),
                              deadline_utc=plan['deadline_utc'],
                              planned_cpu_hours=plan['planned_recovery_allocated_cpu_hours'])), flush=True)
        submit(args.directory, args.dry_run)
    else:
        run_cell(args.directory, args.index)


if __name__ == '__main__':
    main()
