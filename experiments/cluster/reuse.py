"""Verify completed primary results for reuse without changing their artifacts."""
from __future__ import annotations
import json
import math
from pathlib import Path
from campaign import file_hash, verify
from geometry import contacts

# Reporting, submission and documentation may change. Scientific sources may not.
SCIENTIFIC_FILES = ('worker.py', 'hp_solver.py', 'geometry.py',
                    'environment.sh', 'vendor/rl_reference.py')
SCIENTIFIC_VERSIONS = {'torch':'2.9.0','numpy':'2.3.4','ortools':'9.15.6755'}


def task_directory(directory, index, task):
    return Path(directory)/'tasks'/f'{index:03d}_{task["arm"]}_{task["seq_id"]}_seed{task["seed"]}'


def collect(directory, *, protocol, run_seconds, threads, methods, code_directory):
    directory=Path(directory).resolve()
    manifest=verify(directory)
    if manifest.get('reused_tasks'):
        raise ValueError('Use the original completed campaign, not an already combined continuation')
    if manifest['mode']!='campaign' or manifest['run_seconds']!=run_seconds or manifest['threads']!=threads:
        raise ValueError('Reuse requires the same scientific run budget, CPU count and campaign mode')
    original=json.loads((directory/'bundle/protocol.json').read_text())
    global_settings=lambda p:{k:v for k,v in p.items() if k not in ('configurations','seeds')}
    if global_settings(original)!=global_settings(protocol):
        raise ValueError('Reuse protocol differs: keep the original training, evaluation and repair settings')
    for name in SCIENTIFIC_FILES:
        if file_hash(directory/'bundle'/name)!=file_hash(Path(code_directory)/name):
            raise ValueError(f'Reuse scientific source changed: {name}; use a separate comparison')
    imported=[]; seen=set()
    for index,task in enumerate(manifest['tasks']):
        if task['arm'] not in methods: continue
        if task['seq_id'] not in protocol['configurations'] or task['seed'] not in protocol['seeds']:
            raise ValueError('Reuse task is outside the requested sequence/seed matrix')
        if original['configurations'][task['seq_id']]!=protocol['configurations'][task['seq_id']]:
            raise ValueError(f'Reuse configuration changed: {task["seq_id"]}')
        key=(task['arm'],task['seed'],task['seq_id'])
        if key in seen: raise ValueError('Duplicate task in reuse campaign')
        seen.add(key)
        folder=task_directory(directory,index,task)
        artifacts={name:file_hash(folder/name) for name in ('launch.json','result.json','events.jsonl')}
        launch=json.loads((folder/'launch.json').read_text())
        result=json.loads((folder/'result.json').read_text())
        if (launch.get('status')!='finished' or launch.get('exit_code')!=0
                or launch.get('forced_kill') or launch.get('engineering_only')
                or result.get('engineering_only') or result.get('status') not in ('finished','stopped')):
            raise ValueError(f'Reuse task {index} is failed, unfinished or engineering-only')
        if launch.get('task')!=task or any(result.get(k)!=task[k] for k in ('arm','seed','seq_id')):
            raise ValueError(f'Reuse task {index} identity mismatch')
        if launch.get('device')!='cpu' or launch.get('threads')!=threads:
            raise ValueError(f'Reuse task {index} device/threads mismatch')
        if int(launch.get('allocated_cpus',0))!=threads or not launch.get('cpu_model'):
            raise ValueError(f'Reuse task {index} lacks matching CPU allocation/model metadata')
        for package,wanted in SCIENTIFIC_VERSIONS.items():
            if (result.get('versions',{}).get(package) or '').split('+')[0]!=wanted:
                raise ValueError(f'Reuse task {index} has incompatible {package} version')
        elapsed=launch.get('total_elapsed_s',0)
        if not isinstance(elapsed,(int,float)) or not math.isfinite(elapsed):
            raise ValueError('Invalid reuse elapsed time')
        witnessed=False; full_proof=False
        sequence=protocol['configurations'][task['seq_id']]['sequence']
        with (folder/'events.jsonl').open() as handle:
            for line in handle:
                event=json.loads(line)
                if event.get('type')=='witness':
                    timestamp=event['elapsed_s']
                    if not math.isfinite(timestamp) or timestamp<0 or event['seq']!=sequence:
                        raise ValueError('Reuse witness has wrong sequence or invalid time')
                    if contacts(sequence,event['positions'],len(sequence)//2)!=event['contacts']:
                        raise ValueError('Reuse witness contact mismatch')
                    witnessed |= timestamp<=run_seconds
                if (event.get('type')=='solver_end' and event.get('scope')=='full_declared_cube'
                        and event.get('status')=='OPTIMAL' and event['elapsed_s']<=run_seconds):
                    full_proof=True
        optimal=(task['arm']=='cp_sat' and result.get('full_cube_optimal') and full_proof)
        if not witnessed or (elapsed<run_seconds-1 and not optimal):
            raise ValueError(f'Reuse task {index} lacks a valid witnessed full-budget outcome')
        if {name:file_hash(folder/name) for name in artifacts}!=artifacts:
            raise ValueError('Reuse artifacts changed during validation')
        imported.append(dict(directory=str(directory),manifest_sha256=file_hash(directory/'campaign.json'),
                             source_task=index,task=task,artifacts=artifacts,
                             cpu_model=launch.get('cpu_model')))
    if not imported: raise ValueError('No compatible completed primary tasks to reuse')
    if len({item['cpu_model'] for item in imported})!=1:
        raise ValueError('Reused results have heterogeneous CPU models')
    return imported,manifest['deadline_epoch']


def verify_imports(manifest):
    """Fail before aggregation if a referenced frozen bundle or raw result changed."""
    checked=set()
    for item in manifest.get('reused_tasks',[]):
        directory=Path(item['directory'])
        if str(directory) not in checked:
            verify(directory);checked.add(str(directory))
        if file_hash(directory/'campaign.json')!=item['manifest_sha256']:
            raise RuntimeError('Reused campaign manifest changed')
        folder=task_directory(directory,item['source_task'],item['task'])
        if any(file_hash(folder/name)!=digest for name,digest in item['artifacts'].items()):
            raise RuntimeError('Reused raw results changed; refusing combined analysis')
