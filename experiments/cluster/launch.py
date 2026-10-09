"""Deadline guardian for one array element, including native solver calls."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import time
from campaign import file_hash, save, verify

STOP = False


def request_stop(signum, frame):
    global STOP
    STOP = True


def supervise(command, *, soft_end, hard_end, log, stopped=lambda:False):
    """Bound this child process group. Graceful stop, then compulsory termination."""
    child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    reason, sent_at, killed = None, None, False
    while child.poll() is None:
        now = time.time()
        if sent_at is None and (stopped() or now >= soft_end):
            reason = 'scheduler_signal' if stopped() else 'application_deadline'
            try: os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError: pass
            sent_at = now
        if now >= hard_end or (sent_at is not None and now-sent_at >= 180):
            try: os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError: pass
            killed = True
            break
        time.sleep(min(.5, max(.01,hard_end-now)))
    return dict(exit_code=child.wait(), stop_reason=reason, forced_kill=killed)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--campaign',type=Path,required=True); p.add_argument('--task',type=int,required=True)
    p.add_argument('--resume',action='store_true'); p.add_argument('--smoke',action='store_true')
    args=p.parse_args(); campaign=args.campaign.resolve(); manifest=verify(campaign)
    if not 0 <= args.task < len(manifest['tasks']): raise ValueError('Task index outside manifest')
    task=manifest['tasks'][args.task]; out=campaign/'tasks'/f'{args.task:03d}_{task["arm"]}_{task["seq_id"]}_seed{task["seed"]}'
    out.mkdir(exist_ok=args.resume)
    with (out/'active.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        previous=json.loads((out/'launch.json').read_text()) if args.resume else {}
        base_elapsed=previous.get('total_elapsed_s',0)
        if args.resume:
            if task['arm'] in ('rl','rl_cp_sat') and not (out/'checkpoint.pt').is_file():
                raise ValueError('No saved RL checkpoint to resume')
            if task['arm'] == 'decisionboost':
                raise ValueError('DecisionBoost partial runs are preserved, not automatically restarted')
        started=time.time()
        soft_end=min(manifest['deadline_epoch']-300, started+manifest['run_seconds']-base_elapsed)
        hard_end=min(manifest['deadline_epoch']-30, soft_end+180)
        if args.smoke:
            smoke_seconds=180 if task['arm']=='decisionboost' else 8
            soft_end=min(soft_end,started+smoke_seconds); hard_end=min(hard_end,soft_end+22)
        info=dict(task=task,task_index=args.task,pid=os.getpid(),hostname=platform.node(),
                  platform=platform.platform(),python=sys.version,device='cpu',threads=manifest['threads'],
                  job_id=os.environ.get('SLURM_JOB_ID'),allocated_cpus=os.environ.get('SLURM_CPUS_PER_TASK'),
                  start_epoch=started,compute_end_epoch=soft_end,hard_end_epoch=hard_end,
                  campaign_deadline_utc=manifest['deadline_utc'],base_elapsed_s=base_elapsed,
                  resume=args.resume,engineering_only=args.smoke,status='running',previous_attempt=previous or None)
        if Path('/proc/cpuinfo').exists():
            info['cpu_model']=next((line.split(':',1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines()
                                    if line.startswith('model name')),None)
        save(out/'launch.json',info)
        if soft_end <= started:
            info.update(status='skipped_deadline',total_elapsed_s=base_elapsed)
            save(out/'launch.json',info); return
        if os.environ.get('SLURM_CPUS_PER_TASK') and int(os.environ['SLURM_CPUS_PER_TASK']) != manifest['threads']:
            raise ValueError('Allocated CPU count differs from frozen campaign')
        if manifest.get('expected_cpu_model') and info.get('cpu_model')!=manifest['expected_cpu_model']:
            raise ValueError('CPU model differs from the reused comparison results')
        if task['arm']=='decisionboost':
            for name,digest in manifest['model_files'].items():
                if file_hash(Path(manifest['model_dir'])/name)!=digest:
                    raise ValueError(f'Pinned encoder changed: {name}')
        for sig in (signal.SIGTERM,signal.SIGUSR1,signal.SIGINT): signal.signal(sig,request_stop)
        for variable in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
            os.environ[variable]=str(manifest['threads'])
        os.environ['MPLBACKEND']='Agg'; os.environ['PYTHONDONTWRITEBYTECODE']='1'
        if manifest.get('model_dir'): os.environ['HP_ENCODER_DIR']=manifest['model_dir']
        command=[sys.executable,str(campaign/'bundle/worker.py'),'--out',str(out),
                 '--arm',task['arm'],'--seq-id',task['seq_id'],'--seed',str(task['seed']),
                 '--threads',str(manifest['threads']),'--end-epoch',str(soft_end),
                 '--base-elapsed',str(base_elapsed)]
        if args.resume: command.append('--resume')
        if args.smoke: command.append('--smoke')
        with (out/'worker.log').open('a') as log:
            result=supervise(command,soft_end=soft_end,hard_end=hard_end,log=log,stopped=lambda:STOP)
        elapsed=time.time()-started
        info.update(result,end_epoch=time.time(),total_elapsed_s=base_elapsed+elapsed,
                    attempt_elapsed_s=elapsed,status='finished' if result['exit_code']==0 else 'partial_or_failed',
                    deadline_truncated=soft_end < started+manifest['run_seconds']-base_elapsed-1,
                    ended_utc=datetime.now(timezone.utc).isoformat())
        save(out/'launch.json',info)
        # Deliberate deadline termination is saved as partial; real failures remain nonzero.
        if result['exit_code'] and not result['stop_reason']: sys.exit(1)


if __name__=='__main__': main()
