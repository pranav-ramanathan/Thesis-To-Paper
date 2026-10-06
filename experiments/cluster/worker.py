"""CPU-only, bounded HP experiments. Run through launch.py on compute nodes."""
from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import version, PackageNotFoundError
import json
import math
import os
from pathlib import Path
import random
import signal
import statistics
import sys
import time

from campaign import save
from geometry import contacts

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE/'vendor'))
STOP = False


def stopping(signum, frame):
    global STOP
    STOP = True


class Recorder:
    def __init__(self, out, seq, base_elapsed=0):
        self.out, self.seq, self.base_elapsed = out, seq, base_elapsed
        self.start = time.monotonic()
        self.best = None
        self.best_contacts = None
        path = out/'events.jsonl'
        # A killed write may leave one incomplete trailing record. Preserve its bytes.
        if path.exists():
            with path.open('rb+') as handle:
                good_end = 0
                for line in handle:
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        remainder = line + handle.read()
                        if b'\n' in line or remainder != line:
                            raise ValueError('Corrupt committed JSONL record; refusing to truncate history')
                        (out/'interrupted_tail.bin').write_bytes(remainder)
                        handle.seek(good_end); handle.truncate()
                        break
                    good_end = handle.tell()
                    if row.get('type') == 'witness' and row.get('positions') is not None:
                        self.consider(row['positions'], row['seq'], row['contacts'])
        self.handle = path.open('a', buffering=1)

    def elapsed(self):
        return self.base_elapsed + time.monotonic()-self.start

    def consider(self, fold, seq, expected=None):
        score = contacts(seq, fold, len(seq)//2)
        if expected is not None and score != expected:
            raise ValueError('Witness contact mismatch')
        if self.best_contacts is None or score > self.best_contacts:
            self.best, self.best_contacts = fold, score

    def emit(self, row):
        row = dict(row, elapsed_s=self.elapsed(), epoch=time.time())
        if row.get('type') == 'witness':
            self.consider(row['positions'], row['seq'], row['contacts'])
        self.handle.write(json.dumps(row, allow_nan=False)+'\n')

    def progress(self, **extra):
        save(self.out/'progress.json', dict(elapsed_s=self.elapsed(), best_contacts=self.best_contacts,
                                          best_positions=self.best, **extra))


def make_agent(config):
    import rl_reference as ref
    env = ref.HPProteinFoldingEnv(config['sequence'])
    keys = ('batch_size','memory_size','target_update_freq','d_model','nhead','num_layers','dim_feedforward','gamma','lr')
    return ref.DQNAgent(env, config['length'], device='cpu', **{k:config[k] for k in keys})


def seed_all(seed):
    import numpy as np
    import torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def evaluate(agent, recorder, *, episodes=10, end_epoch=float('inf')):
    was_training = agent.policy_net.training
    rewards = []; complete_scores = []
    agent.policy_net.eval()
    try:
        for _ in range(episodes):
            if STOP or time.time() >= end_epoch:
                break
            state = agent.env.reset(); reward_sum = 0
            for _ in range(2*agent.env.length):
                state, reward, done, _ = agent.env.step(agent.select_action(state, 0.0))
                reward_sum += reward
                if done:
                    break
            rewards.append(float(reward_sum))
            if len(agent.env.positions) == agent.env.length:
                fold = [list(map(int,p)) for p in agent.env.positions]
                score = contacts(agent.env.sequence, fold, agent.env.radius)
                if score != agent.env._calculate_hh_bonds():
                    raise RuntimeError('RL contact counter disagrees with independent witness')
                complete_scores.append(score)
                recorder.emit(dict(type='witness', source='greedy_evaluation', seq=agent.env.sequence,
                                   positions=fold, contacts=score))
    finally:
        agent.policy_net.train(was_training)
    return dict(episodes=len(rewards), completed_folds=len(complete_scores),
                mean_reward=statistics.mean(rewards) if rewards else None,
                mean_complete_contacts=statistics.mean(complete_scores) if complete_scores else None)


def checkpoint(agent, episode, recorder, next_evaluation, repairs):
    import numpy as np
    import torch
    started = time.monotonic()
    temporary = recorder.out/'checkpoint.pt.tmp'
    torch.save(dict(schema_version=1, episode=episode, policy=agent.policy_net.state_dict(),
                    target=agent.target_net.state_dict(), optimizer=agent.optimizer.state_dict(),
                    replay=agent.memory.__dict__, steps_done=agent.steps_done,
                    python_rng=random.getstate(), numpy_rng=np.random.get_state(), torch_rng=torch.get_rng_state(),
                    best_positions=recorder.best, best_contacts=recorder.best_contacts,
                    next_evaluation=next_evaluation, repairs=repairs, elapsed_s=recorder.elapsed()), temporary)
    temporary.replace(recorder.out/'checkpoint.pt')
    recorder.emit(dict(type='checkpoint', episode=episode, write_s=time.monotonic()-started,
                       replay_size=len(agent.memory), updates=agent.steps_done))


def rl_run(args, protocol, config, recorder):
    import numpy as np
    import torch
    seed_all(args.seed)
    agent = make_agent(config)
    episode, repairs, next_evaluation = 0, 0, 0
    checkpoint_path = args.out/'checkpoint.pt'
    if args.resume:
        # Only load this campaign's own trusted local checkpoint, never arbitrary downloaded weights.
        state = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        agent.policy_net.load_state_dict(state['policy']); agent.target_net.load_state_dict(state['target'])
        agent.optimizer.load_state_dict(state['optimizer']); agent.memory.__dict__.update(state['replay'])
        agent.steps_done = state['steps_done']; episode = state['episode']; repairs = state['repairs']
        random.setstate(state['python_rng']); np.random.set_state(state['numpy_rng']); torch.set_rng_state(state['torch_rng'])
        next_evaluation = state['next_evaluation']
        recorder.emit(dict(type='resume_checkpoint', episode=episode, restored_elapsed_s=state['elapsed_s'],
                           previously_consumed_elapsed_s=recorder.base_elapsed))
    agent.policy_net.train()
    max_episodes = protocol['max_episodes']
    last_checkpoint = recorder.elapsed()
    try:
        while not STOP and time.time() < args.end_epoch:
            training = episode < max_episodes
            agent.policy_net.train(training)
            epsilon = (1.0 if episode == 0 else config['epsilon_end']+(1-config['epsilon_end'])*math.exp(-(episode-1)*5/max_episodes)) if training else .05
            state = agent.env.reset(); total_reward = 0
            for _ in range(2*config['length']):
                if STOP or time.time() >= args.end_epoch:
                    break
                action = agent.select_action(state, epsilon)
                new_state, reward, done, _ = agent.env.step(action)
                if training:
                    agent.store_transition(state, action, reward, new_state, done)
                state = new_state; total_reward += reward
                if done:
                    break
            if STOP or time.time() >= args.end_epoch:
                break
            loss = agent.update() if training else None  # Reference: at most one update per episode.
            episode += 1
            complete = len(agent.env.positions) == config['length']
            score = None
            if complete:
                fold = [list(map(int,p)) for p in agent.env.positions]
                score = contacts(config['sequence'], fold, agent.env.radius)
                if score != agent.env._calculate_hh_bonds():
                    raise RuntimeError('RL independent score mismatch')
                if recorder.best_contacts is None or score > recorder.best_contacts:
                    recorder.emit(dict(type='witness', source='rl_training' if training else 'frozen_policy',
                                       seq=config['sequence'], positions=fold, contacts=score, episode=episode))
            recorder.emit(dict(type='episode', episode=episode, training=training, epsilon=epsilon,
                               complete=complete, contacts=score, reward=float(total_reward), loss=loss,
                               updates=agent.steps_done))
            if args.arm == 'rl_cp_sat' and episode % protocol['repair_every_episodes'] == 0 and recorder.best is not None:
                from hp_solver import solve
                # Alternate suffix/internal windows; selection is fixed, not a learned controller.
                size = min(protocol['repair_size'], config['length']-2)
                start = config['length']-size if repairs%2 == 0 else max(2, (config['length']-size)//2)
                duration = min(protocol['repair_seconds'], max(0,args.end_epoch-time.time()-5))
                if duration > 0 and not STOP:
                    solve(config['sequence'], seconds=duration, threads=args.threads,
                          seed=(args.seed*1000003+repairs)%2147483647,
                          emit=recorder.emit, incumbent=recorder.best,
                          movable=set(range(start,start+size)), stopped=lambda:STOP)
                    repairs += 1
            due_time = recorder.elapsed() >= next_evaluation
            if due_time or episode % config['eval_interval'] == 0 or episode == max_episodes:
                result = evaluate(agent, recorder, end_epoch=args.end_epoch)
                recorder.emit(dict(type='evaluation', episode=episode, **result))
                next_evaluation = (int(recorder.elapsed()//3600)+1)*3600
            due_checkpoint = recorder.elapsed()-last_checkpoint >= protocol['checkpoint_seconds']
            if due_checkpoint or episode % protocol['checkpoint_episodes'] == 0 or episode == max_episodes:
                checkpoint(agent, episode, recorder, next_evaluation, repairs)
                last_checkpoint = recorder.elapsed()
            if episode%100 == 0:
                recorder.progress(status='running', episode=episode, updates=agent.steps_done, training=training)
    finally:
        checkpoint(agent, episode, recorder, next_evaluation, repairs)
        recorder.progress(status='stopped' if STOP else 'budget_complete', episode=episode,
                          updates=agent.steps_done, training_complete=episode >= max_episodes)
    return dict(episode=episode, updates=agent.steps_done, training_complete=episode>=max_episodes,
                full_cube_optimal=False, repairs=repairs)


def pilot_run(args, protocol, recorder):
    from resource_pilot import run
    return run(args, protocol, recorder, stopped=lambda: STOP)


def np_valid(env):
    import numpy as np
    return [int(a) for a in np.flatnonzero(env.get_valid_actions())]


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--out',type=Path,required=True); p.add_argument('--arm',required=True)
    p.add_argument('--seq-id',required=True); p.add_argument('--seed',type=int,required=True)
    p.add_argument('--threads',type=int,required=True); p.add_argument('--end-epoch',type=float,required=True)
    p.add_argument('--base-elapsed',type=float,default=0); p.add_argument('--resume',action='store_true')
    p.add_argument('--smoke',action='store_true'); args=p.parse_args()
    for sig in (signal.SIGTERM, signal.SIGUSR1): signal.signal(sig, stopping)
    protocol=json.loads((HERE/'protocol.json').read_text())
    config=protocol['configurations'].get(args.seq_id)
    if args.smoke:
        protocol.update(max_episodes=4, checkpoint_episodes=2, repair_every_episodes=2, repair_seconds=.15)
        if config:
            config=dict(config,batch_size=8,memory_size=64,d_model=16,nhead=2,num_layers=1,dim_feedforward=32,eval_interval=2)
    recorder=Recorder(args.out,config['sequence'] if config else None,args.base_elapsed)
    result=dict(status='running', arm=args.arm, seq_id=args.seq_id, seed=args.seed, engineering_only=args.smoke)
    try:
        import torch
        versions={}
        for package in ('torch','numpy','ortools','transformers','safetensors','huggingface-hub'):
            try: versions[package]=version(package)
            except PackageNotFoundError: versions[package]=None
        required={'torch':'2.9.0','numpy':'2.3.4','ortools':'9.15.6755'}
        if args.arm=='decisionboost':required['transformers']='4.57.3'
        if not args.smoke:
            for package,wanted in required.items():
                if (versions.get(package) or '').split('+')[0]!=wanted:
                    raise RuntimeError(f'{package} version mismatch: {versions.get(package)}; run setup.sh for {wanted}')
        result['versions']=versions
        torch.set_num_threads(args.threads); torch.set_num_interop_threads(1)
        recorder.emit(dict(type='worker_start',arm=args.arm,seed=args.seed,device='cpu',threads=args.threads,
                           torch_version=str(torch.__version__),versions=versions,end_epoch=args.end_epoch))
        if args.arm in ('rl','rl_cp_sat'):
            result.update(rl_run(args,protocol,config,recorder))
        elif args.arm=='cp_sat':
            from hp_solver import solve
            _,status=solve(config['sequence'],seconds=max(.001,args.end_epoch-time.time()),threads=args.threads,
                           seed=args.seed,emit=recorder.emit,incumbent=recorder.best if args.resume else None,stopped=lambda:STOP)
            result.update(solver_status=status,full_cube_optimal=status=='OPTIMAL')
        elif args.arm=='pilot': result.update(pilot_run(args,protocol,recorder))
        elif args.arm=='decisionboost':
            from decisionboost_run import run
            result.update(run(args,recorder,stopped=lambda:STOP))
        else: raise ValueError('Unknown arm')
        if result['status']=='running': result['status']='stopped' if STOP else 'finished'
    except BaseException as exc:
        result.update(status='failed',error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        result.update(elapsed_s=recorder.elapsed(),best_contacts=recorder.best_contacts,best_positions=recorder.best)
        save(args.out/'result.json',result); recorder.emit(dict(type='worker_end',**result)); recorder.handle.close()


if __name__=='__main__': main()
