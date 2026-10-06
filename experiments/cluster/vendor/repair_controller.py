"""Three-part pilot: frozen transformer proposer, learned region choice, CP repair.

Controller targets describe witnessed improvement under a short solver cap,
not optimal action values. No API, downloads, RL campaign, or proposer updates.
"""
import argparse
import hashlib
import json
import random
import shutil
import signal
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
import cheap_hp as hp
from solver_pilot import Pilot, best, safe_fallback


def windows(n):
    """Intervals [start,end); aliases keep action names stable for static policy."""
    result=[]
    for name,size,suffix in [('suffix6',6,True),('suffix10',10,True),
                             ('internal6',6,False),('internal10',10,False)]:
        start=max(2,n-size) if suffix else max(2,(n-size)//2)
        end=min(n,start+size)
        existing=next((w for w in result if (w['start'],w['end'])==(start,end)),None)
        if existing is not None:existing['aliases'].append(name)
        else:result.append(dict(name=name,aliases=[name],start=start,end=end))
    return result


def action_for(menu, name):
    return next(i for i,w in enumerate(menu) if name in w['aliases'])


def features(seq,fold,menu):
    n=len(seq);occupied={p:i for i,p in enumerate(fold)};total=hp.score(seq,fold)
    rows=[]
    for w in menu:
        start,end=w['start'],w['end'];size=end-start
        region=set(range(start,end));h=sum(seq[i]=='H' for i in region)
        internal=cross=free=external_h=0
        for i in region:
            for d in hp.DIRECTIONS:
                p=tuple(a+b for a,b in zip(fold[i],d));j=occupied.get(p)
                free+=j is None
                if j is not None and seq[j]=='H' and seq[i]=='H' and abs(i-j)>1:
                    internal+=j in region;cross+=j not in region
        # Sequence/geometry available before any candidate repair; no oracle data.
        for j in range(n):
            if j not in region and seq[j]=='H':
                external_h+=min(hp.manhattan(fold[j],fold[i]) for i in region)<=3
        boundary=hp.manhattan(fold[start-1],fold[end]) if end<n else 0
        coords=np.asarray(fold[start:end],dtype=float)
        span=coords.max(0)-coords.min(0)
        # Teacher archives are canonicalized; sampled evaluation folds are not.
        # Signed y/z swaps preserve the +x anchor, so do not encode that arbitrary
        # orientation in the controller inputs.
        span=np.asarray([span[0],*sorted(span[1:])])
        rows.append([n/20,total/n,seq.count('H')/n,start/n,end/n,size/n,
                     end==n,h/size,internal/(2*size),cross/size,free/(6*size),
                     boundary/(size+1),external_h/n,*list(span/size),
                     float(np.linalg.norm(span))/size,seq[start-1]=='H',
                     seq[end]=='H' if end<n else 0,seq[start:end].count('HP')/size])
    return rows


class RegionChoice(nn.Module):
    def __init__(self):
        super().__init__()
        self.project=nn.Linear(20,32)
        layer=nn.TransformerEncoderLayer(32,4,64,dropout=0,batch_first=True,norm_first=True)
        self.transformer=nn.TransformerEncoder(layer,1,enable_nested_tensor=False)
        self.output=nn.Linear(32,1)

    def forward(self,x,mask):
        z=self.transformer(self.project(x),src_key_padding_mask=~mask)
        return self.output(z).squeeze(-1).masked_fill(~mask,-1e9)


def tensors(states):
    x=torch.zeros(len(states),4,20);mask=torch.zeros(len(states),4,dtype=torch.bool)
    target=torch.zeros(len(states),4)
    for i,s in enumerate(states):
        k=len(s['windows']);x[i,:k]=torch.tensor(s['features']);mask[i,:k]=True
        if 'target' in s:target[i,:k]=torch.tensor(s['target'])
    return x,mask,target


def train_controller(states,seed):
    st=time.perf_counter();torch.manual_seed(seed);model=RegionChoice()
    optimizer=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.01)
    x,mask,target=tensors(states);g=torch.Generator().manual_seed(seed);losses=[]
    model.train()
    for _ in range(240):
        idx=torch.randint(len(states),(32,),generator=g)
        loss=-(target[idx]*model(x[idx],mask[idx]).log_softmax(-1)).sum(-1).mean()
        optimizer.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1)
        optimizer.step();losses.append(float(loss.detach()))
    model.eval()
    return model,dict(seed=seed,steps=240,seconds=time.perf_counter()-st,
        first20_loss=statistics.mean(losses[:20]),last20_loss=statistics.mean(losses[-20:]),
        parameters=sum(p.numel() for p in model.parameters()))


def solve_window(pilot,seq,fold,w,phase,seed):
    frozen=[i for i in range(len(seq)) if not w['start']<=i<w['end']]
    return pilot.solve(seq,phase=phase,seed=seed,seconds=.25,proposal=fold,frozen_indices=frozen)


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    p.add_argument('--wall-limit',type=float,default=900);args=p.parse_args()
    pilot=Pilot(args.out,args.wall_limit)
    def expired(signum,frame):raise TimeoutError('Pilot wall deadline reached')
    signal.signal(signal.SIGALRM,expired);signal.setitimer(signal.ITIMER_REAL,args.wall_limit)
    torch.set_num_threads(4)
    prior=hp.ROOT/'outputs/decisionboost_solver_pilot_20261004'
    checkpoint=torch.load(prior/'decisionboost.pt',weights_only=True,map_location='cpu')
    corpus=hp.corpus();train=checkpoint['manifest']['splits']['train']
    test=sum(([s for s in corpus['test'] if len(s)==n][4:12] for n in (12,16)),[])
    ood=corpus['ood'][4:12];splits=dict(train=train,test=test,ood=ood)
    keys=[min(s,s[::-1]) for ss in splits.values() for s in ss];assert len(keys)==len(set(keys))
    cache_path=hp.ROOT/'outputs/decisionboost_cheap_run/encoder_features.pt'
    cache=torch.load(cache_path,weights_only=True,map_location='cpu');assert cache['revision']==hp.REVISION
    embedding={s:cache['features'][i] for i,s in enumerate(cache['sequences'])}
    proposer=hp.ChoiceHead(cache['features'].shape[-1]);proposer.load_state_dict(checkpoint['state_dict'])
    proposer.eval().requires_grad_(False)
    initial_proposer_hash=hashlib.sha256(b''.join(t.detach().numpy().tobytes() for t in proposer.state_dict().values())).hexdigest()
    manifest=dict(splits=splits,controller_seeds=[101,102,103],proposer_checkpoint=str(prior/'decisionboost.pt'),
        proposer_checkpoint_sha256=hashlib.sha256((prior/'decisionboost.pt').read_bytes()).hexdigest(),
        proposer_frozen=True,encoder_features_reused=True,encoder_revision=hp.REVISION,
        teacher_folds_per_sequence=2,proposal_attempts=16,solver_seconds=.25,solver_workers=1,
        actions=['suffix6','suffix10','internal6','internal10'],alias_rule='Deduplicate identical [start,end) intervals, retain named aliases',
        features_symmetry='Invariant under all eight signed y/z swaps preserving the fixed +x first bond',
        labels='Observed nondegrading contact gain under 0.25s; targets uniform among maximum observed gain, all uniform if no gain',
        static_rule='Largest mean observed teacher gain by action name; ties use action order',
        evaluation_arms=['fixed_suffix','static_teacher','random','learned'],
        evaluation_protocol='Same proposal and CP seed/cap for each arm, independent fresh CP calls; three seeds, sequence-grouped reporting',
        scope='Reused exploratory held-out groups; test indices4:12 unused by prior CP pilot but used by earlier cheap pilot',
        wall_limit_s=args.wall_limit,api_calls=0,
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    hp.save(args.out/'manifest.json',manifest)
    sources=args.out/'executed_sources';sources.mkdir()
    for filename in ('repair_controller.py','solver_pilot.py','cheap_hp.py'):
        shutil.copyfile(Path(__file__).parent/filename,sources/filename)
    result=dict(status='running',splits=splits,api_calls=0,teacher=[],training=[],evaluation=[])
    try:
        proposals,inference_s=hp.sample_folds(proposer,train,embedding,16,9200)
        result['teacher_proposer_inference_s']=inference_s
        for i,seq in enumerate(train):
            pilot.event(dict(type='proposals',phase='controller_teacher',seq=seq,folds=proposals[seq]))
            unique=hp.update_archive(seq,[],proposals[seq],limit=2)
            if not unique:unique=[safe_fallback(seq,9200+i)]
            if len(unique)==1:unique=unique*2
            for j,fold in enumerate(unique):
                state_id=f'train_{i}_{j}';menu=windows(len(seq));state=dict(seq=seq,state_id=state_id,
                    positions=fold,windows=menu,features=features(seq,fold,menu),outcomes=[])
                for k,w in enumerate(menu):
                    state['outcomes'].append(solve_window(pilot,seq,fold,w,f'teacher_{state_id}_{k}',9300+i*10+j))
                gains=[r['actual_contacts']-hp.score(seq,fold) for r in state['outcomes']]
                winners=[k for k,gain in enumerate(gains) if gain==max(gains)]
                state['gains']=gains;state['target']=[1/len(winners) if k in winners else 0 for k in range(len(menu))]
                result['teacher'].append(state)
            print(f'Teacher sequence {i+1}/{len(train)}',flush=True)
        action_gains={name:[] for name in manifest['actions']}
        for state in result['teacher']:
            for name in manifest['actions']:
                action_gains[name].append(state['gains'][action_for(state['windows'],name)])
        mean_gains={name:statistics.mean(v) for name,v in action_gains.items()}
        static=max(manifest['actions'],key=lambda name:mean_gains[name])
        result['static_teacher_action']=static;result['teacher_action_mean_gain']=mean_gains
        models={}
        for seed in manifest['controller_seeds']:
            model,log=train_controller(result['teacher'],seed);models[seed]=model;result['training'].append(log)
            torch.save({'state_dict':model.state_dict(),'manifest':manifest},args.out/f'controller_{seed}.pt')
        hp.save(args.out/'teacher.json',result['teacher'])
        eval_seqs=test+ood
        for seed in manifest['controller_seeds']:
            proposals,inference_s=hp.sample_folds(proposer,eval_seqs,embedding,16,10000+seed)
            result.setdefault('evaluation_proposer_costs',[]).append(dict(seed=seed,seconds=inference_s))
            for i,seq in enumerate(eval_seqs):
                pilot.event(dict(type='proposals',phase=f'eval_controller_{seed}',seq=seq,folds=proposals[seq]))
                fold=best(seq,proposals[seq]);fallback=fold is None
                if fallback:fold=safe_fallback(seq,11000+seed+i)
                menu=windows(len(seq));f=features(seq,fold,menu);x,mask,_=tensors([dict(windows=menu,features=f)])
                st=time.perf_counter()
                with torch.inference_mode():scores=models[seed](x,mask)[0,:len(menu)].tolist()
                choice_s=time.perf_counter()-st
                choice=max(range(len(menu)),key=lambda k:scores[k]);random_choice=random.Random(12000+seed*100+i).randrange(len(menu))
                choices=dict(fixed_suffix=action_for(menu,'suffix6'),static_teacher=action_for(menu,static),
                             random=random_choice,learned=choice)
                row=dict(seq=seq,state_id=f'eval_{seed}_{i}',seed=seed,proposal=fold,windows=menu,features=f,
                    scores=scores,choice=choice,choice_s=choice_s,choices=choices,fallback=fallback,rows={})
                # Rotate execution order to distribute any systematic warmup effect.
                names=list(choices);offset=i%len(names)
                for name in names[offset:]+names[:offset]:
                    k=choices[name]
                    row['rows'][name]=solve_window(pilot,seq,fold,menu[k],f'eval_{seed}_{i}_{name}',13000+seed*100+i)
                result['evaluation'].append(row)
            print(f'Evaluation seed {seed} complete',flush=True)
        result['proposer_weights_unchanged']=initial_proposer_hash==hashlib.sha256(b''.join(t.detach().numpy().tobytes() for t in proposer.state_dict().values())).hexdigest()
        assert result['proposer_weights_unchanged']
        result['solver_calls']=pilot.solver_calls;result['status']='complete'
        result['core_wall_s']=time.perf_counter()-pilot.started
        hp.save(args.out/'results.json',result)
        print(json.dumps(dict(status='complete',calls=pilot.solver_calls,wall_s=result['core_wall_s'])),flush=True)
    except BaseException as exc:
        result['status']='partial';result['error']=type(exc).__name__+': '+str(exc)
        hp.save(args.out/'results.json',result)
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);pilot.events.close()


if __name__=='__main__':main()
