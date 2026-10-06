"""Predeclared offline overnight confirmation, with absolute London-time cutoff.

Five fresh transformer/decision-controller pipelines, fresh grouped sequences,
matched solver allowances, stronger static/cheap controls, and append-only audit.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import signal
import statistics
import time

import numpy as np
import torch
import cheap_hp as hp
from repair_controller import RegionChoice, features
from solver_pilot import Pilot, best, safe_fallback
from overnight_encoder import encode_sequences

hp.MAX_N=64
ACTION_NAMES=['suffix6','suffix10','suffix16','internal6','internal10','internal16']
ARMS=['fixed6','fixed10','static_global','static_length','random','learned','cheap_fixed10']


def stable_seed(*parts):
    return int(hashlib.sha256(':'.join(map(str,parts)).encode()).hexdigest()[:8],16)%2147483647


def region_menu(n, context):
    menu=[]
    for name in ACTION_NAMES:
        size=int(name.removeprefix('suffix').removeprefix('internal'))
        start=max(2,n-size) if name.startswith('suffix') else random.Random(context+size).randint(2,max(2,n-size-1))
        end=min(n,start+size)
        existing=next((w for w in menu if (w['start'],w['end'])==(start,end)),None)
        if existing:existing['aliases'].append(name)
        else:menu.append(dict(name=name,aliases=[name],start=start,end=end))
    return menu


def action_index(menu,name):
    return next(i for i,w in enumerate(menu) if name in w['aliases'])


def fresh_corpus():
    old=hp.corpus()
    configs=json.loads((hp.ROOT/'outputs/rl_profile_20261004/configurations.json').read_text())
    blocked={min(s,s[::-1]) for ss in old.values() for s in ss}
    blocked.update(min(c['sequence'],c['sequence'][::-1]) for c in configs.values())
    used=set(blocked);rng=random.Random(76261004);splits={};records=[]
    for split,counts in [('train',{n:32 for n in (16,20,28,36)}),
                         ('validation',{n:8 for n in (16,20,28,36)}),
                         ('test',{n:12 for n in (16,20,28,36,48,58)})]:
        splits[split]=[]
        for n,count in counts.items():
            for i in range(count):
                fraction=(.35,.5,.65)[i%3];family='iid' if i%2==0 else 'blocks'
                while True:
                    if family=='iid':seq=''.join('H' if rng.random()<fraction else 'P' for _ in range(n))
                    else:
                        letters=['H' if rng.random()<fraction else 'P']
                        for _ in range(n-1):
                            letters.append(letters[-1] if rng.random()<.65 else ('H' if rng.random()<fraction else 'P'))
                        seq=''.join(letters)
                    key=min(seq,seq[::-1])
                    if key not in used and .2<=seq.count('H')/n<=.8:break
                used.add(key);splits[split].append(seq)
                records.append(dict(seq=seq,split=split,length=n,family=family,target_h_fraction=fraction))
    paper={name:configs[name]['sequence'] for name in configs}
    return splits,records,sorted(blocked),paper


class NightPilot(Pilot):
    def __init__(self,out,compute_end):
        super().__init__(out,max(1,compute_end-time.time()))
        self.compute_end=compute_end;self.event_number=0;self.call_number=0;self.active_call=None
        self.pending_calls=set()
        self.active_solver_seed=None
        self.completed_cases=0

    def event(self,record):
        record=dict(record);self.event_number+=1;record['event_id']=self.event_number
        record['utc']=datetime.now(timezone.utc).isoformat()
        if record.get('type')=='solver':
            record['call_id']=self.active_call;record['solver_seed']=self.active_solver_seed
        super().event(record)

    def solve(self,seq,**kwargs):
        self.check(kwargs['seconds']+10)
        self.call_number+=1;self.active_call=self.call_number
        self.active_solver_seed=kwargs['seed']
        self.pending_calls.add(self.active_call)
        self.event(dict(type='solver_start',call_id=self.active_call,seq=seq,phase=kwargs['phase'],budget_s=kwargs['seconds'],solver_seed=kwargs['seed']))
        try:
            row=super().solve(seq,**kwargs)
            row['call_id']=self.active_call
            row['solver_seed']=self.active_solver_seed
            self.pending_calls.discard(self.active_call)
            return row
        finally:self.active_call=None;self.active_solver_seed=None

    def check(self,reserve=10):
        if self.remaining()<reserve:raise TimeoutError('Compute cutoff reached; preserving partial run')

    def progress(self,phase,**kwargs):
        hp.save(self.out/'progress.json',dict(status='running',phase=phase,pid=os.getpid(),
            core_elapsed_s=time.perf_counter()-self.started,solver_calls=self.solver_calls,
            completed_cases=self.completed_cases,remaining_compute_s=max(0,self.remaining()),**kwargs))


def archive_for(sequences,seed):
    st=time.perf_counter();rng=random.Random(seed);archive={};complete=0
    for seq in sequences:
        folds=[hp.grow(seq,rng) for _ in range(64)];complete+=sum(p is not None for p in folds)
        elite=hp.update_archive(seq,[],folds,limit=4)
        if not elite:elite=[safe_fallback(seq,seed)]
        for _ in range(64):elite=hp.update_archive(seq,elite,[hp.improve(seq,rng.choice(elite),rng)],limit=4)
        archive[seq]=elite
    return archive,dict(wall_s=time.perf_counter()-st,attempts=64*len(sequences),complete=complete)


def pack_states(states):
    x=torch.zeros(len(states),6,20);mask=torch.zeros(len(states),6,dtype=torch.bool);target=torch.zeros(len(states),6)
    for i,s in enumerate(states):
        k=len(s['windows']);x[i,:k]=torch.tensor(s['features']);mask[i,:k]=True
        if 'target' in s:target[i,:k]=torch.tensor(s['target'])
    return x,mask,target


def fit_region(pilot,states,seed,steps):
    st=time.perf_counter();torch.manual_seed(seed);model=RegionChoice()
    opt=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.01)
    x,mask,target=pack_states(states);g=torch.Generator().manual_seed(seed);losses=[]
    for i in range(steps):
        if i%100==0:pilot.check(20)
        idx=torch.randint(len(states),(64,),generator=g)
        loss=-(target[idx]*model(x[idx],mask[idx]).log_softmax(-1)).sum(-1).mean()
        opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step()
        losses.append(float(loss.detach()))
    model.eval()
    return model,dict(wall_s=time.perf_counter()-st,steps=steps,first20_loss=statistics.mean(losses[:20]),last20_loss=statistics.mean(losses[-20:]))


def repair(pilot,seq,fold,w,phase,seed,seconds):
    frozen=[i for i in range(len(seq)) if not w['start']<=i<w['end']]
    return pilot.solve(seq,phase=phase,seed=seed,seconds=seconds,proposal=fold,frozen_indices=frozen)


def state_hash(model):
    return hashlib.sha256(b''.join(t.detach().numpy().tobytes() for t in model.state_dict().values())).hexdigest()


def seed_pipeline(pilot,seqs,embedding,seed,args):
    pilot.progress('bootstrap',seed=seed);pilot.check(120)
    seedout=pilot.out/f'seed_{seed}';seedout.mkdir()
    archive,archive_cost=archive_for(seqs,stable_seed(seed,'archive'))
    hp.save(seedout/'initial_archive.json',archive)
    for seq in seqs:pilot.event(dict(type='archive',seq=seq,seed=seed,stage='initial',folds=archive[seq]))
    st=time.perf_counter();data=hp.teacher_examples(seqs,archive);data_s=time.perf_counter()-st
    torch.manual_seed(seed);proposer=hp.ChoiceHead(next(iter(embedding.values())).shape[-1])
    opt=torch.optim.AdamW(proposer.parameters(),lr=3e-4,weight_decay=.01)
    pilot.progress('proposer_training',seed=seed)
    proposer_training=hp.train(proposer,opt,data,seqs,embedding,args.proposer_steps,stable_seed(seed,'proposer_train'))
    del data
    pilot.check(60);pilot.progress('controller_teaching',seed=seed)
    proposals,inference_s=hp.sample_folds(proposer,seqs,embedding,16,stable_seed(seed,'teacher_proposals'))
    states=[];teacher_cost=0;teacher_start=time.perf_counter()
    for i,seq in enumerate(seqs):
        pilot.check(20)
        pilot.event(dict(type='proposals',seq=seq,seed=seed,phase='teacher',folds=proposals[seq]))
        number=2 if i%32<16 else 1
        folds=hp.update_archive(seq,[],proposals[seq],limit=number)
        if not folds:folds=[best(seq,archive[seq])]
        for j,fold in enumerate(folds):
            ident=f'train_{seed}_{i}_{j}';context=stable_seed(seed,ident,'menu')
            menu=region_menu(len(seq),context);outcomes=[];gains=[]
            for k,w in enumerate(menu):
                row=repair(pilot,seq,fold,w,f'teacher_{ident}_{k}',stable_seed(seed,ident,'cp'),args.solver_seconds)
                outcomes.append(row['call_id']);gains.append(row['actual_contacts']-hp.score(seq,fold));teacher_cost+=row['wall_s']
                archive[seq]=hp.update_archive(seq,archive[seq],[row['positions']],limit=4)
            winners=[k for k,g in enumerate(gains) if g==max(gains)]
            state=dict(type='teacher',seq=seq,seed=seed,state_id=ident,positions=fold,
                windows=menu,context_seed=context,features=features(seq,fold,menu),outcome_ids=outcomes,
                gains=gains,target=[1/len(winners) if k in winners else 0 for k in range(len(menu))])
            states.append(state);pilot.event(state)
        archive[seq]=hp.update_archive(seq,archive[seq],proposals[seq],limit=4)
        if i%8==0:pilot.progress('controller_teaching',seed=seed,teacher_sequences=i+1)
    all_gains={name:[] for name in ACTION_NAMES};by_length=defaultdict(lambda:{name:[] for name in ACTION_NAMES})
    teacher_wall_s=time.perf_counter()-teacher_start
    for state in states:
        for name in ACTION_NAMES:
            gain=state['gains'][action_index(state['windows'],name)]
            all_gains[name].append(gain);by_length[len(state['seq'])][name].append(gain)
    means={name:statistics.mean(v) for name,v in all_gains.items()}
    length_means={n:{name:statistics.mean(v) for name,v in d.items()} for n,d in by_length.items()}
    static=max(ACTION_NAMES,key=lambda name:means[name])
    statics={n:max(ACTION_NAMES,key=lambda name:d[name]) for n,d in length_means.items()}
    pilot.check(60);pilot.progress('feedback_training',seed=seed)
    st=time.perf_counter();data=hp.teacher_examples(seqs,archive);feedback_data_s=time.perf_counter()-st
    feedback=hp.train(proposer,opt,data,seqs,embedding,args.feedback_steps,stable_seed(seed,'feedback_train'));del data
    proposer.eval().requires_grad_(False)
    controller,controller_cost=fit_region(pilot,states,stable_seed(seed,'controller'),args.controller_steps)
    record=dict(type='seed_training',seed=seed,archive_cost=archive_cost,initial_data_s=data_s,
        proposer_training=proposer_training,teacher_proposer_inference_s=inference_s,
        teacher_solver_s=teacher_cost,teacher_wall_s=teacher_wall_s,
        teacher_states=len(states),static_global_action=static,static_by_length=statics,
        teacher_action_means=means,teacher_length_action_means=length_means,
        feedback_data_s=feedback_data_s,feedback_training=feedback,controller_training=controller_cost,
        proposer_state_hash=state_hash(proposer))
    torch.save({'state_dict':proposer.state_dict(),'max_n':64},seedout/'proposer.pt')
    torch.save({'state_dict':controller.state_dict()},seedout/'controller.pt')
    hp.save(seedout/'training.json',record);hp.save(seedout/'final_archive.json',archive)
    for seq in seqs:pilot.event(dict(type='archive',seq=seq,seed=seed,stage='final',folds=archive[seq]))
    pilot.event(record)
    return dict(proposer=proposer,controller=controller,static=static,statics=statics,training=record)


def evaluate_case(pilot,pipeline,seq,seed,ident,steps,seconds,phase):
    # Reserve the entire case block at worst-case caps before launching it.
    pilot.check(len(ARMS)*steps*(seconds+.15)+steps*seconds+15)
    st=time.perf_counter();case_id=f'{phase}_{seed}_{ident}'
    proposals,inference_s=hp.sample_folds(pipeline['proposer'],[seq],
        pipeline['embedding'],16,stable_seed(seed,case_id,'proposals'))
    model_fold=best(seq,proposals[seq]);rng=random.Random(stable_seed(seed,case_id,'cheap'))
    cheap_st=time.perf_counter();cheap_folds=[hp.grow(seq,rng) for _ in range(16)]
    cheap_fold=best(seq,cheap_folds) or safe_fallback(seq,stable_seed(seed,case_id,'fallback'))
    cheap_s=time.perf_counter()-cheap_st;fallback=model_fold is None
    if fallback:model_fold=cheap_fold
    pilot.event(dict(type='proposals',seq=seq,seed=seed,phase=phase,case_id=case_id,source='transformer',folds=proposals[seq]))
    pilot.event(dict(type='proposals',seq=seq,seed=seed,phase=phase,case_id=case_id,source='cheap',folds=cheap_folds))
    results={};order=ARMS[ident%len(ARMS):]+ARMS[:ident%len(ARMS)]
    for arm in order:
        pos=list(cheap_fold if arm=='cheap_fixed10' else model_fold);initial_contacts=hp.score(seq,pos)
        arm_st=time.perf_counter();ids=[];checkpoints=[]
        for step in range(steps):
            policy_st=time.perf_counter();context=stable_seed(seed,case_id,step,'menu')
            menu=region_menu(len(seq),context);f=None;scores=None
            if arm in ('fixed10','cheap_fixed10'):choice=action_index(menu,'suffix10')
            elif arm=='fixed6':choice=action_index(menu,'suffix6')
            elif arm=='static_global':choice=action_index(menu,pipeline['static'])
            elif arm=='static_length':
                nearest=min(pipeline['statics'],key=lambda n:(abs(n-len(seq)),n));choice=action_index(menu,pipeline['statics'][nearest])
            elif arm=='random':choice=random.Random(stable_seed(seed,case_id,arm,step,'choice')).randrange(len(menu))
            else:
                f=features(seq,pos,menu);x,mask,_=pack_states([dict(windows=menu,features=f)])
                with torch.inference_mode():scores=pipeline['controller'](x,mask)[0,:len(menu)].tolist()
                choice=max(range(len(menu)),key=lambda k:scores[k])
            policy_s=time.perf_counter()-policy_st
            before=list(pos);row=repair(pilot,seq,before,menu[choice],f'eval_{case_id}_{arm}_{step}',
                stable_seed(seed,case_id,step,'cp'),seconds)
            ids.append(row['call_id']);pos=row['positions']
            step_record=dict(type='evaluation_step',phase=phase,case_id=case_id,seq=seq,seed=seed,arm=arm,
                step=step,proposal=before,windows=menu,context_seed=context,features=f,scores=scores,
                choice=choice,window=menu[choice],call_id=row['call_id'],policy_s=policy_s,
                wall_s=policy_s+row['wall_s'],elapsed_arm_s=time.perf_counter()-arm_st,
                contacts=row['actual_contacts'])
            pilot.event(step_record);checkpoints.append(dict(contacts=row['actual_contacts'],elapsed_s=step_record['elapsed_arm_s']))
        results[arm]=dict(initial_contacts=initial_contacts,final_contacts=hp.score(seq,pos),positions=pos,
            call_ids=ids,wall_s=time.perf_counter()-arm_st,checkpoints=checkpoints,
            online_s=(cheap_s if arm=='cheap_fixed10' else inference_s)+time.perf_counter()-arm_st)
    cold=pilot.solve(seq,phase=f'cold_{case_id}',seed=stable_seed(seed,case_id,'cold'),seconds=steps*seconds)
    event=dict(type='evaluation_case',phase=phase,case_id=case_id,seq=seq,seed=seed,
        initial_positions=model_fold,cheap_positions=cheap_fold,transformer_fallback_to_cheap=fallback,
        proposals_complete=sum(p is not None for p in proposals[seq]),cheap_complete=sum(p is not None for p in cheap_folds),
        proposer_inference_s=inference_s,cheap_generation_s=cheap_s,arms=results,
        cold_call_id=cold['call_id'],cold_contacts=cold['actual_contacts'],cold_status=cold['status'],
        wall_s=time.perf_counter()-st)
    pilot.event(event);pilot.completed_cases+=1
    pilot.progress('evaluation',seed=seed,phase_detail=phase,last_case=case_id)
    return event


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    p.add_argument('--compute-end',default='2026-10-05T04:45:00+00:00')
    p.add_argument('--hard-end',default='2026-10-05T04:50:00+00:00')
    p.add_argument('--proposer-steps',type=int,default=1200);p.add_argument('--feedback-steps',type=int,default=256)
    p.add_argument('--controller-steps',type=int,default=1200);p.add_argument('--solver-seconds',type=float,default=1.)
    p.add_argument('--smoke',action='store_true');args=p.parse_args()
    compute_end=datetime.fromisoformat(args.compute_end).timestamp();hard_end=datetime.fromisoformat(args.hard_end).timestamp()
    pilot=NightPilot(args.out,compute_end)
    def expired(sig,frame):raise TimeoutError('Absolute hard compute deadline')
    signal.signal(signal.SIGALRM,expired);signal.setitimer(signal.ITIMER_REAL,max(.01,hard_end-time.time()))
    torch.set_num_threads(4);splits,sequence_records,blocked,paper=fresh_corpus()
    if args.smoke:
        splits={key:[next(s for s in seqs if len(s)==n) for n in (16,36)] for key,seqs in splits.items()}
        args.proposer_steps=2;args.feedback_steps=2;args.controller_steps=2;args.solver_seconds=.05
    seeds=[0] if args.smoke else list(range(5))
    manifest=dict(schema_version=1,created_utc=datetime.now(timezone.utc).isoformat(),splits=splits,
        sequence_records=sequence_records,blocked_reversal_keys=blocked,
        blocked_sha256=hashlib.sha256(json.dumps(blocked).encode()).hexdigest(),paper_sequences=paper,
        train_lengths=[16,20,28,36],ood_test_lengths=[48,58],max_n=64,pipeline_seeds=seeds,
        actions=ACTION_NAMES,arms=ARMS,proposer_steps=args.proposer_steps,feedback_steps=args.feedback_steps,
        controller_steps=args.controller_steps,solver_seconds=args.solver_seconds,teacher_budget_s=args.solver_seconds,
        evaluation_steps=1 if args.smoke else 4,paper_steps=6,paper_solver_seconds=2.,
        fresh_primary=True,primary_comparators=['fixed10','static_length'],
        primary_endpoint='Mean final HH contact difference, grouped by fresh sequence, common solver-call/search-cap allowance',
        secondary_endpoint='Logged online time/quality tradeoff and standalone offline cost accounting; no wall-budget superiority inferred from call matching',
        static_length_rule='Nearest training length; ties smaller length; static choice only teacher outcomes',
        internal_window_rule='Random(context_seed+size).randint(2,max(2,n-size-1)); shared context per seed/case/step',
        tie_rule='First maximum in declared action/menu order; uniform teacher target on all maximal observed gains',
        encoder_offline=True,encoder_revision=hp.REVISION,features_symmetry=True,api_calls=0,
        compute_end=args.compute_end,hard_compute_end=args.hard_end,final_deadline_utc='2026-10-05T05:00:00+00:00',
        smoke=args.smoke,code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    hp.save(args.out/'manifest.json',manifest)
    source=args.out/'executed_sources';source.mkdir()
    for name in ('overnight_test.py','overnight_encoder.py','repair_controller.py','solver_pilot.py','cheap_hp.py',
                 'overnight_supervisor.py','analyse_overnight.py','verify_overnight.py','verify_solver_pilot.py','verify_controller_pilot.py'):
        shutil.copyfile(Path(__file__).parent/name,source/name)
    result=dict(status='running',schema_version=1,pipeline_seeds=seeds,completed_seeds=[],completed_primary_cases=0,
        completed_paper_cases=0,api_calls=0,manifest_sha256=hashlib.sha256((args.out/'manifest.json').read_bytes()).hexdigest())
    try:
        pilot.progress('encoding');sequences=[s for v in splits.values() for s in v]
        if not args.smoke:sequences+=list(paper.values())
        embedding,encoder_cost=encode_sequences(sequences,args.out,max_n=64);result['encoder']=encoder_cost
        pipelines={}
        for seed in seeds:
            pipeline=seed_pipeline(pilot,splits['train'],embedding,seed,args)
            pipeline['embedding']=embedding;pipelines[seed]=pipeline;result['completed_seeds'].append(seed)
            hp.save(args.out/'results.json',result)
        test=splits['test']
        # Round robin by sequence index then pipeline; lengths already cycle in
        # a transposed ordering, preserving coverage if the deadline truncates.
        grouped=[[s for s in test if len(s)==n] for n in sorted({len(s) for s in test})]
        ordered=[s for chunk in zip(*grouped) for s in chunk]
        for i,seq in enumerate(ordered):
            for seed in seeds:
                evaluate_case(pilot,pipelines[seed],seq,seed,i,manifest['evaluation_steps'],args.solver_seconds,'primary')
                result['completed_primary_cases']+=1;hp.save(args.out/'results.json',result)
        if not args.smoke:
            for i,(name,seq) in enumerate(paper.items()):
                for seed in seeds:
                    evaluate_case(pilot,pipelines[seed],seq,seed,1000+i,6,2.,'paper')
                    result['completed_paper_cases']+=1;hp.save(args.out/'results.json',result)
        for seed,pipeline in pipelines.items():
            unchanged=state_hash(pipeline['proposer'])==pipeline['training']['proposer_state_hash']
            assert unchanged;pilot.event(dict(type='freeze_check',seed=seed,proposer_weights_unchanged=unchanged))
        result['status']='complete'
    except TimeoutError as exc:
        result['status']='deadline_partial';result['reason']=str(exc)
    except BaseException as exc:
        result['status']='failed';result['error']=type(exc).__name__+': '+str(exc)
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        result['solver_calls']=pilot.solver_calls;result['core_wall_s']=time.perf_counter()-pilot.started
        result['ended_utc']=datetime.now(timezone.utc).isoformat();hp.save(args.out/'results.json',result)
        pilot.event(dict(type='run_end',status=result['status'],reason=result.get('reason',result.get('error')),
            solver_calls=pilot.solver_calls,completed_cases=pilot.completed_cases,pending_calls=sorted(pilot.pending_calls)))
        pilot.events.close();hp.save(args.out/'progress.json',dict(result,pid=os.getpid(),phase='finished'))
        print(json.dumps(result),flush=True)


if __name__=='__main__':main()
