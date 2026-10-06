"""Portable replication of the tested transformer/controller/CP method.

This secondary study uses the original complete reachable coordinate domains,
single-worker CP call caps, and a fresh frozen corpus. It is separate from the
CPU-only per-sequence RL/CP/hybrid comparison in the restricted common cube.
"""
import json
import os
from pathlib import Path
from types import SimpleNamespace
import time
from campaign import save
from geometry import contacts


def run(args, recorder, *, stopped):
    import torch
    import overnight_encoder as encoder
    import overnight_test as method
    import cheap_hp as hp
    here=Path(__file__).resolve().parent
    data=json.loads((here/'decisionboost_corpus.json').read_text())
    model_dir=Path(os.environ['HP_ENCODER_DIR'])
    encoder.SNAPSHOT=model_dir
    out=args.out/'decisionboost'

    class BoundedPilot(method.NightPilot):
        def check(self,reserve=10):
            if stopped(): raise TimeoutError('Scheduler/application stop requested')
            super().check(reserve)

    pilot=BoundedPilot(out,args.end_epoch)
    settings=SimpleNamespace(proposer_steps=1200,feedback_steps=256,controller_steps=1200,solver_seconds=1.)
    splits=data['splits']
    if args.smoke:
        splits={k:[next(s for s in rows if len(s)==n) for n in (16,36)] for k,rows in splits.items()}
        settings=SimpleNamespace(proposer_steps=2,feedback_steps=2,controller_steps=2,solver_seconds=.05)
    manifest=dict(schema_version=1,pipeline_seeds=[args.seed],splits=splits,
                  actions=method.ACTION_NAMES,arms=method.ARMS,max_n=64,
                  evaluation_steps=1 if args.smoke else 4,solver_seconds=settings.solver_seconds,
                  proposer_steps=settings.proposer_steps,feedback_steps=settings.feedback_steps,
                  controller_steps=settings.controller_steps,blocked_reversal_keys=data['excluded_reversal_keys'],
                  primary_comparators=['fixed10','static_length'],domain='complete reachable coordinate domains; origin/+x anchor',
                  scope='Separate fresh-sequence call-budget replication; not equal elapsed-time CPU comparison',
                  encoder_revision=hp.REVISION,api_calls=0,torch_threads=args.threads,
                  single_worker_cp=True,smoke=args.smoke,compute_end_epoch=args.end_epoch)
    save(out/'manifest.json',manifest)
    result=dict(status='running',completed_primary_cases=0,completed_paper_cases=0,completed_seeds=[])
    try:
        embedding,cost=encoder.encode_sequences([s for rows in splits.values() for s in rows],out,max_n=64)
        torch.set_num_threads(args.threads)  # The original encoder helper sets four.
        result['encoder']=cost
        pipeline=method.seed_pipeline(pilot,splits['train'],embedding,args.seed,settings)
        pipeline['embedding']=embedding; result['completed_seeds']=[args.seed]
        grouped=[[s for s in splits['test'] if len(s)==n] for n in sorted({len(s) for s in splits['test']})]
        ordered=[s for chunk in zip(*grouped) for s in chunk]
        for i,seq in enumerate(ordered):
            method.evaluate_case(pilot,pipeline,seq,args.seed,i,manifest['evaluation_steps'],settings.solver_seconds,'primary')
            result['completed_primary_cases']+=1;save(out/'results.json',result)
        unchanged=method.state_hash(pipeline['proposer'])==pipeline['training']['proposer_state_hash']
        if not unchanged: raise RuntimeError('Test evaluation changed proposer weights')
        pilot.event(dict(type='freeze_check',seed=args.seed,proposer_weights_unchanged=unchanged))
        result['status']='complete'
    except TimeoutError as exc:
        result.update(status='deadline_partial',reason=str(exc))
    finally:
        result.update(solver_calls=pilot.solver_calls,core_wall_s=time.perf_counter()-pilot.started)
        pilot.event(dict(type='run_end',status=result['status'],pending_calls=sorted(pilot.pending_calls),
                         completed_cases=pilot.completed_cases,solver_calls=pilot.solver_calls))
        pilot.events.close();save(out/'results.json',result)
    # Independent witness check uses neither the method's counter nor its checker.
    checked=0
    for line in (out/'events.jsonl').open():
        row=json.loads(line)
        if row['type']=='solver':
            for field in ('positions','solver_positions'):
                if row.get(field) is not None:
                    actual=contacts(row['seq'],row[field])
                    expected=row.get('actual_contacts') if field=='positions' else row.get('objective')
                    if actual!=expected: raise ValueError('DecisionBoost independent witness mismatch')
                    checked+=1
    save(out/'geometry_verification.json',dict(ok=True,checked_witnesses=checked,
                                             scope='Complete fold geometry and contacts; not full policy/RNG protocol audit'))
    recorder.emit(dict(type='decisionboost_complete',status=result['status'],completed_cases=result['completed_primary_cases'],
                       domain=manifest['domain'],scope=manifest['scope']))
    return dict(decisionboost_status=result['status'],completed_cases=result['completed_primary_cases'],
                full_cube_optimal=False,primary_comparison=False)
