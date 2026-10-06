"""Bounded, offline DecisionBoost pilot: cheap proposals -> CP repair -> learning.

Reuses only cached frozen sequence embeddings, not trained policies. No API,
network, sibling imports, production results, or full training campaign.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import random
import statistics
import time
from pathlib import Path

import torch
import ortools
from ortools.sat.python import cp_model
import cheap_hp as hp


class Pilot:
    def __init__(self, out, wall_limit):
        self.out = out
        out.mkdir(parents=True, exist_ok=False)
        self.started = time.perf_counter()
        self.deadline = self.started + wall_limit
        self.events = (out / 'events.jsonl').open('w')
        self.solver_calls = 0

    def event(self, record):
        self.events.write(json.dumps(record) + '\n')
        self.events.flush()

    def remaining(self):
        return self.deadline - time.perf_counter()

    def solve(self, seq, *, phase, seed, seconds=1.0, proposal=None,
              frozen_indices=(), prefix=None):
        if self.remaining() < seconds + 10:
            raise TimeoutError('Bounded pilot deadline; no new solver calls')
        st = time.perf_counter()
        n, radix = len(seq), 2 * len(seq) - 1
        model = cp_model.CpModel()
        xyz = [[model.new_int_var(-i, i, f'r{i}_{a}') for a in range(3)] for i in range(n)]
        cells = [model.new_int_var(0, radix ** 3 - 1, f'site{i}') for i in range(n)]
        for i in range(n):
            model.add(cells[i] == sum((xyz[i][a] + n - 1) * radix ** a for a in range(3)))
        model.add_all_different(cells)
        for a, value in enumerate((1, 0, 0)):
            model.add(xyz[1][a] == value)

        def distance(i, j):
            ds = [model.new_int_var(0, i + j, f'd{i}_{j}_{a}') for a in range(3)]
            for a in range(3):
                model.add_abs_equality(ds[a], xyz[i][a] - xyz[j][a])
            return sum(ds)

        for i in range(n - 1):
            model.add(distance(i, i + 1) == 1)
        contacts = []
        for i in range(n):
            for j in range(i + 3, n, 2):
                if seq[i] == seq[j] == 'H':
                    c = model.new_bool_var(f'c{i}_{j}')
                    d = distance(i, j)
                    model.add(d == 1).only_enforce_if(c)
                    model.add(d >= 2).only_enforce_if(c.Not())
                    contacts.append(c)
        model.maximize(sum(contacts))
        hint = proposal if proposal is not None else prefix
        if hint is not None:
            hp.check(seq, hint, complete=proposal is not None)
            for i, p in enumerate(hint):
                for a in range(3):
                    model.add_hint(xyz[i][a], p[a])
                model.add_hint(cells[i], sum((p[a] + n - 1) * radix ** a for a in range(3)))
        incumbent = hp.score(seq, proposal) if proposal is not None else None
        if incumbent is not None:
            model.add(sum(contacts) >= incumbent)
        for i in frozen_indices:
            for a in range(3):
                model.add(xyz[i][a] == proposal[i][a])
        if prefix is not None:
            for i, p in enumerate(prefix):
                for a in range(3):
                    model.add(xyz[i][a] == p[a])
        build_s = time.perf_counter() - st
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = seconds
        solver.parameters.num_search_workers = 1
        solver.parameters.random_seed = seed
        solve_start = time.perf_counter()
        status = solver.solve(model)
        solve_s = time.perf_counter() - solve_start
        feasible = status in (cp_model.FEASIBLE, cp_model.OPTIMAL)
        found = [tuple(solver.value(v) for v in row) for row in xyz] if feasible else None
        objective = round(solver.objective_value) if feasible else None
        if found is not None:
            hp.check(seq, found)
            assert hp.score(seq, found) == objective
            assert all(found[i] == tuple(proposal[i]) for i in frozen_indices)
            assert prefix is None or found[:len(prefix)] == list(map(tuple, prefix))
        kept = incumbent is not None and (found is None or objective < incumbent)
        final = proposal if kept else found
        row = dict(type='solver', seq=seq, phase=phase, proposal=proposal,
                   positions=final, solver_positions=found, frozen_indices=list(frozen_indices),
                   prefix=prefix, status=solver.status_name(status), objective=objective,
                   actual_contacts=hp.score(seq, final) if final is not None else None,
                   kept_incumbent=kept, build_s=build_s, solve_s=solve_s,
                   wall_s=time.perf_counter()-st, budget_s=seconds,
                   bound=solver.best_objective_bound,
                   bound_scope='restricted repair' if frozen_indices or prefix else 'full sequence')
        self.event(row)
        self.solver_calls += 1
        return row

    def repair(self, seq, fold, phase, seed, round_no=0):
        # Six movable residues. Alternate suffix and internal windows; keep both
        # the prefix and distant suffix fixed for the internal-window condition.
        start = len(seq) - 6 if round_no % 2 == 0 else max(2, len(seq)//2 - 3)
        movable = set(range(start, start + 6))
        frozen = [i for i in range(len(seq)) if i not in movable]
        return self.solve(seq, phase=phase, seed=seed, proposal=fold, frozen_indices=frozen)


def best(seq, folds):
    valid = [p for p in folds if p is not None]
    return max(valid, key=lambda p: hp.score(seq, p)) if valid else None


def safe_fallback(seq, seed):
    """A trapped fallback growth must never masquerade as a complete witness."""
    fold = hp.grow(seq, random.Random(seed))
    return fold if fold is not None else [(i, 0, 0) for i in range(len(seq))]


def cheap_archive(sequences, seed):
    st = time.perf_counter()
    rng = random.Random(seed)
    out = {}
    for seq in sequences:
        folds = [hp.grow(seq, rng) for _ in range(16)]
        archive = hp.update_archive(seq, [], folds, limit=8)
        if not archive:
            archive = [safe_fallback(seq, seed)]
        for _ in range(16):
            archive = hp.update_archive(seq, archive, [hp.improve(seq, rng.choice(archive), rng)], limit=8)
        out[seq] = archive
    return out, time.perf_counter()-st


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--wall-limit', type=float, default=900)
    args = parser.parse_args()
    pilot = Pilot(args.out, args.wall_limit)
    torch.set_num_threads(4)
    torch.manual_seed(404)
    source = hp.corpus()
    # Explicit per-length selection, never select instances by observed scores.
    train_seqs = sum(([s for s in source['train'] if len(s)==n][:12] for n in (12,16)), [])
    test_seqs = sum(([s for s in source['test'] if len(s)==n][:4] for n in (12,16)), [])
    ood_seqs = source['ood'][:4]
    splits = dict(train=train_seqs, test=test_seqs, ood=ood_seqs)
    feature_path = hp.ROOT/'outputs/decisionboost_cheap_run/encoder_features.pt'
    cache = torch.load(feature_path, weights_only=True, map_location='cpu')
    assert cache['revision'] == hp.REVISION
    embedding = {s:cache['features'][i] for i,s in enumerate(cache['sequences'])}
    assert all(s in embedding for rows in splits.values() for s in rows)
    manifest = dict(seed=404, wall_limit_s=args.wall_limit, splits=splits,
        scope='Exploratory one-seed small-sequence pilot; reused held-out corpus; no convergence claim',
        encoder=cache['metadata'], encoder_features_reused=True,
        generation='16 contact-biased growth + 16 suffix-regrowth attempts, eight elites per training sequence',
        feedback_rounds=2, proposals_per_sequence=16, gradient_steps_per_stage=64,
        training_arms=['decisionboost','fixed_teacher'], torch_threads=4,
        solver=dict(version=ortools.__version__,workers=1,seconds_per_call=1,movable_residues=6),
        evaluation='16 proposals per sequence then one suffix CP repair; equal proposal/call counts, not total-time-matched',
        api_calls=0, code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    hp.save(args.out/'manifest.json',manifest)
    result = dict(splits=splits,self_tests=[],initial_generation=[],training=[],evaluation={},api_calls=0)
    try:
        for i, seq in enumerate(('HHHH','HHPHHH','HPHHPHHH')):
            row = pilot.solve(seq, phase='self_test', seed=404+i, seconds=3)
            result['self_tests'].append(dict(seq=seq,status=row['status'],objective=row['objective'],
                positions=row['solver_positions'],bound=row['bound']))
        # A prefix is a hard commitment, not merely a hint.
        prefix_seq = train_seqs[0]
        prefix = safe_fallback(prefix_seq, 405)[:6]
        prefix_test = pilot.solve(prefix_seq,phase='prefix_self_test',seed=405,prefix=prefix)
        assert prefix_test['solver_positions'] is not None, 'Prefix completion self-test returned no witness'
        print('Geometry/solver self-tests complete', flush=True)
        # Nine cases, three per length, each measured three times. Heuristic gets
        # the cold CP call's measured total wall time, including model assembly.
        benchmark = sum(([s for s in train_seqs if len(s)==n][:3] for n in (12,16)),[]) + ood_seqs[:3]
        for i, seq in enumerate(benchmark):
            for rep in range(3):
                row = pilot.solve(seq,phase='initial_generation',seed=4400+i*10+rep)
                st=time.perf_counter(); rng=random.Random(4400+i*10+rep); elite=None; attempts=0
                while time.perf_counter()-st < row['wall_s']:
                    q=hp.grow(seq,rng) if elite is None or attempts%2==0 else hp.improve(seq,elite,rng)
                    if q is not None and (elite is None or hp.score(seq,q)>hp.score(seq,elite)):
                        elite=q
                    attempts+=1
                h=dict(type='initial_heuristic',seq=seq,positions=elite,
                       actual_contacts=hp.score(seq,elite),wall_s=time.perf_counter()-st,attempts=attempts)
                pilot.event(h)
                result['initial_generation'].append(dict(seq=seq,repetition=rep,cp=row,heuristic=h))
            print(f'Initial generation n{len(seq)} case {i+1}/{len(benchmark)}',flush=True)
        initial, prep_s = cheap_archive(train_seqs, 404)
        result['initial_archive']=initial;result['archive_preparation_s']=prep_s
        hp.save(args.out/'initial_archive.json',initial)
        data_start=time.perf_counter();initial_data=hp.teacher_examples(train_seqs, initial)
        result['initial_teacher_materialization_s']=time.perf_counter()-data_start
        initial_model=hp.ChoiceHead(next(iter(embedding.values())).shape[-1])
        models={name:copy.deepcopy(initial_model) for name in ('decisionboost','fixed_teacher')}
        opts={name:torch.optim.AdamW(m.parameters(),lr=3e-4,weight_decay=.01) for name,m in models.items()}
        archives={name:copy.deepcopy(initial) for name in models}
        for name in models:
            row=hp.train(models[name],opts[name],initial_data,train_seqs,embedding,64,404)
            result['training'].append(dict(arm=name,stage=0,**row))
        for round_no in range(2):
            for name, model in models.items():
                proposals,inference_s=hp.sample_folds(model,train_seqs,embedding,16,5000+round_no)
                improvements=[];new=copy.deepcopy(archives[name]);st=time.perf_counter()
                for i,seq in enumerate(train_seqs):
                    pilot.event(dict(type='proposals',phase=f'train_{name}_{round_no}',seq=seq,folds=proposals[seq]))
                    fold=best(seq,proposals[seq])
                    # If all model proposals fail, use a disclosed incumbent.
                    fallback=fold is None
                    if fallback: fold=best(seq,archives[name][seq])
                    repaired=pilot.repair(seq,fold,f'train_{name}_{round_no}',6000+round_no*100+i,round_no)
                    improvements.append(dict(seq=seq,before=hp.score(seq,fold),after=repaired['actual_contacts'],fallback=fallback))
                    if name=='decisionboost':
                        new[seq]=hp.update_archive(seq,archives[name][seq],proposals[seq]+[repaired['positions']],limit=8)
                archives[name]=new
                prep_start=time.perf_counter();data=hp.teacher_examples(train_seqs,new)
                data_s=time.perf_counter()-prep_start
                row=hp.train(model,opts[name],data,train_seqs,embedding,64,5100+round_no)
                result['training'].append(dict(arm=name,stage=round_no+1,inference_s=inference_s,
                    feedback_wall_s=prep_start-st,teacher_materialization_s=data_s,improvements=improvements,**row))
                print(f'{name} feedback {round_no+1}: {sum(x["after"]>x["before"] for x in improvements)}/{len(train_seqs)} repaired better',flush=True)
        result['final_archives']=archives
        eval_seqs=test_seqs+ood_seqs
        for name,model in models.items():
            proposals,inference_s=hp.sample_folds(model,eval_seqs,embedding,16,7100)
            rows=[]
            for i,seq in enumerate(eval_seqs):
                pilot.event(dict(type='proposals',phase=f'eval_{name}',seq=seq,folds=proposals[seq]))
                fold=best(seq,proposals[seq]);fallback=fold is None
                if fallback: fold=safe_fallback(seq,7200+i)
                cp=pilot.repair(seq,fold,f'eval_{name}',7300+i)
                rows.append(dict(seq=seq,folds=proposals[seq],raw_best=hp.score(seq,fold),
                    raw_mean=statistics.mean(hp.score(seq,p) for p in proposals[seq] if p is not None) if not fallback else None,
                    complete=sum(p is not None for p in proposals[seq]),fallback=fallback,
                    final_contacts=cp['actual_contacts'],positions=cp['positions'],solver=cp))
            result['evaluation'][name]=rows
            result.setdefault('evaluation_costs',{})[name]=dict(inference_s=inference_s,
                solver_wall_s=sum(r['solver']['wall_s'] for r in rows))
            print(f'Evaluation {name} complete',flush=True)
        rows=[];cold=[]
        st=time.perf_counter()
        for i,seq in enumerate(eval_seqs):
            rng=random.Random(7100+i);folds=[hp.grow(seq,rng) for _ in range(16)]
            pilot.event(dict(type='proposals',phase='eval_heuristic',seq=seq,folds=folds))
            fold=best(seq,folds);fallback=fold is None
            if fallback:fold=safe_fallback(seq,7200+i)
            cp=pilot.repair(seq,fold,'eval_heuristic',7300+i)
            rows.append(dict(seq=seq,folds=folds,raw_best=hp.score(seq,fold),
                raw_mean=statistics.mean(hp.score(seq,p) for p in folds if p is not None),
                complete=sum(p is not None for p in folds),fallback=fallback,final_contacts=cp['actual_contacts'],positions=cp['positions'],solver=cp))
            cold.append(pilot.solve(seq,phase='eval_cold_cp',seed=7300+i))
        result['evaluation']['heuristic']=rows;result['evaluation']['cold_cp']=cold
        result['status']='complete'
        result['total_wall_s']=time.perf_counter()-pilot.started
        result['solver_calls']=pilot.solver_calls
        for name,model in models.items():
            torch.save({'state_dict':model.state_dict(),'manifest':manifest},args.out/f'{name}.pt')
        hp.save(args.out/'results.json',result)
        print(json.dumps(dict(status='complete',wall_s=result['total_wall_s'],out=str(args.out))),flush=True)
    except BaseException as exc:
        result['status']='partial';result['error']=type(exc).__name__+': '+str(exc)
        result['total_wall_s']=time.perf_counter()-pilot.started
        hp.save(args.out/'results.json',result)
        raise
    finally:
        pilot.events.close()


if __name__=='__main__':
    main()
