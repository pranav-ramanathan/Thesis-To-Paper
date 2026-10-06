"""Cheap, fixed-sequence DecisionBoost pilot with a frozen open encoder.

No Jev API, CP-SAT, RL or proprietary checkpoint is used. Trains a typed
Choice transformer head on randomised growth / suffix-regrowth elites.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import random
import statistics
import time

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
MAX_N = 20
DIRECTIONS = [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]
ENCODER = "answerdotai/ModernBERT-large"
REVISION = "45bb4654a4d5aaff24dd11d4781fa46d39bf8c13"


def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n'); temp.replace(path)


def manhattan(a,b): return sum(abs(x-y) for x,y in zip(a,b))


def score(seq, pos):
    return sum(seq[i] == seq[j] == 'H' and manhattan(pos[i],pos[j]) == 1
               for i in range(len(pos)) for j in range(i+2,len(pos)))


def check(seq,pos,complete=True):
    assert 2 <= len(pos) <= len(seq)
    assert len(set(map(tuple,pos))) == len(pos)
    assert all(len(p)==3 and all(isinstance(x,int) for x in p) for p in pos)
    assert all(manhattan(a,b)==1 for a,b in zip(pos,pos[1:]))
    assert tuple(pos[0])==(0,0,0) and tuple(pos[1])==(1,0,0)
    if complete: assert len(pos)==len(seq)


def candidates(seq,pos):
    occupied=set(pos); i=len(pos); options=[]
    for action,d in enumerate(DIRECTIONS):
        p=tuple(x+y for x,y in zip(pos[-1],d))
        if p in occupied: continue
        gain=sum(seq[i]==seq[j]=='H' and manhattan(p,pos[j])==1 for j in range(i-1))
        free=sum(tuple(x+y for x,y in zip(p,dd)) not in occupied for dd in DIRECTIONS)
        options.append((action,p,gain,free))
    return options


def grow(seq,rng,prefix=None):
    pos=list(prefix or [(0,0,0),(1,0,0)])
    while len(pos)<len(seq):
        options=candidates(seq,pos)
        if not options:return None
        best=max(c[2] for c in options)
        choice=rng.choice([c for c in options if c[2]==best]) if rng.random()>.15 else rng.choice(options)
        pos.append(choice[1])
    check(seq,pos);return pos


def transform(pos,kind):
    swap=kind//4; sy=-1 if kind%4//2 else 1; sz=-1 if kind%2 else 1
    return [(x,sy*(z if swap else y),sz*(y if swap else z)) for x,y,z in pos]


def canonical(pos):return min(tuple(transform(pos,k)) for k in range(8))


def update_archive(seq,archive,new,limit=20):
    pool={canonical(p):score(seq,p) for p in archive}
    for p in new:
        if p is not None:
            check(seq,p);pool[canonical(p)]=score(seq,p)
    # Stable spatial tie-break; symmetries are collapsed before retaining elites.
    return [list(p) for p,q in sorted(pool.items(),key=lambda item:(-item[1],item[0]))[:limit]]


def improve(seq,pos,rng):
    if pos is None:return grow(seq,rng)
    check(seq,pos)
    q=grow(seq,rng,pos[:rng.randrange(2,len(seq))])
    if q is not None and score(seq,q)>score(seq,pos):return q
    return pos


def corpus():
    rng=random.Random(20260930); splits={'train':[],'validation':[],'test':[],'ood':[]}; seen=set()
    for n in (8,12,16,20):
        # n=8 has only 126 reversal classes with 2..6 H residues.
        counts={'train':64 if n==8 else 100,'validation':12,'test':20} if n<20 else {'ood':20}
        possible=[''.join(s) for s in itertools.product('HP',repeat=n)] if n==8 else None
        if possible:
            possible=[s for s in possible if 2<=s.count('H')<=n-2];rng.shuffle(possible)
        for split,count in counts.items():
            added=0
            while added<count:
                seq=possible.pop() if possible is not None else ''.join(rng.choice('HP') for _ in range(n))
                if not 2<=seq.count('H')<=n-2:continue
                key=min(seq,seq[::-1])
                if key in seen:continue
                seen.add(key);splits[split].append(seq);added+=1
    assert len({min(s,s[::-1]) for v in splits.values() for s in v})==sum(map(len,splits.values()))
    return splits


def features(seq,pos):
    n=len(seq); i=len(pos)
    geo=np.zeros((MAX_N,8),dtype=np.float32)
    for j in range(n):
        if j<i:geo[j,:3]=np.asarray(pos[j])/MAX_N
        geo[j,3:]=[j<i,j==i,seq[j]=='H',j/MAX_N,n/MAX_N]
    candidate=np.zeros((6,15),dtype=np.float32);mask=np.zeros(6,dtype=np.bool_)
    centre=np.mean(pos,axis=0);current=score(seq,pos)
    for action,p,gain,free in candidates(seq,pos):
        mask[action]=True
        candidate[action]=list(DIRECTIONS[action])+list(np.asarray(p)/MAX_N)+[
            gain/4,free/6,seq[i]=='H',(n-i)/MAX_N]+list((np.asarray(p)-centre)/MAX_N)+[current/MAX_N,1]
    return geo,candidate,mask


def teacher_examples(sequences,archives):
    examples=[]
    for si,seq in enumerate(sequences):
        groups={}
        for fold in archives[seq]:
            # Eight lattice symmetries leave the fixed +x first bond unchanged.
            for kind in range(8):
                pos=transform(fold,kind)
                for i in range(2,len(seq)):
                    prefix=tuple(pos[:i]);delta=tuple(x-y for x,y in zip(pos[i],pos[i-1]))
                    label=DIRECTIONS.index(delta)
                    if prefix not in groups:groups[prefix]=np.zeros(6,dtype=np.float32)
                    groups[prefix][label]+=1
        for prefix,counts in groups.items():
            geo,cand,mask=features(seq,list(prefix));target=counts/counts.sum()
            assert np.all(target[~mask]==0)
            examples.append((si,geo,cand,mask,target))
    return {'indices':torch.tensor([e[0] for e in examples]),
            'geo':torch.from_numpy(np.stack([e[1] for e in examples])),
            'candidates':torch.from_numpy(np.stack([e[2] for e in examples])),
            'mask':torch.from_numpy(np.stack([e[3] for e in examples])),
            'targets':torch.from_numpy(np.stack([e[4] for e in examples]))}


class ChoiceHead(nn.Module):
    def __init__(self,encoder_dim):
        super().__init__()
        self.sequence=nn.Linear(encoder_dim,64)
        self.geometry=nn.Linear(8,64)
        self.candidate=nn.Linear(15,64)
        layer=nn.TransformerEncoderLayer(64,4,128,dropout=0,batch_first=True,norm_first=True)
        self.transformer=nn.TransformerEncoder(layer,2,enable_nested_tensor=False)
        self.output=nn.Linear(64,1)

    def forward(self,encoded,lengths,geo,candidates,mask):
        state=self.sequence(encoded)+self.geometry(geo)
        choice=self.candidate(candidates)
        tokens=torch.cat([state,choice],dim=1)
        padding=torch.cat([torch.arange(MAX_N)[None,:]>=lengths[:,None],~mask],dim=1)
        x=self.transformer(tokens,src_key_padding_mask=padding)
        return self.output(x[:,MAX_N:]).squeeze(-1).masked_fill(~mask,-1e9)


def embed(splits,out):
    path=out/'encoder_features.pt'
    sequences=[s for values in splits.values() for s in values]
    if path.exists():
        d=torch.load(path,weights_only=True)
        assert d['sequences']==sequences and d['revision']==REVISION
        return {s:d['features'][i] for i,s in enumerate(sequences)},d['metadata']
    from transformers import AutoModel,AutoTokenizer
    st=time.perf_counter()
    tokenizer=AutoTokenizer.from_pretrained(ENCODER,revision=REVISION)
    model=AutoModel.from_pretrained(ENCODER,revision=REVISION,attn_implementation='sdpa',reference_compile=False)
    model.eval().requires_grad_(False)
    result=[]
    with torch.inference_mode():
        for start in range(0,len(sequences),16):
            batch=sequences[start:start+16]
            inputs=tokenizer([list(s) for s in batch],is_split_into_words=True,return_tensors='pt',padding=True,truncation=False)
            outputs=model(**inputs).last_hidden_state
            for j,seq in enumerate(batch):
                word_ids=inputs.word_ids(j)
                tensor=torch.zeros(MAX_N,outputs.shape[-1])
                for k in range(len(seq)):
                    loc=[i for i,w in enumerate(word_ids) if w==k]
                    assert loc,'Missing residue tokenizer coverage'
                    tensor[k]=outputs[j,loc].mean(0)
                # Unit-scale per-residue features stabilise the fresh projection/head.
                tensor=tensor/torch.sqrt(tensor.square().mean(-1,keepdim=True)+1e-8)
                result.append(tensor)
            print(f'encoder: {min(start+16,len(sequences))}/{len(sequences)} sequences',flush=True)
    features=torch.stack(result)
    metadata={'model':ENCODER,'revision':REVISION,'parameters':sum(p.numel() for p in model.parameters()),
              'encoding_and_load_s':time.perf_counter()-st,'frozen':True,
              'input':'full HP sequence as spaced residue words; one cached token feature per residue',
              'head_input':'frozen sequence features plus explicit partial geometry and candidate features'}
    torch.save({'sequences':sequences,'features':features,'revision':REVISION,'metadata':metadata},path)
    return {s:features[i] for i,s in enumerate(sequences)},metadata


def train(model,optim,data,sequences,embedding,steps,seed):
    st=time.perf_counter();model.train();g=torch.Generator().manual_seed(seed)
    enc=torch.stack([embedding[s] for s in sequences]);lengths=torch.tensor(list(map(len,sequences)))
    losses=[]
    for k in range(steps):
        sample=torch.randint(len(data['indices']),(128,),generator=g);si=data['indices'][sample]
        logits=model(enc[si],lengths[si],data['geo'][sample],data['candidates'][sample],data['mask'][sample])
        loss=-(data['targets'][sample]*logits.log_softmax(-1)).sum(-1).mean()
        optim.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1);optim.step()
        losses.append(float(loss.detach()))
    return {'steps':steps,'examples':len(data['indices']),'seconds':time.perf_counter()-st,
            'first_20_loss':statistics.mean(losses[:20]),'last_20_loss':statistics.mean(losses[-20:])}


@torch.inference_mode()
def sample_folds(model,sequences,embedding,attempts,seed):
    st=time.perf_counter();model.eval();rng=np.random.default_rng(seed);out={s:[] for s in sequences}
    jobs=[s for s in sequences for _ in range(attempts)]
    for begin in range(0,len(jobs),256):
        batch=jobs[begin:begin+256];positions=[[(0,0,0),(1,0,0)] for _ in batch]
        dead=set()
        for t in range(2,max(map(len,batch))):
            active=[i for i,s in enumerate(batch) if i not in dead and len(positions[i])<len(s)]
            if not active:break
            states=[features(batch[i],positions[i]) for i in active]
            live=[j for j,e in enumerate(states) if e[2].any()]
            for j,e in enumerate(states):
                if not e[2].any():dead.add(active[j])
            if not live:continue
            ai=[active[j] for j in live]
            geo=torch.from_numpy(np.stack([states[j][0] for j in live]));cand=torch.from_numpy(np.stack([states[j][1] for j in live]))
            masks=torch.from_numpy(np.stack([states[j][2] for j in live]))
            probs=model(torch.stack([embedding[batch[i]] for i in ai]),torch.tensor([len(batch[i]) for i in ai]),geo,cand,masks).softmax(-1).numpy()
            for row,i in enumerate(ai):
                # Fixed 10% uniform exploration, declared identically in both model arms.
                p=.9*probs[row]+.1*masks[row].numpy()/masks[row].sum().item();p=p/p.sum()
                action=int(rng.choice(6,p=p));d=DIRECTIONS[action]
                positions[i].append(tuple(x+y for x,y in zip(positions[i][-1],d)))
        for i,seq in enumerate(batch):
            if i in dead:out[seq].append(None)
            else:check(seq,positions[i]);out[seq].append(positions[i])
    return out,time.perf_counter()-st


def initial_archives(sequences,seed):
    rng=random.Random(seed);out={};st=time.perf_counter();complete=0
    for seq in sequences:
        folds=[grow(seq,rng) for _ in range(200)];complete+=sum(p is not None for p in folds)
        archive=update_archive(seq,[],folds)
        for _ in range(200):
            p=improve(seq,rng.choice(archive),rng)
            archive=update_archive(seq,archive,[p])
        out[seq]=archive
    return out,{'seconds':time.perf_counter()-st,'growth_attempts':200*len(sequences),
                'regrowth_attempts':200*len(sequences),'complete_growth':complete}


def model_round(model,sequences,embedding,archive,seed):
    proposals,inference_s=sample_folds(model,sequences,embedding,40,seed)
    rng=random.Random(seed);st=time.perf_counter();out={}
    for seq in sequences:
        improved=[improve(seq,p,rng) for p in proposals[seq]]
        out[seq]=update_archive(seq,archive[seq],improved)
    return out,{'inference_s':inference_s,'improvement_s':time.perf_counter()-st,
                'proposals':40*len(sequences),'regrowth_attempts':40*len(sequences),
                'proposal_completion':sum(p is not None for v in proposals.values() for p in v)/sum(map(len,proposals.values()))}


def evaluate(model,sequences,embedding,seed,method,outpath,matched_allowances=None):
    rows=[];all_started=time.perf_counter()
    if model is not None:
        folds,inf_s=sample_folds(model,sequences,embedding,32,seed)
        inference_per_seq=inf_s/len(sequences)
    else:folds={};inference_per_seq=0
    for si,seq in enumerate(sequences):
        rng=random.Random(seed+si);st=time.perf_counter()
        proposals=folds[seq] if model is not None else [grow(seq,rng) for _ in range(32)]
        complete=[p for p in proposals if p is not None]
        raw=[score(seq,p) for p in complete]
        improved=[improve(seq,p,rng) for p in proposals]
        completed=[p for p in improved if p is not None]
        final=[score(seq,p) for p in completed]
        search_s=time.perf_counter()-st
        row={'sequence':seq,'length':len(seq),'method':method,'attempts':32,
             'raw_complete':len(complete),'raw_mean_complete':statistics.mean(raw) if raw else None,
             'raw_mean_zero_for_failures':sum(raw)/32,'raw_best':max(raw,default=0),
             'improved_complete':len(completed),'improved_mean_zero_for_failures':sum(final)/32,
             'improved_best':max(final,default=0),'inference_s':inference_per_seq,'search_s':search_s,
             'raw_folds':proposals,'improved_folds':improved}
        if matched_allowances is not None:
            # Additional heuristic restarts/regrowth for the model's allocated serial time.
            target=matched_allowances[seq];best=max(final,default=0);count=32;hst=time.perf_counter()
            while time.perf_counter()-hst<target:
                p=grow(seq,rng);q=improve(seq,p,rng);count+=1
                if q is not None:best=max(best,score(seq,q))
            row.update({'matched_extra_budget_s':target,'matched_extra_actual_s':time.perf_counter()-hst,
                        'matched_total_attempts':count,'matched_best':best})
        rows.append(row)
    result={'method':method,'rows':rows,'wall_s':time.perf_counter()-all_started}
    save(outpath,result);return result


def aggregate(result):
    out={}
    for n in sorted({r['length'] for r in result['rows']}):
        rows=[r for r in result['rows'] if r['length']==n]
        out[str(n)]={'instances':len(rows),'raw_completion':sum(r['raw_complete'] for r in rows)/(32*len(rows)),
            'raw_best_mean':statistics.mean(r['raw_best'] for r in rows),
            'raw_mean_zero_for_failures':statistics.mean(r['raw_mean_zero_for_failures'] for r in rows),
            'improved_best_mean':statistics.mean(r['improved_best'] for r in rows),
            'improved_completion':sum(r['improved_complete'] for r in rows)/(32*len(rows))}
        if 'matched_best' in rows[0]:out[str(n)]['matched_best_mean']=statistics.mean(r['matched_best'] for r in rows)
    return out


@torch.inference_mode()
def decision_metrics(model,sequences,embedding,archives,seed):
    """Held-out matching/calibration against empirical elite teacher choices."""
    model.eval();out={}
    for n in sorted(set(map(len,sequences))):
        seqs=[s for s in sequences if len(s)==n]
        data=teacher_examples(seqs,archives);g=torch.Generator().manual_seed(seed+n)
        sample=torch.randperm(len(data['indices']),generator=g)[:5000]
        enc=torch.stack([embedding[s] for s in seqs]);lengths=torch.tensor(list(map(len,seqs)))
        probs=[];targets=[];heuristic_probs=[]
        for begin in range(0,len(sample),256):
            rows=sample[begin:begin+256];si=data['indices'][rows];mask=data['mask'][rows]
            logits=model(enc[si],lengths[si],data['geo'][rows],data['candidates'][rows],mask)
            probs.append(logits.softmax(-1));targets.append(data['targets'][rows])
            gain=data['candidates'][rows,:,6].masked_fill(~mask,-1)
            tied=(gain==gain.max(-1,keepdim=True).values)&mask
            heuristic_probs.append(.85*tied/tied.sum(-1,keepdim=True)+.15*mask/mask.sum(-1,keepdim=True))
        p=torch.cat(probs);y=torch.cat(targets);h=torch.cat(heuristic_probs)
        selected=p.argmax(-1);confidence=p.max(-1).values;agreement=y.gather(1,selected[:,None]).squeeze(1)
        reliability=[];ece=0.
        for k in range(10):
            ids=(confidence>=k/10)&(confidence<(k+1)/10 if k<9 else confidence<=1)
            if ids.any():
                cp=float(confidence[ids].mean());ap=float(agreement[ids].mean());count=int(ids.sum())
                ece+=count/len(p)*abs(cp-ap)
                reliability.append({'lower':k/10,'upper':(k+1)/10,'count':count,
                                    'mean_confidence':cp,'teacher_agreement':ap})
        out[str(n)]={'states':len(p),'teacher_choice_ce':float(-(y*p.clamp_min(1e-9).log()).sum(-1).mean()),
            'teacher_choice_brier':float((p-y).square().sum(-1).mean()),
            'top_choice_teacher_agreement':float(agreement.mean()),'teacher_choice_ece_10bins':ece,
            'heuristic_teacher_choice_ce':float(-(y*h.clamp_min(1e-9).log()).sum(-1).mean()),
            'heuristic_teacher_choice_brier':float((h-y).square().sum(-1).mean()),'reliability':reliability}
    return out


def make_report(out,runs,metadata,args):
    lines=['# Cheap DecisionBoost HP pilot','',
           'Fixed-sequence 3D HP folding. Higher non-chain HH contacts are better. No CP-SAT, paid API, environment RL or RLCD policy-gradient update was used.','',
           f"Frozen encoder: `{ENCODER}` at `{REVISION}`. Encoder parameters: {metadata['parameters']:,}. Only the fresh typed Choice transformer head is trained.",'',
           '## Held-out results','',
           '| Length | Method | Mean best raw contacts | Mean best after regrowth | Raw completion |',
           '|---|---|---:|---:|---:|']
    summary={'encoder':metadata,'runs':runs,'settings':vars(args).copy()};summary['settings']['out']=str(args.out)
    for n in ('8','12','16','20'):
        for method in ('heuristic','one_shot','decisionboost'):
            values=[r['evaluation'][method][n] for r in runs]
            lines.append(f"| {n} | {method} | {statistics.mean(v['raw_best_mean'] for v in values):.2f} | {statistics.mean(v['improved_best_mean'] for v in values):.2f} | {statistics.mean(v['raw_completion'] for v in values):.1%} |")
    lines+=['','Each row averages sequence-level best-of-32 results across independent training seeds. Length 20 is out of the training curriculum.','',
            '## Time-matched heuristic','', '| Length | Heuristic best contacts with extra model-time allowance |','|---|---:|']
    for n in ('8','12','16','20'):
        lines.append(f"| {n} | {statistics.mean(r['evaluation']['heuristic'][n]['matched_best_mean'] for r in runs):.2f} |")
    lines+=['','The heuristic gets its own 32 attempts plus extra restart/regrowth time equal to DecisionBoost\'s batched inference time apportioned uniformly per sequence and its improvement time. This deliberately favours the heuristic slightly. It is an approximate serial inference/search budget comparison; model training and initial teacher cost are additional and separately logged.','',
            '## Experimental controls','',
            '- Train/validation/test splits exclude both duplicate sequences and reversed-sequence duplicates. Length 8 uses 64 training sequences because the allowed sequence space has only 126 reversal classes; lengths 12 and 16 use 100 each. Every training stage also replays earlier lengths.',
            '- Both model arms use identical initial weights, optimiser, initial teacher folds, curriculum and update counts. One-shot training uses the fixed initial archive; DecisionBoost adds elite folds from three model/search feedback rounds per length.',
            '- Both arms generate 40 proposals and apply 40 regrowth attempts per training sequence per feedback round. One-shot proposals are logged but excluded from its subsequent training data.',
            '- The initial teacher uses 200 growth and 200 regrowth attempts per sequence, with 20 canonical elites retained. Teacher archives are shared between paired arms.',
            '- Each evaluation arm gets 32 construction attempts and one suffix-regrowth attempt per proposal. Dead ends are recorded; a dead proposal\'s improvement call uses a fresh heuristic growth and is counted.',
            '- The head sees full-sequence frozen encoder features, explicit partial geometry and exact candidate features including immediate contact gain. Cheap deterministic legality is enforced before sampling.',
            '- The loss is soft-target cross-entropy against elite teacher-choice frequencies. Held-out teacher-choice CE/Brier/reliability are saved in each seed result. They measure agreement with empirical heuristic elites, not optimum-action probabilities or folding-success calibration. This is a proper-scoring-loss baseline, not a reproduced Jev RLCD training algorithm.',
            '- Predeclared curriculum runs 8, 12 and 16 for diagnosis; larger-length evaluation is not evidence that promotion gates were met. No hyperparameters are tuned using the final test set.',
            '- Geometry and objective consistency are checked on every saved complete fold. No optimality claim follows from these heuristic results.','',
            '## Costs and reproducibility','']
    for r in runs:
        lines.append(f"- Seed {r['seed']}: head parameters {r['head_parameters']:,}; total run {r['wall_s']:.1f}s; training DecisionBoost {r['training_seconds']['decisionboost']:.1f}s, one-shot {r['training_seconds']['one_shot']:.1f}s.")
    lines+=['',f"Encoder download/loading/feature preparation: {metadata['encoding_and_load_s']:.1f}s (shared by all arms/seeds).",'',
            'See manifest.json, summary.json, per-seed logs/checkpoints/archives, and evaluation JSON files with full coordinates. API spend: $0.','']
    save(out/'summary.json',summary);(out/'report.md').write_text('\n'.join(lines))


def self_test():
    splits=corpus();assert len(splits['train'])==264
    seq='HHHH';fold=[(0,0,0),(1,0,0),(1,1,0),(0,1,0)]
    assert score(seq,fold)==1
    check(seq,fold)
    for kind in range(8):assert score(seq,transform(fold,kind))==1
    rng=random.Random(42)
    for n in (8,12,16,20):
        seq='HP'*(n//2)
        for _ in range(30):
            p=grow(seq,rng)
            if p is not None:
                check(seq,p);q=improve(seq,p,rng);check(seq,q);assert score(seq,q)>=score(seq,p)
    seq='HPHHPHPH';p=grow(seq,rng)
    data=teacher_examples([seq],{seq:[p]})
    model=ChoiceHead(16);enc=torch.randn(1,MAX_N,16);ln=torch.tensor([8]);sample=slice(0,3)
    logits=model(enc.expand(3,-1,-1),ln.expand(3),data['geo'][sample],data['candidates'][sample],data['mask'][sample])
    probs=logits.softmax(-1);assert torch.all(probs[~data['mask'][sample]]==0)
    assert torch.allclose(probs.sum(-1),torch.ones(3))
    loss=-(data['targets'][sample]*logits.log_softmax(-1)).sum(-1).mean();loss.backward()
    assert math.isfinite(float(loss.detach()))
    print('Geometry, monotonic improvement, symmetry, split, teacher-target and masked-gradient checks passed',flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,default=ROOT/'outputs/decisionboost_cheap_run')
    parser.add_argument('--seeds',type=int,nargs='+',default=[0,1,2])
    parser.add_argument('--bootstrap-steps',type=int,default=128);parser.add_argument('--round-steps',type=int,default=64)
    parser.add_argument('--encode-only',action='store_true');parser.add_argument('--self-test',action='store_true')
    args=parser.parse_args();torch.set_num_threads(4);torch.set_num_interop_threads(1)
    if args.self_test:self_test();return
    args.out.mkdir(parents=True,exist_ok=True);splits=corpus()
    manifest={'splits':splits,'seeds':args.seeds,'bootstrap_steps':args.bootstrap_steps,'round_steps':args.round_steps,
              'encoder':ENCODER,'revision':REVISION,'rounds_per_length':3,'torch':torch.__version__,
              'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'training_loss':'soft-target cross-entropy',
              'generation_exploration':.1,'growth_random_exploration':.15,'training_head':'2-layer 64-width 4-head bidirectional transformer',
              'test_attempts':32,'initial_growth':200,'initial_regrowth':200,'feedback_proposals':40,
              'curriculum':[8,12,16],'held_out_length':20,'api_calls':0}
    save(args.out/'manifest.json',manifest)
    embeddings,metadata=embed(splits,args.out)
    if args.encode_only:print(json.dumps(metadata),flush=True);return
    runs=[]
    for seed in args.seeds:
        seedout=args.out/f'seed_{seed}';seedout.mkdir(exist_ok=True)
        if (seedout/'result.json').exists():runs.append(json.loads((seedout/'result.json').read_text()));continue
        runst=time.perf_counter();torch.manual_seed(seed);np.random.seed(seed)
        initial,teacher_cost=initial_archives(splits['train'],1000+seed)
        save(seedout/'initial_archive.json',initial)
        boost=ChoiceHead(next(iter(embeddings.values())).shape[-1]);one=copy.deepcopy(boost)
        optimizers={name:torch.optim.AdamW(m.parameters(),lr=1e-3,weight_decay=.01) for name,m in [('decisionboost',boost),('one_shot',one)]}
        archives={'decisionboost':copy.deepcopy(initial),'one_shot':copy.deepcopy(initial)}
        logs=[];training_seconds={'decisionboost':0.,'one_shot':0.}
        for stage,n in enumerate((8,12,16)):
            sequences=[s for s in splits['train'] if len(s)<=n]
            for name,model in [('decisionboost',boost),('one_shot',one)]:
                data=teacher_examples(sequences,archives[name]);cost=train(model,optimizers[name],data,sequences,embeddings,args.bootstrap_steps,seed*10000+stage*100)
                training_seconds[name]+=cost['seconds'];logs.append({'stage':n,'round':0,'method':name,'train':cost})
                print(f'seed {seed} length {n} {name} bootstrap loss {cost["last_20_loss"]:.3f}, {cost["seconds"]:.1f}s',flush=True)
            for round_id in range(1,4):
                for name,model in [('decisionboost',boost),('one_shot',one)]:
                    updated,search_cost=model_round(model,sequences,embeddings,archives[name],seed*10000+stage*100+round_id)
                    if name=='decisionboost':archives[name].update(updated)
                    data=teacher_examples(sequences,archives[name]);cost=train(model,optimizers[name],data,sequences,embeddings,args.round_steps,seed*10000+stage*100+round_id)
                    training_seconds[name]+=cost['seconds'];logs.append({'stage':n,'round':round_id,'method':name,'train':cost,'search':search_cost})
                    torch.save({'head':model.state_dict(),'encoder_model':ENCODER,'encoder_revision':REVISION,'length':n,'round':round_id,'seed':seed},seedout/f'{name}_n{n}_round{round_id}.pt')
                    print(f'seed {seed} length {n} round {round_id} {name}: loss {cost["last_20_loss"]:.3f}, proposal completion {search_cost["proposal_completion"]:.1%}',flush=True)
                save(seedout/'training_log.json',logs)
            # Validation is observational; no tuning or checkpoint selection on these values.
            val_sequences=[s for s in splits['validation'] if len(s)==n]
            val=evaluate(boost,val_sequences,embeddings,90000+seed*100+n,'validation',seedout/f'validation_n{n}.json')
            print(f'seed {seed} length {n} validation {aggregate(val)}',flush=True)
        evaluation={};test_sequences=splits['test']+splits['ood']
        br=evaluate(boost,test_sequences,embeddings,80000+seed,'decisionboost',seedout/'evaluation_decisionboost.json')
        evaluation['decisionboost']=aggregate(br)
        orr=evaluate(one,test_sequences,embeddings,80000+seed,'one_shot',seedout/'evaluation_one_shot.json');evaluation['one_shot']=aggregate(orr)
        allowances={r['sequence']:r['inference_s']+r['search_s'] for r in br['rows']}
        hr=evaluate(None,test_sequences,embeddings,80000+seed,'heuristic',seedout/'evaluation_heuristic.json',allowances);evaluation['heuristic']=aggregate(hr)
        heldout_archives,heldout_teacher_cost=initial_archives(test_sequences,70000+seed)
        save(seedout/'heldout_teacher_archive.json',heldout_archives)
        metrics={name:decision_metrics(model,test_sequences,embeddings,heldout_archives,60000+seed)
                 for name,model in [('decisionboost',boost),('one_shot',one)]}
        save(seedout/'final_archive.json',archives['decisionboost'])
        result={'seed':seed,'head_parameters':sum(p.numel() for p in boost.parameters()),'evaluation':evaluation,
                'teacher':teacher_cost,'training_seconds':training_seconds,'decision_metrics':metrics,
                'heldout_teacher_cost':heldout_teacher_cost,'wall_s':time.perf_counter()-runst}
        save(seedout/'result.json',result);runs.append(result);make_report(args.out,runs,metadata,args)
        print(f'Seed {seed} complete after {result["wall_s"]:.1f}s',flush=True)
    make_report(args.out,runs,metadata,args);print(f'Report: {args.out / "report.md"}',flush=True)


if __name__=='__main__':main()
