"""Read-only campaign aggregation with independent geometry checks and coverage."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import statistics
from campaign import file_hash, save, verify
from geometry import contacts


def records(path):
    if not path.exists(): return
    with path.open('rb') as f:
        for line in f:
            try: yield json.loads(line)
            except json.JSONDecodeError:
                if f.read(): raise ValueError(f'Corrupt nonfinal JSONL record: {path}')
                return


def analyse(directory):
    m=verify(directory)
    rows=[]; witnesses=0; secondary=[]; errors=[]; offline=[]
    protocol=json.loads((directory/'bundle/protocol.json').read_text())
    timepoints=protocol['timepoints_seconds']
    for i,task in enumerate(m['tasks']):
        out=directory/'tasks'/f'{i:03d}_{task["arm"]}_{task["seq_id"]}_seed{task["seed"]}'
        launch=json.loads((out/'launch.json').read_text()) if (out/'launch.json').exists() else {}
        result=json.loads((out/'result.json').read_text()) if (out/'result.json').exists() else {}
        scores=[]; scopes=[]; cases=[]; evaluations=[]; last_episode=None; full_search_end=None
        empty_evaluations=0
        try:
            for event in records(out/'events.jsonl'):
                if event['type']=='witness':
                    value=contacts(event['seq'],event['positions'],len(event['seq'])//2)
                    if value!=event['contacts']: raise ValueError('Logged contacts disagree with independent count')
                    scores.append((event['elapsed_s'],value)); witnesses+=1
                if event['type']=='solver_end': scopes.append(event.get('scope'))
                if event['type']=='solver_end' and event.get('scope')=='full_declared_cube':
                    full_search_end={k:event.get(k) for k in ('status','contacts','bound','scope','elapsed_s')}
                if event['type']=='episode' and event['elapsed_s']<=m['run_seconds']:
                    last_episode={k:event.get(k) for k in ('episode','updates','epsilon','loss','elapsed_s')}
                if event['type']=='evaluation':
                    if event.get('episodes',0)==0:
                        empty_evaluations+=1
                    elif event['elapsed_s']<=m['run_seconds']:
                        evaluations.append({k:event.get(k) for k in ('elapsed_s','episode','episodes','completed_folds','mean_complete_contacts','mean_reward')})
            for event in records(out/'decisionboost/events.jsonl'):
                if event['type']=='solver':
                    for field,expected in [('positions',event.get('actual_contacts')),('solver_positions',event.get('objective'))]:
                        if event.get(field) is not None:
                            if contacts(event['seq'],event[field])!=expected: raise ValueError('DecisionBoost witness mismatch')
                            witnesses+=1
                if event['type']=='evaluation_case': cases.append(event)
                if event['type']=='seed_training':
                    offline.append(dict(task=i,seed=task['seed'],engineering_only=launch.get('engineering_only',False),
                                        costs=event))
        except (ValueError,KeyError) as exc:
            errors.append(dict(task=i,error=str(exc)))
        if cases:
            for case in cases:
                learned=case['arms']['learned']['final_contacts']
                secondary.append(dict(seed=task['seed'],sequence=case['seq'],length=len(case['seq']),
                    engineering_only=launch.get('engineering_only',False),
                    learned=learned,fixed10=case['arms']['fixed10']['final_contacts'],
                    static_length=case['arms']['static_length']['final_contacts'],
                    learned_minus_static=learned-case['arms']['static_length']['final_contacts'],
                    online_s=case['arms']['learned']['online_s'],cold_contacts=case['cold_contacts']))
        elapsed=launch.get('total_elapsed_s',result.get('elapsed_s',0))
        optimal=result.get('full_cube_optimal',False)
        trajectory=[]
        for t in timepoints:
            values=[score for when,score in scores if when<=t]
            covered=elapsed>=t or optimal
            trajectory.append(dict(seconds=t,contacts=max(values) if values and covered else None,
                                   observed_contacts=max(values) if values else None,covered=covered))
        rows.append(dict(task=i,**task,status=launch.get('status','not_started'),
                         worker_status=result.get('status'),elapsed_s=elapsed,
                         best_contacts=max((score for _,score in scores),default=None),
                         full_cube_optimal=optimal,solver_scopes=scopes,trajectory=trajectory,
                         episode=result.get('episode'),training_complete=result.get('training_complete'),
                         updates=result.get('updates'),repairs=result.get('repairs'),
                         last_episode_within_budget=last_episode,
                         evaluations_within_budget=evaluations,
                         first_greedy_evaluation=evaluations[0] if evaluations else None,
                         latest_greedy_evaluation=evaluations[-1] if evaluations else None,
                         empty_evaluation_records=empty_evaluations,
                         full_search_end=full_search_end,
                         checkpoint_mib=(out/'checkpoint.pt').stat().st_size/1048576 if (out/'checkpoint.pt').is_file() else None,
                         engineering_only=launch.get('engineering_only',False),
                         truncated=launch.get('deadline_truncated',False)))
    summary=[]
    for seq_id in protocol['configurations']:
        for arm in ('cp_sat','rl','rl_cp_sat'):
            subset=[r for r in rows if r['seq_id']==seq_id and r['arm']==arm and not r['engineering_only']]
            at_budget=[]
            for r in subset:
                t=m['run_seconds']
                # Result means require the whole requested budget or an exact cube proof.
                if r['elapsed_s']>=t-1 or r['full_cube_optimal']:
                    # Use the last valid witness at/before the budget, not post-cutoff saving time.
                    out=directory/'tasks'/f'{r["task"]:03d}_{arm}_{seq_id}_seed{r["seed"]}'
                    vals=[e['contacts'] for e in records(out/'events.jsonl') if e['type']=='witness' and e['elapsed_s']<=t]
                    if vals:at_budget.append(max(vals))
            summary.append(dict(seq_id=seq_id,arm=arm,requested_seeds=len(m['tasks']) and len(protocol['seeds']),
                                seeds_with_valid_budget_result=len(at_budget),
                                mean_contacts=statistics.mean(at_budget) if at_budget else None,
                                sd_contacts=statistics.stdev(at_budget) if len(at_budget)>1 else None))
    report=dict(schema_version=2,analyser_sha256=file_hash(Path(__file__)),deadline_utc=m['deadline_utc'],run_budget_s=m['run_seconds'],
                independent_witnesses_checked=witnesses,verification_errors=errors,
                primary=summary,tasks=rows,decisionboost_cases=secondary,decisionboost_offline_costs=offline,
                interpretation='Missing outcomes remain null. Historical MPS/published results are excluded. DecisionBoost is a separate call-budget study.')
    save(directory/'analysis.json',report)
    lines=['# CPU cluster campaign', '', f"Deadline: {m['deadline_utc']}; per-run compute cutoff: {m['run_seconds']/3600:g} hours.",
           '', f'Independent witnesses checked: {witnesses}; verification errors: {len(errors)}.',
           '', 'Scores are integer HH contacts; energy is their negative. Decimal values below are means across seeds.',
           'Only outcomes with full requested elapsed-budget coverage or a full declared-cube optimality proof enter the final means.',
           'Missing witnesses and interrupted runs are not assigned zero. Read analysis.json for every run and timepoint.', '',
           '| Sequence | Method | Seeds with budget result | Mean contacts | SD |','|---|---|---:|---:|---:|']
    for r in summary:
        val='—' if r['mean_contacts'] is None else f"{r['mean_contacts']:.2f}"
        sd='—' if r['sd_contacts'] is None else f"{r['sd_contacts']:.2f}"
        lines.append(f"| {r['seq_id']} | {r['arm']} | {r['seeds_with_valid_budget_result']}/{r['requested_seeds']} | {val} | {sd} |")
    if m.get('stage') == 'seed0-feasibility':
        lines.extend(['', 'Seed-0 feasibility only: one seed per method/sequence. These outcomes assess learning and sustained resource use; they do not establish a multi-seed method ranking.',
                      'The rl_cp_sat arm uses fixed repairs, not the learned DecisionBoost controller. No full campaign follows automatically.'])
    lines.extend(['', '## Best witnessed contacts over time', '',
                  '| Sequence | Seed | Method | 1h | 2h | 12h | 24h |', '|---|---:|---|---:|---:|---:|---:|'])
    for row in rows:
        if row['arm'] not in ('cp_sat','rl','rl_cp_sat') or row['engineering_only']: continue
        trajectory={t['seconds']:t['contacts'] for t in row['trajectory']}
        values=['—' if trajectory.get(t) is None else str(trajectory[t]) for t in (3600,7200,43200,86400)]
        lines.append(f"| {row['seq_id']} | {row['seed']} | {row['arm']} | {' | '.join(values)} |")
    lines.extend(['', '## RL training and greedy policy diagnostics', '',
                  'Latest greedy evaluations use the last nonempty evaluation completed within the compute budget. Empty deadline evaluations are excluded; an evaluation with attempted but incomplete folds remains visible. A zero contact score on completed folds remains zero.',
                  'Greedy rollouts of one policy are not independent training seeds. Episode/update totals are final worker counters; epsilon uses the latest episode event within the budget.', '',
                  '| Sequence | Seed | Method | Episodes | Updates | Epsilon | Repairs | First greedy mean | Latest greedy mean | Complete / attempted | Latest eval hour | Empty eval records |',
                  '|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|'])
    def value(v): return '—' if v is None else str(v)
    def number(v,places=3): return '—' if v is None else f'{v:.{places}f}'
    for row in rows:
        if row['arm'] not in ('rl','rl_cp_sat') or row['engineering_only']: continue
        first=row['first_greedy_evaluation'] or {};last=row['latest_greedy_evaluation'] or {}
        epsilon=(row['last_episode_within_budget'] or {}).get('epsilon')
        completion='—' if not last else f"{last['completed_folds']}/{last['episodes']}"
        hour=None if not last else last['elapsed_s']/3600
        cells=[row['seq_id'],str(row['seed']),row['arm'],value(row['episode']),value(row['updates']),number(epsilon),value(row['repairs']),number(first.get('mean_complete_contacts')),number(last.get('mean_complete_contacts')),completion,number(hour),str(row['empty_evaluation_records'])]
        lines.append('| '+' | '.join(cells)+' |')
    lines.extend(['', '## Full-search solver status', '',
                  'Bounds below are for the declared full cube. Restricted repair bounds are excluded; FEASIBLE is not an optimality proof.', '',
                  '| Sequence | Seed | Status | Final contacts | Upper bound |', '|---|---:|---|---:|---:|'])
    for row in rows:
        if row['arm']!='cp_sat' or row['engineering_only']: continue
        final=row['full_search_end'] or {}
        lines.append(f"| {row['seq_id']} | {row['seed']} | {value(final.get('status'))} | {value(final.get('contacts'))} | {value(final.get('bound'))} |")
    scientific_secondary=[r for r in secondary if not r['engineering_only']]
    if scientific_secondary:
        lines.extend(['','## Separate DecisionBoost replication','',f'{len(scientific_secondary)} completed fresh sequence/seed cases.',
                      'Single-worker CP call caps match across repair policies. Online costs exclude the separately logged encoder/training/teaching costs.',
                      'These cases use complete reachable coordinate domains and cannot be merged with the primary restricted-cube time comparison.'])
    (directory/'report.md').write_text('\n'.join(lines)+'\n')
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory',type=Path);args=p.parse_args()
    result=analyse(args.directory.resolve())
    print(json.dumps(dict(tasks=len(result['tasks']),witnesses_checked=result['independent_witnesses_checked'],errors=result['verification_errors'])))
    if result['verification_errors']: raise SystemExit(1)
