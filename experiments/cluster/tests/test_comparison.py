"""Reuse/aggregation/submission checks with synthetic artifacts, no training."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))
from campaign import create,file_hash,hashes,save,verify
from geometry import contacts
from reuse import task_directory
from analyse import analyse


def complete(directory,index,task):
    protocol=json.loads((directory/'bundle/protocol.json').read_text())
    sequence=protocol['configurations'][task['seq_id']]['sequence']
    fold=[[i%2 if (i//2)%2==0 else 1-i%2,i//2,0] for i in range(len(sequence))]
    count=contacts(sequence,fold,len(sequence)//2)
    folder=task_directory(directory,index,task);folder.mkdir()
    save(folder/'launch.json',dict(task=task,status='finished',exit_code=0,total_elapsed_s=86406,
                                  device='cpu',threads=8,allocated_cpus='8',cpu_model='synthetic test CPU',engineering_only=False))
    save(folder/'result.json',dict(**task,status='stopped',full_cube_optimal=False,engineering_only=False,
                                  versions={'torch':'2.9.0+cpu','numpy':'2.3.4','ortools':'9.15.6755'}))
    (folder/'events.jsonl').write_text(json.dumps(dict(type='witness',seq=sequence,positions=fold,
                                                     contacts=count,elapsed_s=3600))+'\n')
    return count


def original(root):
    out=root/'completed seed0'
    with patch('campaign.time.time',return_value=time.time()-3*86400):
        manifest=create(out,mode='campaign',stage='seed0-feasibility')
    for i,task in enumerate(manifest['tasks']):
        if task['arm'] in ('cp_sat','rl'):complete(out,i,task)
    return out


def reseal(directory):
    m=json.loads((directory/'campaign.json').read_text())
    m['bundle_sha256']=hashes(directory/'bundle')
    save(directory/'campaign.json',m)
    (directory/'campaign.sha256').write_text(file_hash(directory/'campaign.json')+'\n')


class ComparisonTests(unittest.TestCase):
    def test_full_two_method_matrix_is_eighty(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)/'run';m=create(out,mode='campaign',stage='rl-vs-cp-sat')
            self.assertEqual(len(m['tasks']),80)
            self.assertEqual({t['arm'] for t in m['tasks']},{'cp_sat','rl'})
            self.assertEqual(m['expected_primary_tasks'],80)
            self.assertEqual(m['planned_new_cpu_hours'],15360)
            self.assertEqual(len(analyse(out)['primary']),16)

    def test_reuse_excludes_exactly_six_and_preserves_old_deadline_and_raw_files(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);old=original(root);before=hashes(old)
            m=create(root/'comparison',mode='campaign',stage='rl-vs-cp-sat',reuse_campaign=old)
            self.assertEqual(len(m['tasks']),74)
            self.assertEqual(len(m['reused_tasks']),6)
            self.assertEqual(m['expected_primary_tasks'],80)
            self.assertEqual(m['planned_new_cpu_hours'],14208)
            self.assertEqual(m['deadline_epoch'],verify(old)['deadline_epoch'])
            queued={(t['arm'],t['seed'],t['seq_id']) for t in m['tasks']}
            reused={(t['task']['arm'],t['task']['seed'],t['task']['seq_id']) for t in m['reused_tasks']}
            self.assertFalse(queued&reused)
            self.assertEqual(len(queued|reused),80)
            self.assertEqual(hashes(old),before)

    def test_combined_analysis_has_five_seed_denominator_and_no_double_counting(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);old=original(root);before=hashes(old);out=root/'comparison'
            m=create(out,mode='campaign',stage='rl-vs-cp-sat',reuse_campaign=old)
            index,task=next((i,t) for i,t in enumerate(m['tasks']) if t==dict(arm='rl',seed=1,seq_id='3d4'))
            value=complete(out,index,task)
            r=analyse(out)
            self.assertEqual(len(r['tasks']),80)
            self.assertEqual(r['reused_tasks'],6)
            self.assertEqual(r['independent_witnesses_checked'],7)
            self.assertEqual(r['verification_errors'],[])
            row=next(s for s in r['primary'] if s['seq_id']=='3d4' and s['arm']=='rl')
            self.assertEqual(row['requested_seeds'],5)
            self.assertEqual(row['seeds_with_valid_budget_result'],2)
            self.assertEqual(row['mean_contacts'],value)
            self.assertEqual(row['sd_contacts'],0)
            self.assertTrue(all(t['arm'] in ('cp_sat','rl') for t in r['tasks']))
            self.assertEqual(sum(t['reused'] for t in r['tasks']),6)
            self.assertEqual(hashes(old),before)
            text=(out/'report.md').read_text()
            self.assertIn('| 3d4 | rl | 2/5 |',text)
            self.assertIn('74 new, 6 reused',text)

    def test_changed_raw_imports_are_rejected_before_reanalysis(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);old=original(root);out=root/'comparison'
            create(out,mode='campaign',stage='rl-vs-cp-sat',reuse_campaign=old)
            p=old/'tasks/000_cp_sat_3d4_seed0/events.jsonl';p.write_text(p.read_text()+'\n')
            with self.assertRaisesRegex(RuntimeError,'raw results changed'):analyse(out)
            self.assertFalse((out/'analysis.json').exists())

    def test_reuse_rejects_budget_threads_configuration_and_scientific_code_changes(self):
        for change in ('budget','threads','config','code'):
            with self.subTest(change=change),tempfile.TemporaryDirectory() as d:
                root=Path(d);old=original(root);kwargs={}
                if change=='budget':kwargs['run_hours']=12
                if change=='threads':kwargs['threads']=4
                if change=='config':
                    p=old/'bundle/protocol.json';v=json.loads(p.read_text());v['configurations']['3d4']['batch_size']=1;save(p,v);reseal(old)
                if change=='code':
                    p=old/'bundle/worker.py';p.write_text(p.read_text()+'\n# changed scientific source\n');reseal(old)
                with self.assertRaises(ValueError):create(root/'invalid',mode='campaign',stage='rl-vs-cp-sat',reuse_campaign=old,**kwargs)
                self.assertFalse((root/'invalid').exists())

    def test_failed_short_engineering_or_wrong_identity_results_are_not_reused(self):
        for change in ('failure','short','engineering','identity','device','cores','model','versions'):
            with self.subTest(change=change),tempfile.TemporaryDirectory() as d:
                root=Path(d);old=original(root);p=old/'tasks/000_cp_sat_3d4_seed0/launch.json';v=json.loads(p.read_text())
                if change=='failure':v['exit_code']=-9
                if change=='short':v['total_elapsed_s']=3600
                if change=='engineering':v['engineering_only']=True
                if change=='identity':v['task']['seed']=99
                if change=='device':v['device']='mps'
                if change=='cores':v['allocated_cpus']='4'
                if change=='model':v['cpu_model']='different synthetic CPU'
                if change=='versions':
                    result_path=p.parent/'result.json';r=json.loads(result_path.read_text());r['versions']['torch']='2.11.0';save(result_path,r)
                save(p,v)
                with self.assertRaises(ValueError):create(root/'invalid',mode='campaign',stage='rl-vs-cp-sat',reuse_campaign=old)
                self.assertFalse((root/'invalid').exists())

    def test_wrong_sequence_bad_score_and_post_budget_only_witness_are_rejected(self):
        for change in ('sequence','contacts','time'):
            with self.subTest(change=change),tempfile.TemporaryDirectory() as d:
                root=Path(d);old=original(root);p=old/'tasks/000_cp_sat_3d4_seed0/events.jsonl';v=json.loads(p.read_text())
                if change=='sequence':v['seq']='H'*len(v['seq'])
                if change=='contacts':v['contacts']+=1
                if change=='time':v['elapsed_s']=86401
                p.write_text(json.dumps(v)+'\n')
                with self.assertRaises(ValueError):create(root/'invalid',mode='campaign',stage='rl-vs-cp-sat',reuse_campaign=old)
                self.assertFalse((root/'invalid').exists())

    def test_reuse_does_not_reset_a_nearly_expired_deadline(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);old=original(root);m=verify(old)
            with patch('campaign.time.time',return_value=m['deadline_epoch']-3600):
                with self.assertRaisesRegex(ValueError,'insufficient time'):create(root/'invalid',mode='campaign',stage='rl-vs-cp-sat',reuse_campaign=old)
            self.assertFalse((root/'invalid').exists())

    def test_new_tasks_reject_a_cpu_model_different_from_the_reused_runs(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);old=original(root);out=root/'comparison'
            create(out,mode='campaign',stage='rl-vs-cp-sat',reuse_campaign=old)
            r=subprocess.run([sys.executable,str(out/'bundle/launch.py'),'--campaign',str(out),'--task','0'],capture_output=True,text=True)
            self.assertNotEqual(r.returncode,0)
            self.assertIn('CPU model differs',r.stderr)
            self.assertFalse(list((out/'tasks').glob('*/worker.log')))

    def test_wrapper_submits_seventy_four_array_tasks_with_email(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);old=original(root);before=hashes(old)
            venv=root/'venv';(venv/'bin').mkdir(parents=True);(venv/'bin/python').symlink_to(sys.executable)
            fake=root/'commands';fake.mkdir();capture=root/'arguments.txt'
            sbatch=fake/'sbatch';sbatch.write_text('#!/bin/bash\nprintf "%s\\n" "$@" > "$HP_CAPTURE_FILE"\nprintf "12345\\n"\n');sbatch.chmod(0o755)
            env=dict(os.environ,HP_VENV_PATH=str(venv),HP_NODE_CONSTRAINT='ehc',HP_CAPTURE_FILE=str(capture),PATH=str(fake)+os.pathsep+os.environ['PATH'])
            out=root/'comparison with spaces'
            r=subprocess.run(['bash',str(HERE/'rl-cp-comparison.sh'),str(old),str(out)],env=env,capture_output=True,text=True)
            self.assertEqual(r.returncode,0,r.stderr)
            for arg in ('--array=0-73%24','--cpus-per-task=8','--constraint=ehc','--mail-type=END,FAIL'):
                self.assertIn(arg,capture.read_text().splitlines())
            self.assertEqual(verify(out)['deadline_epoch'],verify(old)['deadline_epoch'])
            self.assertEqual((out/'job_id.txt').read_text().strip(),'12345')
            self.assertEqual(hashes(old),before)
            capture.unlink()
            r=subprocess.run(['bash',str(HERE/'rl-cp-comparison.sh'),str(old),str(root/'dry'),'--dry-run'],env=env,capture_output=True,text=True)
            self.assertEqual(r.returncode,0,r.stderr)
            self.assertFalse(capture.exists())
            self.assertIn('--array=0-73%24',r.stdout)


if __name__=='__main__':unittest.main()
