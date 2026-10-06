"""Behavioural engineering checks. Never start a scientific campaign."""
import importlib.util
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE));sys.path.insert(0,str(HERE/'vendor'))
from campaign import create, verify, save
from geometry import contacts
from launch import supervise

TORCH=importlib.util.find_spec('torch') is not None
ORTOOLS=importlib.util.find_spec('ortools') is not None
FOLD=[[0,0,0],[1,0,0],[1,1,0],[0,1,0],[0,2,0],[1,2,0],[1,3,0],[0,3,0]]


class ProtocolTests(unittest.TestCase):
    def test_unique_five_seed_matrix_and_deadline(self):
        with tempfile.TemporaryDirectory() as d:
            m=create(Path(d)/'run',mode='campaign')
            self.assertEqual(len(m['tasks']),120)
            self.assertEqual(len({(r['arm'],r['seed'],r['seq_id']) for r in m['tasks']}),120)
            self.assertEqual(m['deadline_epoch']-m['created_epoch'],864000)
            self.assertEqual(m['run_seconds'],86400)
            self.assertEqual(verify(Path(d)/'run'),m)

    def test_bundle_and_deadline_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)/'run';create(out,mode='campaign')
            p=out/'bundle/protocol.json';p.write_text(p.read_text()+' ')
            with self.assertRaises(RuntimeError):verify(out)
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)/'run';m=create(out,mode='campaign');m['deadline_epoch']+=864000
            save(out/'campaign.json',m)
            with self.assertRaises(RuntimeError):verify(out)

    def test_deadline_skips_without_importing_worker(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)/'run'
            with patch('campaign.time.time',return_value=time.time()-865000):create(out,mode='campaign')
            result=subprocess.run([sys.executable,str(out/'bundle/launch.py'),'--campaign',str(out),'--task','1'],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            launch=json.loads(next((out/'tasks').glob('*/launch.json')).read_text())
            self.assertEqual(launch['status'],'skipped_deadline')
            self.assertFalse(list((out/'tasks').glob('*/worker.log')))

    def test_unknown_results_are_missing(self):
        from analyse import analyse
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)/'run';create(out,mode='campaign')
            r=analyse(out)
            self.assertTrue(all(row['mean_contacts'] is None for row in r['primary']))
            self.assertTrue(all(row['seeds_with_valid_budget_result']==0 for row in r['primary']))

    def test_fresh_corpus_excludes_all_prior_reversals(self):
        m=json.loads((HERE/'decisionboost_corpus.json').read_text())
        keys=[min(s,s[::-1]) for rows in m['splits'].values() for s in rows]
        self.assertEqual(len(keys),232)
        self.assertEqual(len(keys),len(set(keys)))
        self.assertFalse(set(keys)&set(m['excluded_reversal_keys']))

    def test_fake_slurm_submission_and_paths_with_spaces(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);venv=root/'venv';(venv/'bin').mkdir(parents=True)
            (venv/'bin/python').symlink_to(sys.executable)
            fake=root/'commands';fake.mkdir();capture=root/'arguments.txt'
            sbatch=fake/'sbatch';sbatch.write_text('#!/bin/bash\nprintf "%s\\n" "$@" > "$HP_CAPTURE_FILE"\nprintf "12345\\n"\n');sbatch.chmod(0o755)
            env=dict(os.environ,HP_VENV_PATH=str(venv),HP_NODE_CONSTRAINT='verified_test_architecture',
                     HP_CAPTURE_FILE=str(capture),PATH=str(fake)+os.pathsep+os.environ['PATH'])
            out=root/'campaign with spaces'
            r=subprocess.run(['bash',str(HERE/'submit.sh'),'campaign',str(out),'--run-hours','12','--concurrency','3'],env=env,capture_output=True,text=True)
            self.assertEqual(r.returncode,0,r.stderr)
            args=capture.read_text().splitlines()
            self.assertIn('--array=0-119%3',args)
            self.assertIn('--constraint=verified_test_architecture',args)
            self.assertEqual(Path(args[-1]).resolve(),out.resolve())
            self.assertEqual((out/'job_id.txt').read_text().strip(),'12345')


class GeometryTests(unittest.TestCase):
    def test_contacts_counted_once(self):
        self.assertEqual(contacts('HHHHHHHH',FOLD,4),3)

    def test_bad_geometry_and_domain_rejected(self):
        for bad in (FOLD[:-1],FOLD[:3]+[FOLD[0]]+FOLD[4:],[[float(v) for v in p] for p in FOLD]):
            with self.assertRaises(ValueError):contacts('HHHHHHHH',bad,4)
        with self.assertRaises(ValueError):contacts('HHHHHHHH',FOLD,2)


class GuardianTests(unittest.TestCase):
    def test_native_style_child_is_forcibly_stopped(self):
        with tempfile.TemporaryFile() as log:
            start=time.time()
            r=supervise([sys.executable,'-c','import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(60)'],soft_end=start+.3,hard_end=start+.8,log=log)
            self.assertTrue(r['forced_kill'])
            self.assertEqual(r['stop_reason'],'application_deadline')
            self.assertLess(time.time()-start,3)


@unittest.skipUnless(ORTOOLS,'Install the CPU environment to exercise CP-SAT')
class SolverTests(unittest.TestCase):
    def test_full_search_and_restricted_repair(self):
        from hp_solver import solve
        events=[]
        fold,status=solve('HHHHHHHH',seconds=2,threads=1,seed=3,emit=events.append)
        self.assertIsNotNone(fold,status)
        self.assertGreaterEqual(contacts('HHHHHHHH',fold,4),3)
        repaired,status=solve('HHHHHHHH',seconds=1,threads=1,seed=3,emit=events.append,incumbent=FOLD,movable={5,6,7})
        self.assertIsNotNone(repaired,status)
        self.assertEqual(repaired[:5],FOLD[:5])
        self.assertGreaterEqual(contacts('HHHHHHHH',repaired,4),3)
        self.assertEqual(events[-1]['scope'],'restricted_repair')
        for e in events:
            if e['type']=='witness':self.assertEqual(contacts(e['seq'],e['positions'],4),e['contacts'])

    def test_no_time_does_not_fabricate_a_witness(self):
        from hp_solver import solve
        events=[]
        fold,status=solve('HHHHHHHH',seconds=0,threads=1,seed=0,emit=events.append)
        self.assertIsNone(fold);self.assertEqual(status,'NOT_STARTED')
        self.assertFalse(any(e['type']=='witness' for e in events))


@unittest.skipUnless(TORCH,'Install the CPU environment to exercise RL')
class RLTests(unittest.TestCase):
    def setUp(self):
        import torch
        torch.set_num_threads(1)
        from worker import make_agent,seed_all
        seed_all(32)
        self.config=dict(sequence='HHHHHHHH',length=8,batch_size=8,memory_size=64,target_update_freq=2,
                         d_model=16,nhead=2,num_layers=1,dim_feedforward=32,gamma=.98,lr=.0005)
        self.agent=make_agent(self.config)

    def populate(self):
        from worker import np_valid
        state=self.agent.env.reset()
        for i in range(24):
            valid=np_valid(self.agent.env)
            if not valid:state=self.agent.env.reset();valid=np_valid(self.agent.env)
            action=valid[0];new,reward,done,_=self.agent.env.step(action)
            self.agent.store_transition(state,action,reward,new,done);state=new
            if done:state=self.agent.env.reset()

    def test_evaluation_restores_mode_on_success_and_failure(self):
        from worker import evaluate,Recorder
        with tempfile.TemporaryDirectory() as d:
            r=Recorder(Path(d),self.config['sequence'])
            self.agent.policy_net.train();evaluate(self.agent,r,episodes=2)
            self.assertTrue(self.agent.policy_net.training)
            with patch.object(self.agent,'select_action',side_effect=RuntimeError('injected')):
                with self.assertRaises(RuntimeError):evaluate(self.agent,r,episodes=1)
            self.assertTrue(self.agent.policy_net.training)
            self.agent.policy_net.eval();evaluate(self.agent,r,episodes=1)
            self.assertFalse(self.agent.policy_net.training);r.handle.close()

    def test_checkpoint_restores_next_update_exactly(self):
        import numpy as np
        import random
        import torch
        from worker import checkpoint,Recorder,make_agent
        self.populate();self.agent.update()
        with tempfile.TemporaryDirectory() as d:
            r=Recorder(Path(d),self.config['sequence']);checkpoint(self.agent,1,r,3600,0)
            saved=torch.load(Path(d)/'checkpoint.pt',weights_only=False,map_location='cpu')
            expected_loss=self.agent.update()
            expected={k:v.clone() for k,v in self.agent.policy_net.state_dict().items()}
            other=make_agent(self.config)
            other.policy_net.load_state_dict(saved['policy']);other.target_net.load_state_dict(saved['target'])
            other.optimizer.load_state_dict(saved['optimizer']);other.memory.__dict__.update(saved['replay']);other.steps_done=saved['steps_done']
            random.setstate(saved['python_rng']);np.random.set_state(saved['numpy_rng']);torch.set_rng_state(saved['torch_rng'])
            self.assertEqual(other.update(),expected_loss)
            self.assertEqual(other.steps_done,self.agent.steps_done)
            for k,v in other.policy_net.state_dict().items():self.assertTrue(torch.equal(v,expected[k]),k)
            np.testing.assert_array_equal(other.memory.priorities,self.agent.memory.priorities)
            r.handle.close()

    def test_partial_tail_recovery_preserves_witness_and_rejects_midfile_corruption(self):
        from worker import Recorder
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)
            event=dict(type='witness',seq='HHHHHHHH',positions=FOLD,contacts=3,elapsed_s=1)
            (out/'events.jsonl').write_text(json.dumps(event)+'\n'+ '{"type":')
            r=Recorder(out,'HHHHHHHH',2)
            self.assertEqual(r.best_contacts,3)
            self.assertTrue((out/'interrupted_tail.bin').exists())
            r.emit(dict(type='recovered'));r.handle.close()
            self.assertEqual(len((out/'events.jsonl').read_text().splitlines()),2)
            (out/'events.jsonl').write_text('corrupt\n'+json.dumps(event)+'\n')
            with self.assertRaises(ValueError):Recorder(out,'HHHHHHHH',2)


if __name__=='__main__':unittest.main()
