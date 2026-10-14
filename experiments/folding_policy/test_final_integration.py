"""Disposable end-to-end wiring check, never an evaluation of final-study data.

All HP strings below are synthetic eight-residue fixtures, shorter than every
sequence in the actual study. Real models, updates, samplers, CP-SAT searches and
report verification run for fractions of a second. Only the frozen campaign
loader and the 7200/9000-second receipt boundary are replaced with a tiny fixture
contract. Child calls execute in-process so those fixture patches remain active;
subprocess isolation and Slurm submission have separate launcher/worker tests.
"""
import contextlib
from datetime import datetime, timedelta, timezone
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import torch

import final_launch as launch
import final_worker as worker
from final_report import summarize
from final_train import FinalParentTrainer, MODEL, validate_final_checkpoint
from evaluate import load_cp_solver, verified_score


class FinalIntegrationTests(unittest.TestCase):
    def test_real_training_searches_and_report_use_the_same_endpoint_contract(self):
        with tempfile.TemporaryDirectory(prefix='hp-final-wiring-only-') as directory:
            run = Path(directory).resolve()
            data = {
                'train': [{'id': 'fixture_training', 'sequence': 'HHHHHHHH', 'length': 8}],
                'validation': [{'id': 'fixture_unused', 'sequence': 'HHHPHHHH', 'length': 8}],
                'test': [{'id': 'fixture_test', 'sequence': 'HHPHPHHH', 'length': 8}],
                'benchmark': [{'id': 'fixture_benchmark', 'sequence': 'HPHHHPPH', 'length': 8}],
                'held_out_training_lengths': [],
            }
            launch.write_json(run / 'data.json', data)
            config = {'model': MODEL, 'allocation': {'initial_data_seconds': .05}}
            launch.write_json(run / 'training_config.json', config)
            # Warm imports/Adam construction, not training or evaluation, before
            # the very short fixture clocks begin.
            FinalParentTrainer(run / 'data.json', config, 0, 1)
            load_cp_solver()
            jobs = [{'task': 0, 'group': 'parent', 'arm': 'parent', 'seed': 0, 'index': 0,
                     'job_kind': 'parent', 'method': None, 'training_seconds': .75,
                     'parent_task': None, 'evaluation_tasks': []}]
            evaluations = []
            arms = ['grpo', 'scratch_grpo', 'cp_sat', 'greedy']
            for arm in arms:
                model = arm not in ('cp_sat', 'greedy')
                task = {'task': len(jobs), 'group': arm, 'arm': arm, 'seed': 0, 'index': 0,
                        'job_kind': 'endpoint', 'method': 'model' if model else arm,
                        'training_seconds': .25 if arm == 'grpo' else 1.0 if model else 0,
                        'parent_task': 0 if arm == 'grpo' else None, 'evaluation_tasks': []}
                for split in ('test', 'benchmark'):
                    task['evaluation_tasks'].append(len(evaluations))
                    evaluations.append({'task': len(evaluations), 'arm': arm,
                                        'method': task['method'], 'seed': 0, 'split': split,
                                        'sequence_ids': [data[split][0]['id']], 'seconds': .4,
                                        'training_task': task['task'] if model else None,
                                        'job_task': task['task'], 'training_seconds': 1.0 if model else 0})
                jobs.append(task)
            manifest = {
                'threads': 1, 'memory_gib': 16, 'model': MODEL, 'tasks': jobs,
                'evaluation_tasks': evaluations, 'arms': arms, 'seeds': [0],
                'hashes': {'data.json': launch.sha(run / 'data.json')},
                'software_at_freeze': worker.software(), 'python_executable': sys.executable,
                'deadline_utc': (datetime.now(timezone.utc) + timedelta(days=1)).isoformat(),
                'training_lineage_seconds': 1.0, 'inference_timepoints_s': [.1, .4],
                'attempt_checkpoints': [8], 'contrasts': [('grpo', 'scratch_grpo'), ('grpo', 'cp_sat')],
            }
            # The integration shim keeps actual checkpoint validation and all
            # metadata, but accepts the explicitly shortened fixture lineage.
            def verified_training(directory, task, _manifest=None):
                folder = launch.training_directory(directory, task)
                receipt = json.loads((folder / 'result.json').read_text())
                checkpoint = folder / 'final.pt'
                saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
                parent = (verified_training(directory, jobs[task['parent_task']])
                          if task['parent_task'] is not None else None)
                validate_final_checkpoint(saved, manifest['hashes']['data.json'], seed=task['seed'],
                                          arm=task['arm'], parent_sha=parent['file_sha256'] if parent else None)
                self.assertEqual(receipt['status'], 'complete')
                self.assertEqual(receipt['checkpoint_sha256'], launch.sha(checkpoint))
                self.assertGreaterEqual(receipt['compute_elapsed_s'], task['training_seconds'])
                self.assertEqual(saved['training_seconds'], .75 if task['job_kind'] == 'parent' else 1.0)
                self.assertEqual(saved['segment_training_seconds'], task['training_seconds'])
                self.assertEqual(saved['threads'], manifest['threads'])
                keys = ('schema', 'kind', 'role', 'arm', 'seed', 'threads', 'stage', 'condition',
                        'model_config', 'dataset_sha256', 'training_seconds', 'segment_training_seconds',
                        'parent_file_sha256', 'parent_state_sha256', 'initial_state_sha256', 'state_sha256',
                        'selection', 'training_data_split', 'final_test_enabled', 'test_evaluated',
                        'final_campaign', 'status')
                return {'checkpoint': str(checkpoint), 'file_sha256': launch.sha(checkpoint),
                        'state_sha256': saved['state_sha256'], 'training_seconds': saved['training_seconds'],
                        'metadata': {key: saved.get(key) for key in keys}, 'receipt': receipt}

            def verified_endpoint(directory, task):
                return verified_training(directory, jobs[task['training_task']])

            evaluated = []
            def child(command, **_kwargs):
                evaluation_id = int(command[command.index('--evaluation') + 1])
                sequence_id = command[command.index('--sequence') + 1]
                output = Path(command[command.index('--output') + 1])
                result = worker.evaluate_one(run, evaluation_id, sequence_id, output)
                evaluated.append((evaluations[evaluation_id]['method'], result))
                return types.SimpleNamespace(returncode=0)

            real_receipt_validator = launch.verified_training
            with (patch.object(worker, 'verify', return_value=manifest),
                  patch.object(launch, 'verify', return_value=manifest),
                  patch.object(worker, 'verified_training', side_effect=verified_training),
                  patch.object(launch, 'verified_training', side_effect=verified_training),
                  patch.object(launch, 'verified_endpoint', side_effect=verified_endpoint),
                  patch.object(worker.subprocess, 'run', side_effect=child),
                  patch.dict(os.environ, {'SLURM_CPUS_PER_TASK': '1'}),
                  contextlib.redirect_stdout(io.StringIO())):
                for task in jobs:
                    worker.run_job(run, task['group'], task['index'])
                try:
                    analysis = summarize(run)
                except SystemExit:
                    self.fail(json.loads((run / 'analysis.json').read_text())['errors'])

            self.assertEqual(analysis['errors'], [])
            self.assertEqual(analysis['complete_tasks'], 8)
            self.assertEqual(len(analysis['training']), 3)
            self.assertEqual({method for method, _ in evaluated}, {'model', 'cp_sat', 'greedy'})
            self.assertEqual(len(evaluated), 8)
            self.assertTrue((run / 'report.md').is_file())
            self.assertTrue((run / 'paired.csv').is_file())
            self.assertGreater(analysis['verified_witness_records'], 0)
            for method, row in evaluated:
                self.assertIsNotNone(row['best_contacts'], method)
                self.assertEqual(verified_score(row['sequence'], row['best_positions']), row['best_contacts'])
                self.assertIn(row['sequence'], {data['test'][0]['sequence'], data['benchmark'][0]['sequence']})
                if method == 'model':
                    self.assertTrue(row['attempt_prefixes'][0]['covered'])
            for task in jobs:
                record = json.loads((launch.job_directory(run, task) / 'launch.json').read_text())
                self.assertEqual(record['status'], 'complete')
            parent = verified_training(run, jobs[0])
            branch = verified_training(run, jobs[1])
            scratch = verified_training(run, jobs[2])
            self.assertEqual(branch['metadata']['initial_state_sha256'], parent['state_sha256'])
            self.assertIsNone(scratch['metadata']['parent_file_sha256'])
            self.assertEqual(scratch['metadata']['final_campaign']['reference_source'], 'frozen_random_initialization')
            # The production boundary must reject the short fixture rather than
            # let this engineering run masquerade as a 2h/2.5h campaign endpoint.
            with self.assertRaisesRegex(ValueError, 'receipt|lineage'):
                real_receipt_validator(run, jobs[0], manifest)


if __name__ == '__main__':
    unittest.main()
