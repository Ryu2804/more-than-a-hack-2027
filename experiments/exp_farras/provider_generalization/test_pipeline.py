import subprocess
import sys
import textwrap
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent


class PipelineIsolationTests(unittest.TestCase):
    def child(self, code):
        result = subprocess.run([sys.executable, '-B', '-c', textwrap.dedent(code)],
                                cwd=HERE, capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_scorer_rejects_missing_complete_seal_before_any_vault_read(self):
        self.child('''
            import tempfile
            from pathlib import Path
            import run
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                run.b.OUT = root / 'out'
                run.b.CACHE = root / 'cache'
                run.access.initialize().add(roots=[root / 'cache/labels'])
                try:
                    run.compute_scores('a' * 64)
                except FileNotFoundError as error:
                    assert 'predictions_seal.json' in str(error), str(error)
                else:
                    raise AssertionError('Scoring proceeded without complete seal')
                assert run.access.report()['nonprobe_blocked_attempts'] == 0
        ''')

    def test_observation_carrier_with_label_members_is_rejected(self):
        self.child('''
            import tempfile
            from pathlib import Path
            import numpy as np
            import bridge as b
            with tempfile.TemporaryDirectory() as directory:
                b.CACHE = Path(directory)
                observation, features = b.case_paths(b.SC[0], 'clean')
                observation.parent.mkdir(parents=True)
                np.savez_compressed(observation, X=np.zeros((3, 16)), y=np.zeros(3))
                try:
                    b.load_observations(b.SC[0], 'clean')
                except ValueError as error:
                    assert 'Label leakage' in str(error)
                else:
                    raise AssertionError('Label-bearing target carrier was accepted')
        ''')

    def test_bootstrap_weights_providers_equally_not_match_counts(self):
        self.child('''
            import run
            left, right = [], []
            for provider, names in run.b.PROVIDERS.items():
                change = .2 if provider == 'idsse' else -.1
                for name in names:
                    common = dict(match=name, target_provider=provider, condition='clean')
                    left.append(dict(**common, spell_f1_mean=.5 + change))
                    right.append(dict(**common, spell_f1_mean=.5))
            result = run.paired(dict(rows=left), dict(rows=right), 'clean')
            assert abs(result['mean_delta'] - .05) < 1e-12, result
            assert abs(result['lower'] - .05) < 1e-12, result
            assert abs(result['upper'] - .05) < 1e-12, result
            assert result['wins'] == 7, result
        ''')

    def test_model_rejects_provider_overlap_and_target_labels_before_inference(self):
        self.child('''
            import models
            import bridge as b
            fitted = dict(source='idsse', training_names=b.train_names('idsse'))
            for observed in [dict(_name=b.IDSSE[0]), dict(_name=b.SC[0], y=object()), dict(_name=b.SC[0], y25=object())]:
                try:
                    models.predict(fitted, observed)
                except ValueError:
                    pass
                else:
                    raise AssertionError('Provider or label leakage reached model inference')
        ''')


if __name__ == '__main__':
    unittest.main()
