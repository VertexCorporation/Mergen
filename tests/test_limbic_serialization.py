import unittest
import torch
import json
import os
from pathlib import Path

# Mock dependencies to instantiate LimbicExecutiveLayer
class MockEngine:
    def __init__(self):
        self.device = torch.device('cpu')
        self.weights = torch.zeros(1)
        self.eligibility = torch.zeros(1)
        self.trace_pre = torch.zeros(1)
        self.trace_post = torch.zeros(1)
        self.firing_rate_ema = torch.zeros(1)
        self._step_count = 0
        self._da_event_count = 0

class MockBroca:
    def __init__(self):
        self.concept_vocabulary = ["test"]
        self._total_expressions = 0

from cognitive.limbic_executive_layer import LimbicExecutiveLayer

class TestLimbicSerialization(unittest.TestCase):
    def setUp(self):
        self.mx_path = Path("test_old_brain.mx")
        self.limbic = LimbicExecutiveLayer(
            mergen_engine=MockEngine(),
            broca=MockBroca(),
            mx_path=str(self.mx_path),
            user_id="test_user"
        )

    def tearDown(self):
        if self.mx_path.exists():
            os.remove(self.mx_path)

    def test_legacy_load_state(self):
        """Simulate loading an old .mx file that has no neuromodulation data."""
        # Create an old state dictionary
        legacy_state = {
            'version': '1.0',
            'timestamp': '2025-01-01',
            'user_id': 'test_user',
            'engine': {},
            'broca': {},
            'limbic': {
                'episodic_memory': [],
                'internal_thoughts': [],
                'reward_history': [],
                'user_corrections': 0,
                'total_responses': 0,
                'passive_rejections': 0,
                'self_corrections': 0,
                'cumulative_reward': 0.0,
                'efficiency_score': 0.0,
                'dmn_cycles': 0
            }
            # Note: 'neuromodulation' key is missing!
        }
        
        # Manually encrypt it the same way LimbicExecutiveLayer does
        json_bytes = json.dumps(legacy_state).encode('utf-8')
        encrypted = self.limbic._xor_encrypt(json_bytes)
        with open(self.mx_path, 'wb') as f:
            f.write(self.limbic.MX_MAGIC)
            f.write(encrypted)
            
        # Corrupt the current neuromodulation values to ensure they get reset to defaults
        self.limbic.neuro.da.fill_(0.9)
        self.limbic.neuro.serotonin.fill_(0.9)
        
        # Load the old file
        success = self.limbic.load_state()
        self.assertTrue(success)
        
        # Verify fallback values were applied (Base levels)
        levels = self.limbic.neuro.get_levels()
        self.assertAlmostEqual(levels['DA'], self.limbic.neuro.BASE_DA)
        self.assertAlmostEqual(levels['5-HT'], self.limbic.neuro.BASE_5HT)
        self.assertAlmostEqual(levels['NE'], self.limbic.neuro.BASE_NE)
        self.assertAlmostEqual(levels['ACh'], self.limbic.neuro.BASE_ACH)
        
        # Verify tensors are still requires_grad=False
        self.assertFalse(self.limbic.neuro.da.requires_grad)

if __name__ == '__main__':
    unittest.main()
