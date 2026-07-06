import unittest
import torch
from learning.neuromodulation import NeuromodulationSystem
from cognitive.thalamus import ThalamicFilter
from cognitive.limbic_executive_layer import LimbicExecutiveLayer

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
        self.n_post = 10
        self.lateral_k = 1

    def forward(self, pre):
        return torch.zeros(self.n_post)
        
    def update_traces(self, pre, post):
        pass

class MockBroca:
    def __init__(self):
        self.concept_vocabulary = ["test"]
        self._total_expressions = 0

class TestThalamicGating(unittest.TestCase):
    def setUp(self):
        self.neuro = NeuromodulationSystem(device='cpu')
        self.thalamus = ThalamicFilter(neuro_system=self.neuro, base_threshold=0.1)

    def test_thalamic_masking(self):
        """Test that spike trains below threshold are zeroed out (masked)."""
        # Base threshold is 0.1
        # Set spike train with values below and above threshold
        spike_train = torch.tensor([
            [0.05, 0.15, 0.01, 0.5]
        ])
        
        filtered = self.thalamus.apply_gating(spike_train)
        
        # 0.05 < 0.1 -> 0
        # 0.15 >= 0.1 -> 0.15
        # 0.01 < 0.1 -> 0
        # 0.5 >= 0.1 -> 0.5
        expected = torch.tensor([
            [0.0, 0.15, 0.0, 0.5]
        ])
        
        self.assertTrue(torch.allclose(filtered, expected))

    def test_dynamic_threshold_ne_da(self):
        """Test that NE increases threshold and DA decreases threshold."""
        # High NE (Focus) -> high threshold
        self.neuro.set_levels({'NE': 1.0, 'DA': 0.0})
        # Threshold: 0.1 + (1.0 * 0.2) - (0.0 * 0.1) = 0.3
        
        spike_train = torch.tensor([[0.25, 0.35]])
        filtered = self.thalamus.apply_gating(spike_train)
        
        # 0.25 is now below threshold (0.3), should be masked
        expected = torch.tensor([[0.0, 0.35]])
        self.assertTrue(torch.allclose(filtered, expected))
        
        # High DA (Curiosity) -> low threshold
        self.neuro.set_levels({'NE': 0.0, 'DA': 1.0})
        # Threshold: 0.1 + 0 - 0.1 = 0.0 -> clamped to 0.02
        spike_train2 = torch.tensor([[0.01, 0.05]])
        filtered2 = self.thalamus.apply_gating(spike_train2)
        
        # 0.01 < 0.02 -> masked, 0.05 > 0.02 -> passes
        expected2 = torch.tensor([[0.0, 0.05]])
        self.assertTrue(torch.allclose(filtered2, expected2))

    def test_global_serotonin_inhibition(self):
        """Test that high 5-HT blocks hallucination via LimbicExecutiveLayer."""
        limbic = LimbicExecutiveLayer(
            mergen_engine=MockEngine(),
            broca=MockBroca(),
            mx_path="temp_test.mx"
        )
        
        # Set high Serotonin (5-HT) for high inhibition
        limbic.neuro.set_levels({'5-HT': 1.0})
        # inhibition_threshold = 1.0 * 1.5 = 1.5
        
        # We simulate a low confidence response by bypassing wernicke and triggering the random fallback
        # which generates neural_intent with values maxed at 0.5.
        # Since 0.5 < 1.5, it should trigger hallucination suppression.
        
        limbic.wernicke = None # Force random intent generation (max 0.5)
        response = limbic.respond("test input")
        
        self.assertEqual(response, "Bu konuda net bir fikrim yok, uydurmak istemiyorum.")

if __name__ == '__main__':
    unittest.main()
