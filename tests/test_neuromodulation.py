import unittest
import torch
from learning.neuromodulation import NeuromodulationSystem

class TestNeuromodulation(unittest.TestCase):
    def setUp(self):
        self.neuro = NeuromodulationSystem()

    def test_tensor_properties(self):
        # 1. Check if tensors are correctly initialized
        self.assertIsInstance(self.neuro.da, torch.Tensor)
        self.assertIsInstance(self.neuro.serotonin, torch.Tensor)
        self.assertIsInstance(self.neuro.noradrenaline, torch.Tensor)
        self.assertIsInstance(self.neuro.acetylcholine, torch.Tensor)
        
        # 2. Check requires_grad is False
        self.assertFalse(self.neuro.da.requires_grad)
        self.assertFalse(self.neuro.serotonin.requires_grad)
        self.assertFalse(self.neuro.noradrenaline.requires_grad)
        self.assertFalse(self.neuro.acetylcholine.requires_grad)
        
        # 3. Check device (cuda if available, else cpu)
        expected_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.assertEqual(self.neuro.device.type, expected_device.type)
        self.assertEqual(self.neuro.da.device.type, expected_device.type)

    def test_apply_signal_in_place(self):
        # Check that apply_signal works in place and doesn't create new tensor objects
        original_id = id(self.neuro.da)
        original_val = self.neuro.da.item()
        
        # Apply a signal
        self.neuro.apply_signal(da_delta=0.5)
        
        # Verify value changed and clamped at 1.0
        self.assertAlmostEqual(self.neuro.da.item(), min(1.0, original_val + 0.5))
        
        # Verify it's the exact same tensor in memory
        self.assertEqual(id(self.neuro.da), original_id)

    def test_homeostasis_decay(self):
        # Apply a big signal to push it away from baseline
        self.neuro.apply_signal(da_delta=0.8, ht_delta=0.4, ne_delta=0.6, ach_delta=0.5)
        
        da_after_signal = self.neuro.da.item()
        
        # Run tick_homeostasis
        original_id = id(self.neuro.da)
        self.neuro.tick_homeostasis()
        
        # Verify value moved towards baseline (decayed)
        da_after_tick = self.neuro.da.item()
        self.assertLess(da_after_tick, da_after_signal)
        self.assertGreater(da_after_tick, self.neuro.BASE_DA)
        
        # Verify in-place operation
        self.assertEqual(id(self.neuro.da), original_id)

    def test_set_and_get_levels(self):
        new_levels = {'DA': 0.9, '5-HT': 0.1, 'NE': 0.8, 'ACh': 0.2}
        self.neuro.set_levels(new_levels)
        
        levels = self.neuro.get_levels()
        for k, v in new_levels.items():
            self.assertAlmostEqual(levels[k], v, places=4)

if __name__ == '__main__':
    unittest.main()
