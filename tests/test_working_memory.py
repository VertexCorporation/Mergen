import unittest
import torch
from learning.neuromodulation import NeuromodulationSystem
from memory.working_memory import WorkingMemory

class TestWorkingMemory(unittest.TestCase):
    def setUp(self):
        self.neuro = NeuromodulationSystem(device='cpu')
        # 3 slotluk kapasite ve 4 boyutlu vektörler
        self.wm = WorkingMemory(capacity=3, vector_dim=4, neuro_system=self.neuro)

    def test_add_concept_ach_encoding(self):
        """ACh seviyesinin kodlama gücünü (başlangıç aktivasyonunu) belirlediğini test eder."""
        self.neuro.set_levels({'ACh': 0.8})
        concept = torch.tensor([1.0, 0.0, 1.0, 0.0])
        
        self.wm.add_concept(concept)
        
        # İlk slotun dolmasını ve aktivasyonunun ACh'ye eşit olmasını bekliyoruz.
        self.assertAlmostEqual(self.wm.activations[0].item(), 0.8, places=5)
        self.assertTrue(torch.allclose(self.wm.slots[0], concept))
        
        # Düşük ACh test edelim
        self.neuro.set_levels({'ACh': 0.0}) # 0.01'den küçükse yazmamalı
        concept2 = torch.tensor([0.0, 1.0, 0.0, 1.0])
        self.wm.add_concept(concept2)
        
        # 0. slotta eski concept durmalı, yeni concept hiçbir yere yazılmamalı
        self.assertTrue(torch.allclose(self.wm.slots[0], concept))
        self.assertEqual(self.wm.activations[1].item(), 0.0)

    def test_activation_based_eviction(self):
        """Kapasite dolduğunda en düşük aktivasyona sahip slotun üzerine yazıldığını test eder."""
        self.neuro.set_levels({'ACh': 1.0})
        
        # 3 slotu da doldur
        c1 = torch.tensor([1.0, 1.0, 1.0, 1.0])
        c2 = torch.tensor([2.0, 2.0, 2.0, 2.0])
        c3 = torch.tensor([3.0, 3.0, 3.0, 3.0])
        
        self.wm.add_concept(c1)
        self.wm.add_concept(c2)
        self.wm.add_concept(c3)
        
        # Şu an aktivasyonlar [1.0, 1.0, 1.0] olmalı.
        # Manuel olarak aktivasyonları değiştirelim ki eviction mantığı test edilebilsin.
        self.wm.activations[0] = 0.5
        self.wm.activations[1] = 0.9
        self.wm.activations[2] = 0.1 # En düşük!
        
        # Yeni bir concept ekleyelim
        c4 = torch.tensor([4.0, 4.0, 4.0, 4.0])
        self.wm.add_concept(c4)
        
        # c4, en düşük aktivasyona sahip olan 3. slotun (index 2) üzerine yazılmış olmalı.
        self.assertTrue(torch.allclose(self.wm.slots[2], c4))
        self.assertEqual(self.wm.activations[2].item(), 1.0) # Yeni ACh (1.0) ile kodlandı
        
        # Diğerleri bozulmamış olmalı
        self.assertTrue(torch.allclose(self.wm.slots[0], c1))
        self.assertTrue(torch.allclose(self.wm.slots[1], c2))

    def test_decay_ne_protection(self):
        """NE seviyesinin yüksekliğinin silinmeyi yavaşlattığını test eder."""
        self.neuro.set_levels({'ACh': 1.0})
        concept = torch.tensor([1.0, 0.0, 0.0, 1.0])
        self.wm.add_concept(concept)
        
        # Slot 0'ın aktivasyonu 1.0
        
        # NE yüksekse decay yavaşlar (base decay = 0.1)
        self.neuro.set_levels({'NE': 1.0})
        self.wm.tick()
        # decay_rate = 0.1 * (1.0 - 1.0) = 0.0 -> max(0.01) -> 0.01
        self.assertAlmostEqual(self.wm.activations[0].item(), 0.99, places=2)
        
        # NE düşükse decay hızlıdır
        self.neuro.set_levels({'NE': 0.0})
        self.wm.tick()
        # decay_rate = 0.1 * (1.0 - 0.0) = 0.1 -> 0.1
        self.assertAlmostEqual(self.wm.activations[0].item(), 0.89, places=2)
        
    def test_in_place_operations(self):
        """Tensör işlemlerinin in-place olduğunu (id korunduğunu) ve gradyan takip etmediğini test eder."""
        slots_id = id(self.wm.slots)
        activations_id = id(self.wm.activations)
        
        self.assertFalse(self.wm.slots.requires_grad)
        self.assertFalse(self.wm.activations.requires_grad)
        
        self.neuro.set_levels({'ACh': 1.0})
        self.wm.add_concept(torch.tensor([1.0, 1.0, 1.0, 1.0]))
        self.wm.tick()
        
        self.assertEqual(slots_id, id(self.wm.slots))
        self.assertEqual(activations_id, id(self.wm.activations))

if __name__ == '__main__':
    unittest.main()
