import unittest
import torch
from learning.neuromodulation import NeuromodulationSystem
from memory.working_memory import WorkingMemory

class TestPredictiveProcessing(unittest.TestCase):
    def setUp(self):
        self.neuro = NeuromodulationSystem(device='cpu')
        # Dopamini sıfırlayalım ki sıçramayı (spike) net görebilelim
        self.neuro.da.zero_()
        self.wm = WorkingMemory(capacity=3, vector_dim=4, neuro_system=self.neuro)
        # ACh (Kodlama) seviyesini maksimuma al
        self.neuro.set_levels({'ACh': 1.0})

    def test_low_surprise_constant_dopamine(self):
        """Birbirine çok benzer vektörler verildiğinde sürprizin ve dopaminin sabit kaldığını (veya sıfır olduğunu) test eder."""
        concept1 = torch.tensor([1.0, 0.5, 0.0, 0.0])
        self.wm.add_concept(concept1)
        
        # İlk eklendiğinde predicted vector henüz yoktu (sıfırdı), o yüzden sürpriz olmadı. DA artmamış olmalı.
        self.assertEqual(self.neuro.da.item(), 0.0)
        
        # Şimdi aynı konsepti tekrar ekleyelim
        self.wm.add_concept(concept1)
        
        # Cosine similarity 1.0 olacak, surprise 0.0 olacak. DA sıçramamalı.
        self.assertEqual(self.neuro.da.item(), 0.0)

    def test_high_surprise_dopamine_spike(self):
        """Aniden çok farklı (ortogonal) bir vektör geldiğinde sürpriz skorunun fırlayıp dopamin sıçrattığını test eder."""
        concept_fruit = torch.tensor([1.0, 1.0, 0.0, 0.0])
        self.wm.add_concept(concept_fruit)
        self.wm.add_concept(concept_fruit) # Beklenti tamamen [1,1,0,0] yönünde oturdu.
        
        # Şu ana kadar dopamin sıfır
        self.assertEqual(self.neuro.da.item(), 0.0)
        
        # Aniden bambaşka bir konsept (ortogonal) ekleyelim
        concept_quantum = torch.tensor([0.0, 0.0, 1.0, 1.0])
        self.wm.add_concept(concept_quantum)
        
        # Cosine similarity 0.0 olacak, surprise 1.0 olacak.
        # da_delta = surprise * 0.5 = 0.5
        self.assertAlmostEqual(self.neuro.da.item(), 0.5, places=5)
        
        # Tamamen zıt bir vektör eklersek (Negatif similarity)
        concept_opposite = torch.tensor([0.0, 0.0, -1.0, -1.0])
        self.wm.add_concept(concept_opposite)
        
        # Mevcut beklenti neydi? 
        # Slots: [Fruit, Fruit, Quantum]. Aktivationlar: [1,1,1].
        # Weighted avg: [2/3, 2/3, 1/3, 1/3]
        # concept_opposite ile cos_sim = -0.something
        # Surprise > 1.0 olacak
        # Yeni DA sıçraması olmalı, ve DA 1.0'a doğru çıkmalı
        self.assertTrue(self.neuro.da.item() > 0.5)
        
    def test_epsilon_zero_protection(self):
        """Sıfır vektör verildiğinde nan/inf oluşmadığını test eder."""
        zero_concept = torch.tensor([0.0, 0.0, 0.0, 0.0])
        self.wm.add_concept(zero_concept)
        
        # Sistemin çökmemesi ve sürpriz üretmemesi gerekir
        self.assertEqual(self.neuro.da.item(), 0.0)
        # Predicted vector hala sıfır kalmalı (çünkü girdi 0'dı ve kaydedilmedi)
        self.assertTrue(torch.allclose(self.wm.predicted_next_vector, zero_concept))

if __name__ == '__main__':
    unittest.main()
