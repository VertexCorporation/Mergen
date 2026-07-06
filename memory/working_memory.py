import torch
from learning.neuromodulation import NeuromodulationSystem

class WorkingMemory:
    """
    Prefrontal Cortex (Çalışma Belleği / Working Memory):
    Sınırlı kapasiteli (ör. 5 slot), anlık kavram ve konseptleri tutan
    kısa süreli bellek alanı.
    
    ACh (Asetilkolin): Kodlama gücünü (yazılma kuvvetini) belirler.
    NE (Noradrenalin): Dikkat ve tutunma süresini (silinme hızını yavaşlatma) belirler.
    """
    def __init__(self, capacity: int, vector_dim: int, neuro_system: NeuromodulationSystem):
        self.capacity = capacity
        self.vector_dim = vector_dim
        self.neuro = neuro_system
        self.device = self.neuro.device
        
        # memory leak'i önlemek için requires_grad=False olarak başlatılır
        self.slots = torch.zeros((self.capacity, self.vector_dim), device=self.device, dtype=torch.float32, requires_grad=False)
        self.activations = torch.zeros(self.capacity, device=self.device, dtype=torch.float32, requires_grad=False)
        
        # Predictive Processing (Beklenti Üretimi)
        self.predicted_next_vector = torch.zeros(self.vector_dim, device=self.device, dtype=torch.float32, requires_grad=False)
        
        self.base_decay = 0.1

    def add_concept(self, concept_vector: torch.Tensor):
        """
        Yeni gelen bir bilgiyi çalışma belleğine yazar.
        ACh yüksekse güçlü yazılır (başlangıç aktivasyonu yüksektir).
        Kapasite doluysa, aktivasyonu en düşük olan (en az dikkat edilen) slotun üstine yazar.
        """
        if concept_vector is None or torch.sum(concept_vector) == 0:
            return

        # BUG-03 FIX: NaN/Inf içeren vektörler predicted_next_vector'ü
        # zehirler ve tüm sonraki surprise hesaplamalarını bozar.
        # Güvenli kontrol: bırakılmadan önce boyut doğrula.
        if not torch.isfinite(concept_vector).all():
            return

        # --- PREDICTIVE PROCESSING: Sürpriz ve Dopamin ---
        # Eğer geçmişten gelen bir beklenti varsa ve yeni konsept boş değilse:
        concept_norm = torch.norm(concept_vector)
        pred_norm = torch.norm(self.predicted_next_vector)
        
        if concept_norm > 1e-8 and pred_norm > 1e-8:
            import torch.nn.functional as F
            # Cosine similarity hesapla (1.0 = tam aynı, -1.0 = zıt)
            cos_sim = F.cosine_similarity(concept_vector.unsqueeze(0), self.predicted_next_vector.unsqueeze(0), eps=1e-8).item()
            # BUG-03 FIX: cos_sim nan içerebilir eğer vektörler sınırda ise.
            if not (cos_sim == cos_sim):  # NaN check (NaN != NaN)
                surprise = 0.0
            else:
                # Sürpriz: 1.0 - cos_sim (Eğer aynıysa 0, eğer dik ise 1, zıt ise 2)
                surprise = max(0.0, 1.0 - cos_sim)
        else:
            surprise = 0.0
            
        # Sürpriz yüksekse Dopamin (DA) sıçraması tetikle
        if surprise > 0.1:
            da_delta = surprise * 0.5
            print(f"\n[WorkingMemory] ⚡ Sürpriz tespit edildi! (Sürpriz: {surprise:.3f}). DA seviyesi artırılıyor (+{da_delta:.3f})")
            self.neuro.apply_signal(da_delta=da_delta)

        # --- MEMORY ENCODING ---
        levels = self.neuro.get_levels()
        ach_level = levels.get('ACh', self.neuro.BASE_ACH)
        
        # Kodlama gücü ACh ile orantılıdır
        encoding_strength = ach_level
        
        if encoding_strength <= 0.01:
            # Çok düşük asetilkolin varsa bilgi belleğe bile giremez
            return
            
        # En düşük aktivasyona sahip slotu bul
        min_idx = torch.argmin(self.activations)
        
        # Slotu in-place güncelle
        self.slots[min_idx].copy_(concept_vector)
        self.activations[min_idx] = encoding_strength
        
        # --- BEKLENTİ ÜRETİMİ (Next Prediction) ---
        # Aktif slotların ağırlıklı ortalaması ile bir sonraki adımı tahmin et
        sum_act = self.activations.sum()
        if sum_act > 1e-8:
            # (capacity,) x (capacity, vector_dim) -> (vector_dim,)
            weighted_sum = torch.matmul(self.activations, self.slots)
            # In-place güncelleme (memory korumalı)
            self.predicted_next_vector.copy_(weighted_sum / sum_act)
        else:
            self.predicted_next_vector.zero_()

    def tick(self):
        """
        Zamanın akışıyla birlikte bellekteki bilgilerin silinmesi (decay).
        NE yüksekse silinme yavaşlar (bilgi daha uzun süre tutulur).
        """
        levels = self.neuro.get_levels()
        ne_level = levels.get('NE', self.neuro.BASE_NE)
        
        # NE decay oranını azaltır (NE=1 ise decay 0 olur)
        decay_rate = self.base_decay * (1.0 - ne_level)
        # Çok küçük decay'leri sıfırlama, minimum bir decay olsun
        decay_rate = max(0.01, decay_rate)
        
        # Aktivasyonları in-place düşür
        self.activations.sub_(decay_rate).clamp_(min=0.0)
        
        # Aktivasyonu sıfıra düşen slotların vektörlerini de sıfırla (temizle)
        # Bunu in-place yapmak için mask kullanıyoruz
        zero_mask = self.activations == 0.0
        # maskeli indexleri bulup 0.0 ile doldur
        if zero_mask.any():
            self.slots[zero_mask] = 0.0
