import torch
from learning.neuromodulation import NeuromodulationSystem

class ThalamicFilter:
    """
    Talamus (Thalamic Gating):
    Wernicke veya diğer duyu alanlarından gelen spike'ları (sinyalleri) süzer.
    Dopamin (DA) ve Noradrenalin (NE) seviyelerine göre eşiği (threshold) dinamik
    olarak hesaplar ve eşiğin altında kalan gürültülü sinyalleri baskılar.
    """
    def __init__(self, neuro_system: NeuromodulationSystem, base_threshold: float = 0.1):
        self.neuro = neuro_system
        self.base_threshold = base_threshold

    def _calculate_dynamic_threshold(self) -> float:
        """
        NE (Odak daralması) -> Eşiği artırır.
        DA (Motivasyon/Öğrenme) -> Eşiği düşürür.
        """
        levels = self.neuro.get_levels()
        ne_level = levels.get('NE', self.neuro.BASE_NE)
        da_level = levels.get('DA', self.neuro.BASE_DA)

        # NE increases threshold by up to 0.2
        # DA decreases threshold by up to 0.1
        threshold = self.base_threshold + (ne_level * 0.2) - (da_level * 0.1)

        # Clamp between [0.02, 0.4] to prevent completely blocking or letting everything through
        return max(0.02, min(0.4, threshold))

    def apply_gating(self, spike_train: torch.Tensor) -> torch.Tensor:
        """
        Gelen spike_train üzerinde dinamik filtreyi in-place/GPU uyumlu uygular.
        Eşiğin altında kalan aktivasyonları 0'a indirger.
        """
        if spike_train is None:
            # BUG-05 FIX: Sessizce None döndürmek type-hint'i ihlal eder ve
            # downstream'de AttributeError üretir. Fail-fast ile erken hata fırlat.
            raise ValueError(
                "ThalamicFilter.apply_gating(): spike_train cannot be None. "
                "Ensure WernickeArea.perceive() returns a valid Tensor before calling gating."
            )

        threshold = self._calculate_dynamic_threshold()

        # In-place/Memory safe tensor masking
        filtered_spikes = torch.where(
            spike_train < threshold,
            torch.zeros_like(spike_train),
            spike_train
        )
        return filtered_spikes
