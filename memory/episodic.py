from typing import List, Dict, Any
from datetime import datetime

class EpisodicMemory:
    """
    Episodik Hafıza: Zaman damgası ve bağlam odaklı anıları depolar.
    Örnek: "Kullanıcı bana saat 14:00'te X dedi."
    """
    def __init__(self):
        # Format: {'timestamp': str, 'text': str, 'concept_ids': List[int], 'weight': float, 'access_count': int, 'memory_type': 'episodic'}
        self.events: List[Dict[str, Any]] = []

    def add_event(self, text: str, concept_ids: List[int], weight: float = 1.0) -> int:
        """Yeni bir anı ekler. Eklenen öğenin 0-tabanlı indeksini döndürür."""
        self.events.append({
            'timestamp': datetime.now().isoformat(),
            'text': text,
            'concept_ids': concept_ids,
            'weight': weight,
            'access_count': 0,
            'memory_type': 'episodic'
        })
        return len(self.events) - 1  # BUG-02/SORUN-06 FIX: boyut değil, 0-tabanlı indeks

    def prune_old_events(self):
        """Önemsiz veya çok eski anıları temizler."""
        # weight < 0.5 ve access_count == 0 olanları silebiliriz.
        # veya FIFO mantığı yapabiliriz. Şimdilik basitçe aktarılmayan veya değersizleri silelim.
        filtered_events = [e for e in self.events if e.get('weight', 1.0) >= 0.5 or e.get('access_count', 0) > 0]
        self.events = filtered_events
        
    def clear(self):
        self.events = []

    def to_list(self) -> List[Dict]:
        return self.events

    def from_list(self, data: List[Dict]):
        self.events = data
        
    def __len__(self):
        return len(self.events)
