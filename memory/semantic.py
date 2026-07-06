from typing import List, Dict, Any, Optional

class SemanticMemory:
    """
    Semantik Hafıza: Zamandan bağımsız, değişmez gerçekleri (facts) depolar.
    Örnek: "Paris Fransa'nın başkentidir."
    """
    def __init__(self):
        # Format: {'text': str, 'concept_ids': List[int], 'weight': float, 'access_count': int, 'memory_type': 'semantic'}
        self.knowledge_base: List[Dict[str, Any]] = []

    def add_fact(self, text: str, concept_ids: List[int], weight: float = 1.0) -> int:
        """Yeni bir gerçek ekler. Eğer benzer bir gerçek varsa ağırlığını artırır."""
        # Basit bir deduplication mantığı (Eğer tamamen aynıysa)
        for fact in self.knowledge_base:
            if fact['text'].lower() == text.lower():
                fact['weight'] = max(fact.get('weight', 0.0), weight)
                fact['access_count'] = fact.get('access_count', 0) + 1
                return len(self.knowledge_base)
                
        self.knowledge_base.append({
            'text': text,
            'concept_ids': concept_ids,
            'weight': weight,
            'access_count': 0,
            'memory_type': 'semantic'
        })
        return len(self.knowledge_base)

    def to_list(self) -> List[Dict]:
        return self.knowledge_base

    def from_list(self, data: List[Dict]):
        self.knowledge_base = data
        
    def __len__(self):
        return len(self.knowledge_base)
