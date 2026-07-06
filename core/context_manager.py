"""
Context Manager and Summarizer for Mergen
"""

import re
from collections import Counter
from typing import List, Dict, Any


class ContextManager:
    """Manages context and provides summarization capabilities."""
    
    def __init__(self):
        self.contexts: List[Dict[str, Any]] = []
    
    def add_context(self, content: str, context_type: str = "text", metadata: Dict[str, Any] = None) -> None:
        """
        Add a context item.
        
        Args:
            content: The text/content to store
            context_type: Type of context (text, code, etc.)
            metadata: Additional metadata
        """
        context_item = {
            "content": content,
            "type": context_type,
            "metadata": metadata or {},
            "timestamp": self._get_timestamp()
        }
        self.contexts.append(context_item)
    
    def get_contexts(self, limit: int = None) -> List[Dict[str, Any]]:
        """
        Get stored contexts.
        
        Args:
            limit: Maximum number of contexts to return (None for all)
            
        Returns:
            List of context items
        """
        if limit is None:
            return self.contexts.copy()
        return self.contexts[-limit:].copy()
    
    def clear_contexts(self) -> None:
        """Clear all stored contexts."""
        self.contexts.clear()
    
    def summarize_text(self, text: str, max_sentences: int = 3, language: str = "tr", stop_words: set = None, min_word_length: int = 2) -> str:
        """
        Summarize text by extracting key sentences.
        
        Args:
            text: Text to summarize
            max_sentences: Maximum number of sentences in summary
            
        Returns:
            Summarized text
        """
        if not text or not text.strip():
            return ""
        
        # Clean text
        text = text.strip()
        
        # Split into sentences (simple regex for Turkish and English)
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) <= max_sentences:
            return text
        
        # Simple word frequency scoring (excluding language-specific stop words)
        # Use provided stop_words or default set for the specified language
        default_stop_words = {
            "tr": {
                'bir', 'bu', 'da', 'de', 'ile', 'için', 'ki', 'mi', 'mı', 'mu', 'mü',
                've', 'veya', 'ama', 'fakat', 'ancak', 'lakin', 'asla', 'hiç', 'her',
                'bu', 'şu', 'o', 'böyle', 'şöyle', 'böylece', 'şöylece'
            },
            "en": {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is',
                   'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
                   'do', 'does', 'did', 'will', 'would', 'should', 'could', 'may', 'might',
                   'must', 'can', 'this', 'that', 'these', 'those', 'a', 'an'},
            "de": {'der', 'die', 'das', 'und', 'oder', 'aber', 'in', 'auf', 'zu', 'für', 'mit', 'von',
                   'ist', 'sind', 'war', 'waren', 'sein', 'haben', 'hat', 'hatte', 'wird', 'kann',
                   'dies', 'das', 'diese', 'jene', 'ein', 'eine'},
            "fr": {'le', 'la', 'les', 'et', 'ou', 'mais', 'dans', 'à', 'pour', 'avec', 'de', 'par',
                   'est', 'sont', 'était', 'étaient', 'être', 'avoir', 'a', 'ont', 'peut', 'ce',
                   'cette', 'ces', 'un', 'une'}
        }
        if stop_words is None:
            stop_words = default_stop_words.get(language, set())
        
        # Calculate word frequencies
        words = re.findall(r'\b\w+\b', text.lower())
        word_freq = Counter(w for w in words if w not in stop_words and len(w) >= min_word_length)
        
        # Score sentences
        sentence_scores = []
        for i, sentence in enumerate(sentences):
            sentence_words = re.findall(r'\b\w+\b', sentence.lower())
            score = sum(word_freq.get(word, 0) for word in sentence_words)
            sentence_scores.append((score, i, sentence))
        
        # Sort by score and position (to maintain order)
        sentence_scores.sort(key=lambda x: (-x[0], x[1]))
        
        # Take top sentences and sort by original order
        top_sentences = sorted(sentence_scores[:max_sentences], key=lambda x: x[1])
        summary = '. '.join(s[2] for s in top_sentences)
        
        # Ensure proper ending
        if summary and not summary.endswith('.'):
            summary += '.'
            
        return summary
    
    def _get_timestamp(self) -> str:
        """Get current timestamp string."""
        from datetime import datetime
        return datetime.now().isoformat()


# Global context manager instance
context_manager = ContextManager()


def summarize_content(content: str, max_sentences: int = 3, language: str = "tr", stop_words: set = None, min_word_length: int = 2) -> str:
    """
    Convenience function to summarize content using the global context manager.
    
    Args:
        content: Content to summarize
        max_sentences: Maximum number of sentences in summary
        
    Returns:
        Summarized content
    """
    return context_manager.summarize_text(content, max_sentences, language, stop_words, min_word_length)


def summarize_contexts(limit: int = 5, max_sentences: int = 3, language: str = "tr", stop_words: set = None, min_word_length: int = 2) -> str:
    """Summarize the most recent contexts up to `limit`.

    Args:
        limit: Number of recent contexts to include.
        max_sentences: Max sentences per summary.
        language: Language code for stop words.
        stop_words: Optional custom stop word set.
        min_word_length: Minimum word length to consider.
    """
    recent = context_manager.get_contexts(limit)
    combined = " ".join(item["content"] for item in recent)
    return context_manager.summarize_text(combined, max_sentences, language, stop_words, min_word_length)


class SummarizerConfig:
    """Configuration holder for summarizer parameters."""
    def __init__(self, max_sentences: int = 3, language: str = "tr", stop_words: set = None, min_word_length: int = 2):
        self.max_sentences = max_sentences
        self.language = language
        self.stop_words = stop_words
        self.min_word_length = min_word_length


if __name__ == "__main__":
    # Test the summarizer
    test_text = """
    Mergen yapay zeka projesi, sürekli öğrenme ve kod geliştirme yeteneğine sahip bir sistemdir.
    Bu sistem, 1. sınıftan 12. sınıfa kadar olan seviyelerde çalışarak kendini geliştirir.
    Her seviyede yeni beceriler öğrenir ve önceki bilgileri birleştirir.
    Sistem, beynin farklı bölgelerini simüle ederek çalışır.
    Örneğin, hipokampus uzun süreli bellek için, korteks kısa süreli işlemler için kullanılır.
    Bu sayede Mergen, deneyimlerden öğrenerek zaman içinde daha akıllı hale gelir.
    """
    
    print("Original text:")
    print(test_text)
    print("\nSummary:")
    print(summarize_content(test_text, max_sentences=2))