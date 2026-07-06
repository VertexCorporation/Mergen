"""
Test script for context_manager.py
"""

from context_manager import summarize_content, ContextManager

def test_summarizer():
    # Test text
    test_text = """
    Mergen yapay zeka projesi, sürekli öğrenme ve kod geliştirme yeteneğine sahip bir sistemdir.
    Bu sistem, 1. sınıftan 12. sınıfa kadar olan seviyelerde çalışarak kendini geliştirir.
    Her seviyede yeni beceriler öğrenir ve önceki bilgileri birleştirir.
    Sistem, beynin farklı bölgelerini simüle ederek çalışır.
    Örneğin, hipokampus uzun süreli bellek için, korteks kısa süreli işlemler için kullanılır.
    Bu sayede Mergen, deneyimlerden öğrenerek zaman içinde daha akıllı hale gelir.
    """
    
    print("Testing summarizer...")
    summary = summarize_content(test_text, max_sentences=2)
    print(f"Original length: {len(test_text)}")
    print(f"Summary length: {len(summary)}")
    print(f"Summary: {summary}")
    
    # Test ContextManager
    cm = ContextManager()
    cm.add_context(test_text, "test")
    contexts = cm.get_contexts()
    print(f"\nContextManager test: {len(contexts)} contexts stored")
    
    # Test empty text
    empty_summary = summarize_content("")
    print(f"Empty text summary: '{empty_summary}'")
    
    # Test short text
    short_text = "Bu kısa bir metin."
    short_summary = summarize_content(short_text, max_sentences=2)
    print(f"Short text summary: '{short_summary}'")

if __name__ == "__main__":
    test_summarizer()