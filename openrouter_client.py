"""
OpenRouter Client - API anahtarı sadece bellekte tutulur
"""

import json
import requests
from typing import Optional, Dict, List


class OpenRouterClient:
    """OpenRouter API ile konuşur. API anahtarı bellekte tutulur, dosyaya yazılmaz."""
    
    BASE_URL = "https://openrouter.ai/api/v1/chat/completions"
    
    def __init__(self, api_key: str, site_name: str = "MergenV2"):
        self.api_key = api_key
        self.site_name = site_name
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "X-Title": self.site_name,
        }
    
    def chat(self, model: str, messages: List[Dict[str, str]], temperature: float = 0.7, max_tokens: int = 4000) -> Optional[str]:
        """Model ile konuşur, yanıtı döndürür."""
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        
        try:
            response = requests.post(self.BASE_URL, headers=self.headers, json=payload, timeout=120)
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"Hata ({model}): {e}")
            # Hata detayını yazdır
            try:
                if response.status_code == 400:
                    error_detail = response.json()
                    print(f"   → API Hata Detayı: {error_detail}")
            except:
                pass
            return None
    
    def clear_key(self):
        """API anahtarını bellekten sil."""
        self.api_key = None
        self.headers["Authorization"] = ""


# Kullanılacak ücretsiz modeller - Güncel ve çalışan modeller
FREE_MODELS = [
    "qwen/qwen3-30b-a3b",
    "qwen/qwen3-coder-480b",
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "qwen/qwen3-235b-a22b",
    "google/gemini-2.5-flash-lite",
    "meta/llama-4-scout",
    "mistralai/mistral-small-3.2",
    "qwen/qwen3-32b",
    "deepseek/deepseek-prover-v2",
    "nvidia/nemotron-3-nano-30b-a3b",
]
