"""
╔══════════════════════════════════════════════════════════════════════╗
║         MERGEN — CONVERSATION MEMORY (Episodic Context)              ║
║                                                                      ║
║  "Mergen remembers what you said — and builds on it."               ║
║                                                                      ║
║  Author:  Vertex Corporation — Mergen Project                       ║
║  License: Apache-2.0                                                ║
╚══════════════════════════════════════════════════════════════════════╝

This module provides multi-turn conversation context for Mergen.
Unlike the IntentAnalyzer's context_buffer (which only resolves pronouns),
this module:

  • Stores full conversation turns (user input + Mergen response)
  • Tracks topic evolution over time
  • Resolves pronouns and references ("bu", "o", "bu konu")
  • Provides context-aware recall for response generation
  • Maintains a rolling window of recent exchanges
"""

import re
import json
import time
import threading  # BUG-07 FIX
from pathlib import Path
from collections import deque
from typing import Optional, Dict, List, Tuple
from datetime import datetime


class ConversationMemory:
    """
    Multi-turn episodic conversation memory for Mergen.

    Stores recent conversation turns and provides context resolution
    for pronouns, references, and topic continuity.
    """

    # Pronouns that refer back to previous context
    REFERENCE_PRONOUNS = {
        'o', 'onu', 'ona', 'onun', 'onda', 'ondan',
        'bu', 'bunu', 'buna', 'bunda', 'bundan',
        'şu', 'şunu', 'şuna', 'şunda', 'şundan',
        'bunlar', 'onlar', 'şunlar',
        'it', 'this', 'that', 'these', 'those',
        'him', 'her', 'them',
    }

    # Topic-tracking keywords
    TOPIC_STOP_WORDS = {
        'bir', 've', 'ile', 'bu', 'şu', 'o', 'bunu', 'şunu', 'onu',
        'ben', 'sen', 'biz', 'siz', 'onlar', 'için', 'gibi', 'kadar',
        'de', 'da', 'ki', 'mi', 'mı', 'mu', 'mü', 'ne', 'nasıl', 'niye',
        'neden', 'ama', 'fakat', 'lakin', 'çünkü',
        'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
        'and', 'or', 'but', 'for', 'with', 'of', 'in', 'to', 'from',
        'i', 'you', 'he', 'she', 'we', 'they', 'this', 'that',
        'what', 'how', 'why', 'when', 'where', 'who',
    }

    def __init__(
        self,
        window_size: int = 20,
        persistence_path: str = './mergen_conversation_memory.json',
    ):
        self.window_size = window_size
        self.persistence_path = Path(persistence_path)

        # Rolling conversation turns
        self.turns: deque = deque(maxlen=window_size)

        # Current topic tracking
        self.current_topics: List[str] = []
        self.topic_history: List[List[str]] = []

        # Last entities mentioned
        self.last_subject: Optional[str] = None
        self.last_object: Optional[str] = None
        self.last_action: Optional[str] = None

        # Telemetry
        self.total_turns = 0
        self.total_resolutions = 0

        # BUG-07 FIX: thread-safe erişim için RLock.
        # DMN thread'i veya başka bağlamdan eş zamanlı add_turn() çağrısında
        # JSON bozulması yaşanmamasını garantiler.
        self._lock = threading.RLock()

        # Load previous session if available
        self._load()

    def add_turn(
        self,
        user_input: str,
        response: str,
        intent: str = 'UNKNOWN',
        subject: Optional[str] = None,
    ):
        """Record a conversation turn."""
        # BUG-07 FIX: Lock ile koru — eş zamanlı append + save güvenli
        with self._lock:
            turn = {
                'timestamp': time.time(),
                'user_input': user_input[:500],
                'response': response[:500],
                'intent': intent,
                'subject': subject,
                'turn_number': self.total_turns,
            }
            self.turns.append(turn)
            self.total_turns += 1

            # Extract and update topics
            topics = self._extract_topics(user_input)
            if topics:
                self.current_topics = topics
                self.topic_history.append(topics)

            # Update last entities
            if subject:
                self.last_subject = subject

            # Persist every 5 turns
            if self.total_turns % 5 == 0:
                self._save()

    def resolve_references(self, text: str) -> str:
        """
        Replace pronouns in the current input with actual entities
        from conversation history.

        Example:
            Input: "O nedir?" (after discussing "kuantum")
            Output: "kuantum nedir?"
        """
        words = re.findall(r'\w+', text.lower())
        if not words:
            return text

        resolved_parts = []
        original_words = re.findall(r'\w+|[^\w\s]', text)

        for word in original_words:
            word_lower = word.lower()
            if word_lower in self.REFERENCE_PRONOUNS:
                # Try to resolve from recent context
                resolved = self._resolve_single(word_lower)
                if resolved:
                    self.total_resolutions += 1
                    resolved_parts.append(resolved)
                else:
                    resolved_parts.append(word)
            else:
                resolved_parts.append(word)

        # Reconstruct text
        return ''.join(
            part if part in ('.', ',', '?', '!', ':', ';', ' ')
            else (' ' + part if resolved_parts[idx-1] not in ('.', ',', '?', '!', ':', ';', ' ') and idx > 0 else part)
            for idx, part in enumerate(resolved_parts)
        ).strip()

    def _resolve_single(self, pronoun: str) -> Optional[str]:
        """Resolve a single pronoun to an entity."""
        # Check recent turns for concrete subjects
        for turn in reversed(self.turns):
            subj = turn.get('subject')
            if subj and subj.lower() not in self.REFERENCE_PRONOUNS:
                return subj

        # Fall back to current topics
        if self.current_topics:
            return self.current_topics[-1]

        return self.last_subject

    def get_context_summary(
        self,
        max_turns: int = 5,
    ) -> Dict:
        """
        Get a summary of recent conversation context for the brain.
        """
        recent_turns = list(self.turns)[-max_turns:]

        return {
            'recent_turns': [
                {
                    'user': t['user_input'][:100],
                    'response': t['response'][:100],
                    'intent': t['intent'],
                }
                for t in recent_turns
            ],
            'current_topics': self.current_topics,
            'last_subject': self.last_subject,
            'total_turns': self.total_turns,
        }

    def find_related_turns(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        Find past conversation turns related to the current query.
        Uses simple keyword overlap.
        """
        query_tokens = set(re.findall(r'\w+', query.lower())) - self.TOPIC_STOP_WORDS
        if not query_tokens:
            return []

        scored_turns = []
        for turn in self.turns:
            turn_tokens = set(
                re.findall(r'\w+', turn['user_input'].lower())
            ) | set(
                re.findall(r'\w+', turn['response'].lower())
            )
            overlap = len(query_tokens & turn_tokens)
            if overlap > 0:
                scored_turns.append((overlap, turn))

        scored_turns.sort(key=lambda x: -x[0])
        return [t for _, t in scored_turns[:top_k]]

    def _extract_topics(self, text: str) -> List[str]:
        """Extract topic keywords from text."""
        words = re.findall(r'\w+', text.lower())
        topics = [
            w for w in words
            if w not in self.TOPIC_STOP_WORDS and len(w) > 2
        ]
        return topics[:5]

    def _save(self):
        """Persist conversation memory to disk."""
        # BUG-07 FIX: _lock zaten add_turn içinden alınmış olabilir (re-entrant).
        # Atomik write (tmp→rename) JSON dosyasının yarım kalmasını önler.
        with self._lock:
            try:
                data = {
                    'turns': list(self.turns),
                    'current_topics': self.current_topics,
                    'topic_history': self.topic_history[-50:],
                    'last_subject': self.last_subject,
                    'total_turns': self.total_turns,
                    'total_resolutions': self.total_resolutions,
                }
                json_str = json.dumps(data, ensure_ascii=False, indent=2)
                # Atomik write: geçici dosyaya yaz, sonra rename et
                tmp_path = self.persistence_path.with_suffix('.json.tmp')
                tmp_path.write_text(json_str, encoding='utf-8')
                tmp_path.replace(self.persistence_path)
            except Exception as e:
                print(f"[ConversationMemory] Save error: {e}")

    def _load(self):
        """Load conversation memory from disk."""
        if self.persistence_path.exists():
            try:
                data = json.loads(
                    self.persistence_path.read_text(encoding='utf-8')
                )
                self.current_topics = data.get('current_topics', [])
                self.topic_history = data.get('topic_history', [])
                self.last_subject = data.get('last_subject')
                self.total_turns = data.get('total_turns', 0)
                self.total_resolutions = data.get('total_resolutions', 0)

                # Restore turns
                for turn in data.get('turns', []):
                    self.turns.append(turn)
            except Exception as e:
                print(f"[ConversationMemory] Load error: {e}")

    def get_telemetry(self) -> Dict:
        return {
            'total_turns': self.total_turns,
            'total_resolutions': self.total_resolutions,
            'current_topics': self.current_topics,
            'window_size': self.window_size,
            'active_turns': len(self.turns),
        }

    def clear(self):
        """Clear all conversation memory."""
        self.turns.clear()
        self.current_topics = []
        self.topic_history = []
        self.last_subject = None
        self.last_object = None
        self.last_action = None
        self.total_turns = 0
        self.total_resolutions = 0
        self._save()
