# MERGEN.PY RUNTIME FLOW

**Date:** 2026-06-19
**Auditor:** Principal AI Systems Architect
**Target:** Architecture A (`Mergen.py` → `brain.py`)

This document traces the complete execution path of the active digital brain from startup to a conversational response.

---

## 1. Boot Sequence

The entry point is `python Mergen.py`. This script modifies `sys.path` to include its parent directory, imports `MergenBrain_v7` from `brain.py`, instantiates it with `verbose=True`, and calls `.run()`.

### 1.1 Initialization (`__init__`)
When `MergenBrain_v7` initializes, it loads or creates the following subsystems in order:

1.  **Config:** Instantiates `MergenConfig`.
2.  **Vocabulary:** Attempts to load `mergen_vocab.json`. If missing, builds a fresh `MergenVocab` instance. Sets `config.OUTPUT_SIZE` to match the vocabulary size.
3.  **Neural Core:** Instantiates `MergenBrain` (from `broca_area.py`).
4.  **Weights Loading:** Attempts to load `./mergen_weights.mx`. If present, populates `MergenBrain`'s `knowledge_base` and `nn.Linear` layers.
5.  **Wernicke Wrapper:** Wraps `MergenBrain` in `EnhancedMergenBrain` (which fails silently if `sentence-transformers` is missing, falling back to basic processing).
6.  **Intent Analyzer:** Instantiates `IntentAnalyzer`, loading telemetry from `./mergen_matrix_memory.json`.
7.  **Conversation Memory:** Instantiates `ConversationMemory`, loading history from `./mergen_conversation_memory.json`.
8.  **Response Generator:** Instantiates `ResponseGenerator`.
9.  **Broca Area (Fallback):** Attempts to instantiate `LanguageEngine`. If unavailable, creates a `StubEngine` and passes it to `BrocaArea`.
10. **Turkish Morphology:** Attempts to instantiate `TurkishMorph` (requires `zeyrek`).
11. **RAG Engine:** Attempts to load ChromaDB from `./mergen_rag_db`. If successful, instantiates `HebbianRAGBridge` and `TurkishDataLoader`.
12. **Signal Handler:** Registers `_signal_shutdown` to intercept `SIGINT` (Ctrl+C).

### 1.2 Interactive Loop (`run`)
After initialization, the `.run()` method starts a blocking `while self.is_running:` loop:
1.  Prompts the user with `Sen > `.
2.  Filters out empty inputs.
3.  Checks for special commands (`/exit`, `/stats`, `/clear`, `/introspect`, `oku:`, `rag:yukle`, `rag:durum`, `/help`) via `handle_command()`.
4.  If not a command, passes the text to `self.respond(user_input)`.
5.  Prints the result prefixed with `Mergen > `.

---

## 2. Response Pipeline (`respond()`)

When a user submits standard text, the `respond()` method executes a 9-step pipeline.

### Step 1: Pronoun Resolution
`self.conv_memory.resolve_references(user_input)`
Replaces pronouns (e.g., "o", "onu") based on the subjects of recent conversational turns.

### Step 2: Intent Analysis
`self.analyzer.analyze_intent(resolved_input)`
Uses regex heuristics (keyword matching) to determine the user's intent (e.g., GREETING, INQUIRY, COMMAND). Extracts a `primary_intent` and a `subject` noun. Wrapped in `try/except: pass`.

### Step 3: Brain Processing
`self.enhanced_brain.process_with_intent(text, intent_report)`
Computes a forward pass. 
- If `WernickeArea` is active, computes semantic embeddings.
- Otherwise, runs the text tokens through `MergenBrain.process()` (a forward pass through the frozen `mx1` and `mx2` linear layers).
Returns a `neural_intent` tensor. Wrapped in `try/except: pass`.

### Step 4: Passive Learning
`self.brain.learn_from_text(resolved_input, ...)`
Performs a Hebbian trace update. Increments the `hebbian_trace` buffer for any words in the input that exist in the vocabulary, and applies exponential decay (`trace *= 0.95`). Wrapped in `try/except: pass`.

### Step 4b: Active Learning
`self._try_learn_from_input(resolved_input, intent)`
Checks if the input is a declarative statement (not a question or command, >3 words). If so, it appends the sentence as a dictionary to `self.brain.knowledge_base` and immediately calls `self.brain.save()` to write `mergen_weights.mx`. If a fact was learned, the pipeline short-circuits and returns an acknowledgment string ("Bunu öğrendim: ...").

### Step 5: Knowledge Recall
`self._recall_knowledge(resolved_input, intent, subject)`
If no new fact was learned, Mergen attempts to find relevant knowledge using 5 strategies:
1.  **Subject-based:** Search KB for `subject`.
2.  **Raw text:** Search KB for `resolved_input`.
3.  **Broad match:** Search KB for `subject` (fallback).
4.  **Semantic:** Use `WernickeArea` similarity (if active).
5.  **RAG:** Query ChromaDB using `BioVectorizer` and `HTM` re-ranking.
Collects up to 8 unique facts and sorts them by relevance.

### Step 6: Context Retrieval
`self.conv_memory.get_context_summary(max_turns=5)`
Retrieves the last 5 conversation turns to maintain dialogue coherence.

### Step 7: Response Synthesis
`self.generator.generate(query, intent, subject, knowledge_facts, conversation_context)`
Constructs a string response based on the intent.
- If `GREETING`, picks a random greeting template.
- If `INQUIRY`, attempts to format the retrieved `knowledge_facts` into an answer.
If `ResponseGenerator` fails to produce >= 3 characters, execution falls back to `self._broca_generate()`, which uses the `LanguageEngine` (or `StubEngine`) to generate text based on the `neural_intent` tensor. If Broca fails, returns "Anlayamadım, tekrar eder misin?".

### Step 8: Memory Update
`self.conv_memory.add_turn(...)`
Saves the user input, the generated response, the intent, and the subject into the short-term memory buffer.

### Step 9: Interaction Logging
Appends the turn metadata to `self.interaction_log` (a Python list).

Returns the response string to the `run()` loop to be printed.

---

## 3. File Ingestion Flow (`oku:file.txt`)

When the user enters `oku:filename.txt`, the `handle_command` method calls `self.ingest_file()`.
1.  Reads the file (trying UTF-8, then Latin-1).
2.  Splits text into paragraphs and sentences.
3.  Loops over the first 60 paragraphs: calls `learn_from_text` (saving to KB, reward=1.5).
4.  Loops over the first 100 sentences: calls `learn_from_text` (saving to KB, reward=0.8).
5.  Extracts key concepts (most frequent words ignoring stop words).
6.  Generates "Summary Facts" (e.g., "X konusu hakkında hafızamda bilgiler var") and injects them into the KB so Mergen has meta-knowledge of having read the file.
7.  Updates Hebbian-RAG traces if RAG is active.
8.  Saves `mergen_weights.mx`.
9.  Starts a background thread (`self._start_reflection`) that runs `_reflect_in_background()`. This thread searches the KB for co-occurring concepts across different facts and increments their `hebbian_trace` concurrently.

---

## 4. Persistence Flow (Shutdown)

Triggered by `/exit` or `Ctrl+C`. `self.shutdown()` is called.

1.  **Vocab:** Saves `mergen_vocab.json`.
2.  **Brain Weights:** Saves `mergen_weights.mx`. This file contains the serialized KB array and the PyTorch `state_dict` for the frozen Linear layers.
3.  **Intent Memory:** Saves `mergen_matrix_memory.json`.
4.  **Interaction Log:** Writes `mergen_interactions.json`.

*(Note: `ConversationMemory` auto-saves to `mergen_conversation_memory.json` on every turn).*

The process then exits with `sys.exit(0)`.
