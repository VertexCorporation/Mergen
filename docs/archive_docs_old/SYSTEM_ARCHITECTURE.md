# SYSTEM ARCHITECTURE

## Project Identity
**What is Mergen?**
Mergen is a biology-inspired experimental cognitive architecture and digital brain. As evidenced by the codebase (`README.md`, `language_engine.py`, `dream.py`), it is a pure Spike-Timing-Dependent Plasticity (STDP) and Hebbian learning-based neural network system. It is designed to emulate human cognitive processes using discrete brain areas (Wernicke, Broca, Limbic) and operate without relying on traditional deep learning backpropagation or external Large Language Models (LLMs) for its core cognition.

## Core Goal
**What is the intended long-term objective?**
The objective is to create a digital brain that learns, thinks, and speaks using its own simulated neurons. It aims to achieve human-like cognition, summarization, language understanding, and self-improvement through biological mechanisms such as memory replay (dreaming), trace-decay, and localized synaptic plasticity (`dream.py`, `broca_area.py`, `language_engine.py`).

## Current Working Architecture
**Describe only what currently exists in code.**
The system operates on a modular, biologically-inspired 4-layer hierarchy (`main.py`, `brain.py`):
1. **Wernicke Area:** Converts raw text/sensory input into normalized spike-train tensors (`wernicke_area.py`).
2. **MergenBrain (Hebbian Engine):** The core neural matrix handling STDP, Hebbian trace memory, and knowledge base associations (`brain.py`, `broca_area.py`).
3. **Broca Area / Language Engine:** Handles language expression. It relies on a pure Spiking Neural Network (SNN) decoder to convert motor spikes to words (`language_engine.py`), with fallback heuristic text generation if necessary (`broca_area.py`).
4. **Limbic/Executive Layer:** Provides metacognitive oversight, persistent state serialization (using XOR/Base64 encryption), and "Default Mode Network" (DMN) background threads for spontaneous neural firing and consolidation (`limbic_executive_layer.py`).

**Sub-Systems:**
- **RAG Engine:** A completely Transformer-free retrieval system utilizing character n-gram hashing (`bio_vectorizer.py`), ChromaDB for storage, and Sparse Distributed Representations (SDR) with spreading activation for ranking (`htm_retriever.py`, `rag_engine.py`). A bridge continuously strengthens synaptic bonds between concepts found in retrieved context (`hebbian_rag_bridge.py`).
- **Dream Module:** An offline memory consolidation engine that runs NREM (replaying low-confidence memories with STDP) and REM (spontaneous firing for novel associations) sleep cycles, alongside homeostatic rebalancing (`dream.py`).
- **Evolution Engine:** Systems that track architecture metrics (`auto_evolution.py`) and allow the AI to rewrite its own source files by querying external LLM APIs (`code_evolution.py`).

## Module Inventory

* **`main.py`**
  * **Purpose:** Core entry point, CLI loop, and initialization of the `MergenCognitiveArchitecture` container.
  * **Inputs:** Command line arguments, raw user text.
  * **Outputs:** Console feedback, coordinates brain processing.
  * **Dependencies:** `brain.py`, `wernicke_area.py`, `broca_area.py`, `limbic_executive_layer.py`, `intent_analyzer.py`, `auto_evolution.py`.
  * **Status:** ACTIVE

* **`brain.py`**
  * **Purpose:** The orchestrator (`MergenBrain_v7`). Contains the glue logic for all primary modules and command handling.
  * **Inputs:** Text strings, system commands.
  * **Outputs:** Generated text responses.
  * **Dependencies:** `broca_area.py`, `wernicke_area.py`, `limbic_executive_layer.py`, `intent_analyzer.py`, `auto_evolution.py`, `mergen_vocab.py`, `hebbian_rag_bridge.py`, `code_evolution.py`.
  * **Status:** ACTIVE

* **`broca_area.py`**
  * **Purpose:** Historically contained the main `MergenBrain` neural core. Currently provides the `BrocaArea` class for heuristic fallback language expression.
  * **Inputs:** Spike tensors, vocabulary indices.
  * **Outputs:** Fallback text generation strings.
  * **Dependencies:** `torch`, `numpy`.
  * **Status:** PARTIAL/ACTIVE (Some core neural components coexist with `brain.py`)

* **`wernicke_area.py`**
  * **Purpose:** Sensory encoding. Tokenizes raw text and converts it to binary PyTorch spike patterns.
  * **Inputs:** Raw text strings.
  * **Outputs:** PyTorch tensors (spike patterns).
  * **Dependencies:** `torch`, `re`.
  * **Status:** ACTIVE

* **`limbic_executive_layer.py`**
  * **Purpose:** Handles executive functions, persistent serialization (via XOR and Base64 encoding), and Default Mode Network (DMN) background threads.
  * **Inputs:** Brain state tensors, telemetry data.
  * **Outputs:** Serialized `.mx` files, background state updates.
  * **Dependencies:** `torch`, `json`, `base64`, `threading`.
  * **Status:** ACTIVE

* **`intent_analyzer.py`**
  * **Purpose:** Evaluates user intent and extracts subjects using regex and heuristic scoring.
  * **Inputs:** Text queries.
  * **Outputs:** Intent categories (e.g., GREETING, IDENTITY) and telemetry data.
  * **Dependencies:** `re`, `json`.
  * **Status:** ACTIVE

* **`auto_evolution.py`**
  * **Purpose:** Evolutionary control module for autonomous self-improvement. Monitors architecture stats and triggers scripts.
  * **Inputs:** Brain metrics.
  * **Outputs:** Subprocess calls to evolution scripts.
  * **Dependencies:** `os`, `subprocess`.
  * **Status:** ACTIVE

* **`code_evolution.py`**
  * **Purpose:** Modifies Mergen's source code by extracting it, sending it to external LLM APIs with a teaching prompt, and saving the results.
  * **Inputs:** Current source code files, grade levels/topics.
  * **Outputs:** Modified `.py` source files.
  * **Dependencies:** `re`, `ast`, `json`.
  * **Status:** ACTIVE/EXP (Experimental self-modification)

* **`rag_engine.py`**
  * **Purpose:** Transformer-free semantic search and retrieval.
  * **Inputs:** Text queries, document texts.
  * **Outputs:** Retrieved context dictionaries.
  * **Dependencies:** `bio_vectorizer.py`, `chromadb`, `htm_retriever.py`, `numpy`.
  * **Status:** ACTIVE

* **`bio_vectorizer.py`**
  * **Purpose:** Encodes text into 512-dimensional vectors using character n-gram hashing and random projection.
  * **Inputs:** Text strings.
  * **Outputs:** Float32 NumPy arrays (vectors).
  * **Dependencies:** `hashlib`, `numpy`.
  * **Status:** ACTIVE

* **`htm_retriever.py`**
  * **Purpose:** Re-ranks RAG results using Sparse Distributed Representations (SDR), spreading activation, and lateral inhibition.
  * **Inputs:** Query vectors, candidate vectors, cosine similarity scores.
  * **Outputs:** Ranked candidate indices.
  * **Dependencies:** `numpy`.
  * **Status:** ACTIVE

* **`hebbian_rag_bridge.py`**
  * **Purpose:** Bridges RAG text and the Hebbian Engine. Strengthens synaptic bonds between concepts appearing close together in retrieved texts.
  * **Inputs:** RAG document texts.
  * **Outputs:** Updates to `brain.hebbian_trace`.
  * **Dependencies:** `torch`, `re`, `threading`.
  * **Status:** ACTIVE

* **`dream.py`**
  * **Purpose:** Offline memory consolidation mimicking mammalian sleep. Replays memories (NREM), generates spontaneous associations (REM), and performs synaptic scaling.
  * **Inputs:** `mergen.mx` state file, training logs (`.npz`).
  * **Outputs:** Updated `mergen.mx` file, `dream_log.npz`.
  * **Dependencies:** `torch`, `numpy`.
  * **Status:** ACTIVE

* **`language_engine.py`**
  * **Purpose:** Pure Spiking Neural Network (SNN) language decoder. Maps motor layer spike signatures directly to words via cosine similarity and Hebbian updates, bypassing LLMs.
  * **Inputs:** Motor layer spike sequences (tensors).
  * **Outputs:** Generated natural text strings.
  * **Dependencies:** `torch`, `numpy`.
  * **Status:** ACTIVE/EXP

* **`response_synthesizer.py`**
  * **Purpose:** Composes coherent responses by summarizing extracted knowledge facts, scoring sentence relevance, and matching user intent.
  * **Inputs:** Knowledge facts, intent strings, original queries.
  * **Outputs:** Natural language strings.
  * **Dependencies:** `torch`, `re`, `random`.
  * **Status:** ACTIVE

* **`mergen_logic.py`**
  * **Purpose:** Synthesizes text by extracting key tokens, measuring intellect level, and adding distinct Mergen commentary.
  * **Inputs:** Raw text content.
  * **Outputs:** Synthesized commentary strings.
  * **Dependencies:** `re`, `collections`, `random`.
  * **Status:** UNUSED/PARTIAL (Active logic, but integration seems superseded by `response_synthesizer.py`).

## Data Flow
1. **Sensory Input:** User text enters `WernickeArea`, converting to a PyTorch spike pattern (`wernicke_area.py`).
2. **Analysis:** `IntentAnalyzer` parses the text to determine intent and subject (`intent_analyzer.py`). 
3. **Retrieval:** If factual memory is needed, `RAGEngine` retrieves memories using `BioVectorizer` and `HTMRetriever` (`rag_engine.py`). Background `HebbianRAGBridge` strengthens relevant synaptic traces (`hebbian_rag_bridge.py`).
4. **Cognitive Processing:** The `MergenBrain` orchestrates these spikes, triggering matrix activations, STDP updates, and modifying the Hebbian trace (`brain.py`).
5. **Motor Output:** Activated motor spike patterns pass to the `LanguageEngine` (or `BrocaArea`) to decode into discrete words (`language_engine.py`, `broca_area.py`).
6. **Synthesis:** `ResponseSynthesizer` forms a coherent final response from the generated words, intents, and facts (`response_synthesizer.py`).
7. **Consolidation:** Offline, `dream.py` loads the saved state, replays logs, and updates core weights via NREM/REM cycles (`dream.py`).

## Runtime Flow
**Startup Sequence (`main.py`, `brain.py`):**
1. System boots via `MergenCognitiveArchitecture`.
2. `LimbicSystem` is initialized and attempts to load persistent state from `mergen_weights.mx`.
3. Architecture dynamically initializes its sub-components: `WernickeArea`, `BrocaArea`, `IntentAnalyzer`, `LanguageEngine`, and `RAGEngine`.
4. Command line loop or REPL begins listening for user input.
5. The Default Mode Network (DMN) background thread (`LimbicSystem`) spins up to handle idle brain consolidation.

**Execution Flow (Per Query):**
1. Raw query is routed to `MergenBrain_v7.process_input()`.
2. `IntentAnalyzer` classifies the query.
3. Information is fetched from the knowledge base and `RAGEngine`.
4. `WernickeArea` spikes the input.
5. Synaptic traces update via localized Hebbian plasticity.
6. A response is formulated by the `LanguageEngine` (from motor spikes) or `ResponseSynthesizer` (from facts).
7. `LimbicSystem` serializes the updated state to disk asynchronously.

## Memory Systems
- **`mergen_weights.mx`:** Encrypted persistent state storing PyTorch neural network weights, synaptic traces, and firing rates (`limbic_executive_layer.py`).
- **`mergen_matrix_memory.json`:** Telemetry, conversational context, and intent state.
- **`mergen_rag_db/`:** Local ChromaDB persistent directory housing vector representations of indexed knowledge (`rag_engine.py`).
- **`logs/*.npz`:** Episodic spike pattern logs utilized by the offline Dream module (`dream.py`).

## Learning Systems
- **STDP (Spike-Timing-Dependent Plasticity):** Synaptic weights are modified dynamically based on pre- and post-synaptic firing timing, predominantly during memory replay in `dream.py` and active decoding in `language_engine.py`.
- **Hebbian Learning & Trace-Decay:** Short-term memory decay and the strengthening of co-occurring concepts ("cells that fire together, wire together") handled in `hebbian_rag_bridge.py` and `broca_area.py`.
- **Reward Modulation:** Dynamic learning rate adjustments based on confidence scores during memory consolidation (`dream.py`).

## Reasoning Systems
- **Spreading Activation & Lateral Inhibition:** Employs biological attention mechanisms via Sparse Distributed Representations (SDR) to contextually associate memories and filter duplicates (`htm_retriever.py`).
- **Response Synthesis:** Extracts information density and scores sentence relevance heuristically to piece together logical answers without relying on an autoregressive language model (`response_synthesizer.py`, `mergen_logic.py`).

## Evolution Systems
- **Auto Evolution (`auto_evolution.py`):** Monitors network metrics (like node count or performance) and triggers scheduled evolution scripts.
- **Code Evolution (`code_evolution.py`):** An experimental module enabling the architecture to rewrite its own source code by communicating with an external "Teacher AI", reading its Python files, applying AST-verified LLM modifications, and saving back to disk.

## External Dependencies
- **Primary Libraries:** `torch`, `numpy`
- **Retrieval:** `chromadb` (used if available, degrades gracefully if not)
- **Standard Library:** `json`, `base64`, `re`, `threading`, `ast`, `hashlib`
- **APIs:** Does not rely on external cloud APIs for core cognitive execution or generation, with the sole exception of the experimental `code_evolution.py` module which makes network requests to external LLMs for code improvements.
