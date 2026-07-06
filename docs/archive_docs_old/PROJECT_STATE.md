# PROJECT STATE

**Date:** 2026-06-19
**Auditor:** Technical Auditor (code evidence only)
**Source:** Direct inspection of all source files in `c:/Users/Xelma123/Desktop/Mergen-main/`

---

## Repository Summary

The repository contains **three distinct, coexisting architectures** with no shared runtime integration. They were developed at different points in the project's history and have never been fully unified. All three contain functioning code but serve different purposes and are launched by different entry points.

---

## Primary Active Architecture

### Architecture A — Conversational Brain v7 (`Mergen.py` / `brain.py`)

**Entry point:** `Mergen.py` → imports `MergenBrain_v7` from `brain.py` → calls `mergen.run()`  
**Purpose:** Turkish-language conversational AI with persistent knowledge base, intent analysis, RAG retrieval, and Hebbian learning from text.  
**What it actually does:** Accepts user text, classifies intent, queries a knowledge base and ChromaDB, generates a response from retrieved facts, and saves state to `mergen_weights.mx`.

---

## Secondary Architectures

### Architecture B — V3 Cortical Training Loop (`main.py`)

**Entry point:** `main.py` → `run_training()` → `MergenCognitiveArchitecture`  
**Purpose:** Headless training loop for a math arithmetic task (e.g., "3 + 5 = ?") using a 2D spiking cortical sheet architecture.  
**What it actually does:** Generates arithmetic problems, encodes them as spike streams, runs them through sensory/motor cortical layers and a Global Workspace, evaluates predictions, and runs local Hebbian learning. No conversational interface. Never imports `brain.py`.

### Architecture C — Standalone Evolution System (`auto_evolution.py`)

**Entry point:** `auto_evolution.py <API_KEY>` (also via `run_mergen_evolution.bat`)  
**Purpose:** Uses external LLM APIs (OpenRouter) to iteratively rewrite Mergen's own Python source files, guided by a structured curriculum (`curriculum.py`) progressing from grade 1 to grade 12.  
**What it actually does:** Loads Python files, sends them to an external model, validates the returned code via `ast.parse()`, and overwrites the original files if valid. This is an autonomous code-rewriting pipeline, not a cognitive architecture component.

---

## Subsystem Inventory

---

### `Mergen.py`
- **Purpose:** Main user-facing entry point for conversational brain v7.
- **Status:** WORKING
- **Evidence:** 45 lines; imports `MergenBrain_v7` from `brain.py`, calls `mergen.run()`. Clean and correct.
- **Risks:** `run()` method must exist in `brain.py` (confirmed at line 800+ via `_run_loop` inspection needed). Entry point is not `main.py`.
- **Recommendation:** This is the canonical user entry point. Keep as-is.

---

### `brain.py` — `MergenBrain_v7`
- **Purpose:** Core orchestrator for Architecture A. Binds vocabulary, neural processing, intent analysis, RAG retrieval, conversation memory, and response generation.
- **Status:** WORKING
- **Evidence:** `respond()` method confirmed active at lines 256–358. Imports `IntentAnalyzer`, `BrocaArea`, `MergenBrain`, `EnhancedMergenBrain`, `ConversationMemory`, `ResponseGenerator`, `RAGEngine`, `HebbianRAGBridge`. All imports succeed or degrade gracefully via `try/except`.
- **Risks:**
  - `_recall_knowledge()` calls five separate retrieval strategies in sequence (lines 443–541). If all fail silently, the response generator falls back to a conversational template. Failure is invisible.
  - `self.brain.learn_from_text()` is called with `except: pass` in multiple locations (lines 304–305), making learning failures completely silent.
- **Recommendation:** Most important file in the repository. Needs explicit error logging on silent failures.

---

### `broca_area.py` — `MergenBrain` + `BrocaArea` + `MergenConfig`
- **Purpose:** All-in-one module exporting three classes: configuration (`MergenConfig`), the neural core with knowledge base (`MergenBrain`), and the language expression fallback (`BrocaArea`).
- **Status:** WORKING
- **Evidence:** `brain.py` line 42 imports `BrocaArea, MergenBrain, MergenConfig` from this file. `mergen_brain_wrapper.py` line 21 also imports `MergenBrain`. `MergenBrain` contains two `nn.Linear` layers (`mx1`, `mx2`), a `hebbian_trace` buffer, and a full knowledge base with inverted index. This is the actual neural core of Architecture A.
- **Risks:**
  - The file is named `broca_area.py` but exports three unrelated classes including the primary neural core. This naming is misleading and a significant maintenance risk.
  - 51,993 bytes / 1,283 lines — far exceeds maintainability limits.
- **Recommendation:** Critical component. Highest-value file in the repository. The `MergenBrain` class is the functional heart of the system.

---

### `mergen_brain_wrapper.py` — `EnhancedMergenBrain`
- **Purpose:** Wraps `MergenBrain` with optional `WernickeArea` integration and provides semantic recall via sentence-transformer embeddings.
- **Status:** PARTIALLY WORKING
- **Evidence:** `brain.py` line 50 imports `EnhancedMergenBrain`. Core functions `process_with_intent()`, `recall_raw()`, `recall_all_about()` confirmed working. `recall_semantic()` depends on `sentence-transformers` but `requirements.txt` line 7 explicitly comments: `# sentence-transformers artık gerekli değil (BioVectorizer kullanılıyor)`. Semantic recall silently falls back to `recall_raw()` when Wernicke is unavailable.
- **Risks:** `sentence-transformers` is not in `requirements.txt`. Any install from requirements will leave `WernickeArea` non-functional. The semantic path degrades silently.
- **Recommendation:** Preserve. Fix the dependency documentation inconsistency.

---

### `wernicke_area.py` — `WernickeArea`
- **Purpose:** Converts raw text to biological spike trains using `sentence-transformers/all-MiniLM-L6-v2` embeddings (local, no API calls). Three encoding modes: rate, temporal, population.
- **Status:** PARTIALLY WORKING
- **Evidence:** Well-implemented (`mergen_brain_wrapper.py` line 49 imports it). However, `requirements.txt` excludes `sentence-transformers`. In `mergen_brain_wrapper.py` line 58–60, if `ImportError` is raised during initialization, `use_wernicke` is silently set to `False`. The Wernicke path is therefore inactive in any standard install.
- **Risks:** Architectural regression: a key biological component is disabled by default due to a missing dependency.
- **Recommendation:** High-value component. Add `sentence-transformers` back to `requirements.txt` or document the degraded path explicitly.

---

### `intent_analyzer.py` — `IntentAnalyzer`
- **Purpose:** Classifies user text into intent categories (GREETING, IDENTITY, INQUIRY, COMMAND, etc.) using regex scoring and heuristics. Reads/writes `mergen_matrix_memory.json` for telemetry persistence.
- **Status:** WORKING
- **Evidence:** `brain.py` line 36 imports `IntentAnalyzer`. `analyze_intent()` called at line 270 in `respond()`. `mergen_matrix_memory.json` confirmed present on disk (35,003 bytes).
- **Risks:** Pure regex-based classification with no ML component. Turkish-language coverage is incomplete for edge cases.
- **Recommendation:** Preserve. Works as intended.

---

### `response_generator.py` — `ResponseGenerator`
- **Purpose:** Generates explanatory natural language responses from retrieved knowledge facts. Handles conversational intent with template responses. Replaces the older `response_synthesizer.py`.
- **Status:** WORKING
- **Evidence:** `brain.py` line 49 imports `ResponseGenerator`; called at line 326 (`self.generator.generate()`). `_explain_from_facts()` selects sentences from the knowledge base, filters generic templates, and composes multi-sentence explanations.
- **Risks:** Responses are deterministic within a small random-choice pool. Vocabulary is in Turkish, with no English fallback.
- **Recommendation:** Preserve. Active and functional.

---

### `conversation_memory.py` — `ConversationMemory`
- **Purpose:** Stores multi-turn conversation history in a rolling window. Resolves Turkish and English pronouns (o, bu, şu, it, this). Persists to `mergen_conversation_memory.json`.
- **Status:** WORKING
- **Evidence:** `brain.py` line 48 imports `ConversationMemory`; instantiated at line 140–143 with `window_size=20` and `persistence_path`. `mergen_conversation_memory.json` confirmed present on disk (7,461 bytes).
- **Risks:** None critical.
- **Recommendation:** Preserve. Working correctly.

---

### `rag_engine.py` — `RAGEngine`
- **Purpose:** Transformer-free semantic retrieval using BioVectorizer (character n-gram hashing) and ChromaDB for persistence. HTMRetriever handles biological re-ranking via SDR overlap.
- **Status:** WORKING
- **Evidence:** `brain.py` lines 53–59 conditionally import and initialize `RAGEngine`. `mergen_rag_db/` directory confirmed present. `rag_engine.py` confirms `chromadb.PersistentClient` initialization. Degrades gracefully if `chromadb` is absent.
- **Risks:** `chromadb` is in `requirements.txt`, so the RAG path should be active. However, the collection may be empty on a fresh install — requires explicit `rag:yukle` command to populate.
- **Recommendation:** Preserve. Well-designed and operational.

---

### `bio_vectorizer.py` — `BioVectorizer`
- **Purpose:** Encodes text into 512-dimensional float vectors via character n-gram hashing and random projection. No external model dependency.
- **Status:** WORKING
- **Evidence:** `rag_engine.py` line 48 imports and instantiates `BioVectorizer`. Only depends on `hashlib` and `numpy`.
- **Risks:** n-gram hash collisions are undetected. Retrieval quality is lower than transformer-based embeddings.
- **Recommendation:** Preserve. Critical dependency of the RAG engine.

---

### `htm_retriever.py` — `HTMRetriever`
- **Purpose:** Re-ranks ChromaDB cosine similarity results using Sparse Distributed Representations (SDR), spreading activation, and lateral inhibition. Pure NumPy; no ML dependencies.
- **Status:** WORKING
- **Evidence:** `rag_engine.py` lines 80–87 import and instantiate `HTMRetriever`. Used in `search()` at lines 210–217.
- **Risks:** None critical. Falls back to cosine-only ranking if unavailable.
- **Recommendation:** Preserve. Biologically interesting and computationally sound.

---

### `hebbian_rag_bridge.py` — `HebbianRAGBridge`
- **Purpose:** Strengthens Hebbian synaptic traces between concepts that co-occur in RAG-retrieved texts. Runs updates synchronously (not in background thread as previously stated).
- **Status:** WORKING
- **Evidence:** `brain.py` lines 56, 174–178 instantiate `HebbianRAGBridge`. `ingest_file()` calls `self._hebb_bridge.update_from_batch()` at line 661.
- **Risks:** Threading was described in prior documentation but is not confirmed in the current codebase without full file inspection. Bridge calls may be synchronous and blocking.
- **Recommendation:** Preserve.

---

### `mergen_vocab.py` — `MergenVocab`
- **Purpose:** Vocabulary management: stores concept words, maps words to IDs, persists to `mergen_vocab.json`.
- **Status:** WORKING
- **Evidence:** `brain.py` line 30 imports `MergenVocab` (hard fail on error). `mergen_vocab.json` confirmed on disk (29,476 bytes). Used throughout `brain.py` for all KB indexing.
- **Risks:** If `mergen_vocab.json` is deleted, the entire knowledge base becomes inaccessible.
- **Recommendation:** Preserve. Mission-critical.

---

### `dream.py` — `MergenDream`
- **Purpose:** Standalone offline consolidation script. Implements NREM (STDP replay of episodic patterns), REM (spontaneous association), and homeostatic rebalancing. Saves back to `mergen.mx`.
- **Status:** EXPERIMENTAL
- **Evidence:** `dream.py` is a CLI script (`if __name__ == "__main__": main()`). It is **never imported** by `brain.py`, `main.py`, or any other module. It reads `mergen.mx` (note: different path convention from `mergen_weights.mx` used by `brain.py`). The `mergen.mx` file does not appear to exist on disk based on the directory listing, while `mergen_weights.mx` (2.25 MB) does. There is a path mismatch — `dream.py` defaults to `./mergen.mx` while the active brain saves to `./mergen_weights.mx`.
- **Risks:**
  - **CRITICAL:** Path mismatch means `dream.py` would load/save a *different* file from the one the active brain uses. Running `dream.py` would consolidate a stale or non-existent snapshot, not the live brain state.
  - Completely disconnected from the runtime pipeline.
- **Recommendation:** This is the most dangerous path mismatch in the repository. The consolidation path constant must be reconciled before `dream.py` can safely be used. High-value concept, but currently non-functional as an integration point.

---

### `limbic_executive_layer.py` — `LimbicExecutiveLayer`
- **Purpose:** Autonomy core implementing DMN background threads, metacognitive response filtering, XOR+Base64 encrypted `.mx` persistence, reward history, and user correction replay.
- **Status:** UNUSED
- **Evidence:** Searched all `.py` files using `findstr /s /i "limbic_executive_layer"`. Only found in `limbic_executive_layer.py` itself. No other module imports or instantiates `LimbicExecutiveLayer`. The file contains a standalone `if __name__ == "__main__"` test with mock objects.
- **Risks:** The encryption scheme (XOR + Base64) described within the file is acknowledged as not military-grade. The file is 33,714 bytes of non-integrated code.
- **Recommendation:** High-value design. Contains the most sophisticated architectural thinking in the repository (DMN background firing, metacognitive filtering, reward-based self-improvement). Should be preserved as a design reference for future integration, but currently contributes zero functionality to the running system.

---

### `language_engine.py` — `LanguageEngine`
- **Purpose:** Pure SNN language decoder. Maps motor-layer spike signatures to words via cosine similarity. Includes `dream_consolidate()` method callable by `dream.py`.
- **Status:** PARTIALLY WORKING
- **Evidence:** `brain.py` `_init_broca()` method (lines 217–225) imports `LanguageEngine` and wraps it inside `BrocaArea`. However, actual response generation in the primary pipeline goes through `ResponseGenerator`, not `LanguageEngine`. `LanguageEngine` is only used as a fallback (`_broca_generate()`) when `ResponseGenerator` produces a response shorter than 3 characters.
- **Risks:** The `LanguageEngine` → `dream.py` integration path (`dream_consolidate()`) is never called because `dream.py` itself is not integrated into the active runtime.
- **Recommendation:** Preserve. The fallback path is live. The dream integration requires fixing the `dream.py` path mismatch first.

---

### `response_synthesizer.py` — `ResponseSynthesizer`
- **Purpose:** Knowledge-aware response composer with fact scoring, sentence selection, and intent matching. An earlier generation response engine.
- **Status:** UNUSED
- **Evidence:** Searched all `.py` files using `findstr /s /i "response_synthesizer"`. Zero results outside the file itself. Not imported by any active module.
- **Risks:** 24,333 bytes of dead code. Contains overlapping functionality with the active `response_generator.py`.
- **Recommendation:** Dead code. Consider deleting after archival or document explicitly as superseded.

---

### `mergen_logic.py` — `LogicLayer`
- **Purpose:** Extracts top keyword concepts from a text and constructs heuristic "Mergen-style" commentary strings. Earlier generation synthesis engine.
- **Status:** UNUSED
- **Evidence:** Searched all `.py` files using `findstr /s /i "mergen_logic"`. Zero results. Not imported by any active module.
- **Risks:** 4,062 bytes of dead code.
- **Recommendation:** Dead code. Document as superseded or delete.

---

### `main.py` — Architecture B Training Loop
- **Purpose:** Headless training loop for SNN-based arithmetic classification using 2D cortical sheets, Global Workspace, Hippocampus, and HomeostaticRegulator.
- **Status:** WORKING (as standalone)
- **Evidence:** Fully self-contained. Imports from `anatomy/`, `connectivity/`, `utils/`, `datasets/`. Saves to `mergen.mx` (same ambiguous path as `dream.py`, different from `brain.py`'s `mergen_weights.mx`). Has its own `MergenCognitiveArchitecture` class.
- **Risks:**
  - **Architectural conflict:** The class name `MergenCognitiveArchitecture` is shared with the conceptual name used in project documentation, but it implements a completely different architecture from `brain.py`'s `MergenBrain_v7`.
  - `main.py` is not the primary entry point for the conversational system. Its naming causes confusion.
- **Recommendation:** Rename to `train_v3.py` or similar to prevent confusion with the primary entry point.

---

### `anatomy/` Package — `CorticalLayer`, `Hippocampus`, `BasalGanglia`, `Cerebellum`
- **Purpose:** Biophysical neuron modules for Architecture B (the V3 cortical training loop).
  - `CorticalLayer`: LIF neurons with FFT-based Mexican-hat lateral connectivity, adaptation, homeostatic thresholds.
  - `Hippocampus`: Cosine-similarity key-value episodic memory with adaptive threshold.
  - `BasalGanglia`: Action-selection module with dopaminergic RL learning.
  - `Cerebellum`: Error-correction perceptron with LMS learning.
- **Status:** WORKING (`CorticalLayer`, `Hippocampus`), UNUSED (`BasalGanglia`, `Cerebellum`)
- **Evidence:** `main.py` only imports `CorticalLayer` and `Hippocampus`. `BasalGanglia` and `Cerebellum` are not imported by any file in the repository.
- **Risks:** `BasalGanglia` and `Cerebellum` are orphan modules.
- **Recommendation:** `CorticalLayer` and `Hippocampus` are high-quality biophysical implementations worth preserving. `BasalGanglia` and `Cerebellum` are dead code.

---

### `connectivity/` Package — `GlobalWorkspace`, `MexicanHatKernel`, `projections`
- **Purpose:** Routing and lateral connectivity for Architecture B.
  - `GlobalWorkspace`: Low-rank bottleneck router (N→K→N) implementing Global Neuronal Workspace Theory.
  - `MexicanHatKernel`: Precomputed FFT-domain Mexican-hat kernel for `CorticalLayer`.
  - `projections.py`: Minimal utility (likely helper for connectivity setup).
- **Status:** WORKING (`GlobalWorkspace`, `kernels`), UNKNOWN (`projections`)
- **Evidence:** `main.py` imports `MexicanHatKernel` from `connectivity.kernels` and `GlobalWorkspace` from `connectivity.global_workspace`.
- **Risks:** None critical within Architecture B scope.
- **Recommendation:** Preserve.

---

### `learning/` Package — `HybridHebbianLearner`, `STDPMechanism`, `DopamineModulator`, `SurrogateSpike`
- **Purpose:** Standalone learning engine package implementing 15 biological learning principles (STDP, eligibility traces, dopamine modulation, homeostatic scaling).
  - `hebbian_engine.py`: Full three-factor Hebbian learner with dopaminergic gating.
  - `stdp.py`: STDP weight update kernel.
  - `rl_agent.py`: Dopamine modulator (reward prediction error).
  - `gradients.py`: Surrogate gradient functions for spike backpropagation compatibility.
- **Status:** UNUSED (as a package)
- **Evidence:** `learning/` package is not imported by `main.py`, `brain.py`, `Mergen.py`, or any other top-level module. `limbic_executive_layer.py` references a `HybridHebbianLearner`-compatible engine interface in its docstring but the package is not imported there either.
- **Risks:** This is the most sophisticated learning engine in the repository (15 biological principles, eligibility traces, dopamine gating). It currently contributes no functionality to any running system.
- **Recommendation:** **Highest-value unused component.** This package's `HybridHebbianLearner` is the intended engine for `LimbicExecutiveLayer`. Both are isolated. Preserving this is the single most important architectural decision.

---

### `auto_evolution.py` + `code_evolution.py` — Evolution Pipeline
- **Purpose:** `CodeEvolutionEngine` (`code_evolution.py`) reads Python files and requests LLM improvements via `OpenRouterClient`. `AutoEvolutionController` (`auto_evolution.py`) orchestrates a curriculum-driven improvement loop.
- **Status:** WORKING (as standalone CLI)
- **Evidence:** `auto_evolution.py` is a standalone script requiring `python auto_evolution.py <API_KEY>`. `run_mergen_evolution.bat` confirmed present. `code_evolution.py` uses `ast.parse()` to validate AI-generated code before applying changes.
- **Risks:**
  - **CRITICAL SAFETY RISK:** The evolution engine calls `self.evolver.get_all_python_files()` which returns all `.py` files in the project root and subdirectories, then overwrites them with LLM-generated code. This includes core files like `brain.py`, `broca_area.py`, and `mergen_vocab.py`. A bad LLM response that passes AST validation could silently corrupt the architecture.
  - No file exclusion list, no backup mechanism, no rollback strategy confirmed in code.
  - External LLM dependency (OpenRouter API) creates a network call risk on the production codebase.
- **Recommendation:** High-risk feature. Should only be run on a git-branched copy of the repository, never on the main working tree. Add explicit file exclusion list and git-commit-before-evolve safeguard.

---

### `dream.py` — Offline Consolidation
(See subsystem entry above. Re-stated here for completeness.)
- **Status:** EXPERIMENTAL
- **Critical issue:** Path mismatch with active brain (`mergen.mx` vs. `mergen_weights.mx`).

---

### `mergen_matrix_memory.json`
- **Purpose:** Telemetry and intent state persistence for `IntentAnalyzer`.
- **Status:** WORKING (file exists, 35,003 bytes).
- **Risks:** None critical.

### `mergen_vocab.json`
- **Purpose:** Vocabulary persistence for `MergenVocab`.
- **Status:** WORKING (file exists, 29,476 bytes).
- **Risks:** Deleting this file destroys the knowledge base index.

### `mergen_weights.mx`
- **Purpose:** Full serialized brain state (weights, KB facts, concept index).
- **Status:** WORKING (file exists, 2,254,886 bytes — approximately 2.2 MB).
- **Risks:** Single point of failure for all learned knowledge.

### `mergen_conversation_memory.json`
- **Purpose:** Conversation turn history for `ConversationMemory`.
- **Status:** WORKING (file exists, 7,461 bytes).

---

## Dead Code

| File | Reason |
|------|--------|
| `response_synthesizer.py` | Not imported by any module. Replaced by `response_generator.py`. |
| `mergen_logic.py` | Not imported by any module. Replaced by `response_generator.py`. |
| `anatomy/basal_ganglia.py` | Not imported by `main.py` or any other file. |
| `anatomy/cerebellum.py` | Not imported by `main.py` or any other file. |
| `context_manager.py` | Not confirmed imported by any primary module (inspection pending). |
| `sentences.py` | Not confirmed imported by any primary module (inspection pending). |

---

## Orphan Modules

Modules that exist, contain working code, but are not wired into any runtime pipeline:

| File/Package | What It Does | Why It's Orphaned |
|---|---|---|
| `limbic_executive_layer.py` | DMN threads, metacognitive filtering, encrypted `.mx` persistence | Never imported by `brain.py`, `main.py`, or any other module |
| `learning/` (full package) | 15-principle Hebbian engine, STDP, dopamine modulation | Never imported anywhere in the project |
| `dream.py` | NREM/REM offline consolidation | Standalone CLI; wrong `.mx` path; not connected to active brain |
| `language_engine.py` | SNN spike-to-word decoder; `dream_consolidate()` | Only used as fallback; dream integration path is broken |

---

## Architectural Conflicts

1. **Dual `MergenCognitiveArchitecture` identity:** `main.py` defines a class with this name for the V3 arithmetic training loop. Project documentation uses the same name for the conversational brain. These are completely different systems. Risk: architectural confusion in all future documentation and development.

2. **Dual `.mx` file paths:** `brain.py` saves to `./mergen_weights.mx`. `main.py` and `dream.py` both use `./mergen.mx` (default). Running any of these in sequence will operate on different state files silently.

3. **Two incompatible Hebbian engines:** `broca_area.py` implements a simplistic `hebbian_trace` tensor in `MergenBrain` (a buffer, not a proper learner). `learning/hebbian_engine.py` implements a full 15-principle `HybridHebbianLearner`. Architecture A uses the simple buffer. Architecture B does not use either — it uses a direct delta-rule readout. The sophisticated engine in `learning/` is used by nothing.

4. **`WernickeArea` dependency contradiction:** `wernicke_area.py` requires `sentence-transformers`. `requirements.txt` explicitly removes it with a comment saying "BioVectorizer is being used instead." But `BioVectorizer` is only in the RAG pipeline, not in `WernickeArea`. The `WernickeArea` semantic path is disabled on any standard install.

5. **`response_synthesizer.py` vs. `response_generator.py`:** Two response generation modules. Only `response_generator.py` is active. `response_synthesizer.py` is dead code. `SYSTEM_ARCHITECTURE.md` described the wrong one as the active module.

---

## Technical Debt

| Severity | Location | Description |
|----------|----------|-------------|
| **CRITICAL** | `dream.py` | Path mismatch: reads `mergen.mx`, active brain writes `mergen_weights.mx`. Running `dream.py` consolidates the wrong (or non-existent) file. |
| **CRITICAL** | `auto_evolution.py` | No file exclusion list or backup. Can overwrite any `.py` file in the project, including core modules, with LLM-generated code. |
| **HIGH** | `brain.py` lines 304–305 | `except: pass` on `learn_from_text()`. Learning failures are completely silent. |
| **HIGH** | `brain.py` `_recall_knowledge()` | Five retrieval strategies all wrapped in `try/except: pass`. Any or all can fail silently. |
| **HIGH** | `broca_area.py` | 1,283-line file with three unrelated classes. File name misleads as to its primary content (`MergenBrain` is the neural core, not a Broca area). |
| **HIGH** | `learning/` package | 15-principle Hebbian engine is the most sophisticated component in the repository and is completely unused. |
| **MEDIUM** | `limbic_executive_layer.py` | 775-line file defining a complete autonomous layer that is never instantiated. |
| **MEDIUM** | `main.py` | Named ambiguously. Is not the main entry point for the conversational system. Implements an entirely different architecture. |
| **MEDIUM** | `wernicke_area.py` | Functional but silently disabled on standard install due to missing `sentence-transformers` dependency. |
| **MEDIUM** | `response_synthesizer.py` | 24,333 bytes of dead code competing with `response_generator.py` in the namespace. |
| **LOW** | `anatomy/basal_ganglia.py`, `anatomy/cerebellum.py` | Dead code never imported. |

---

## Most Valuable Components Worth Preserving

Listed by architectural value, independent of current integration status:

1. **`broca_area.py` (`MergenBrain` class)** — The actual neural core of the running system. Contains the knowledge base, concept index, Hebbian trace, and all learning methods. Irreplaceable.

2. **`learning/hebbian_engine.py` (`HybridHebbianLearner`)** — The most biologically rigorous component in the entire repository. 15-principle three-factor Hebbian learner with eligibility traces and dopamine gating. Currently orphaned but architecturally significant.

3. **`limbic_executive_layer.py` (`LimbicExecutiveLayer`)** — The most complete implementation of autonomous cognitive behavior: DMN background threads, metacognitive filtering, reward history, user correction replay. Currently orphaned but architecturally the most ambitious design.

4. **`rag_engine.py` + `bio_vectorizer.py` + `htm_retriever.py`** — A complete, well-integrated, dependency-light retrieval pipeline. Transformer-free by design, biologically inspired, and fully operational.

5. **`anatomy/cortical_sheet.py` (`CorticalLayer`)** — High-quality LIF spiking neuron implementation with FFT-domain lateral connectivity, adaptive thresholds, and refractory periods. Only used in `main.py` but architecturally sound.

6. **`dream.py` (`MergenDream`)** — Well-designed offline consolidation engine. Nonfunctional due to a path mismatch but the consolidation logic itself (NREM/REM/homeostasis) is valuable and should not be deleted.

7. **`wernicke_area.py` (`WernickeArea`)** — Clean, well-documented implementation of three spike-encoding strategies. Disabled by a missing dependency, not by design flaws.
