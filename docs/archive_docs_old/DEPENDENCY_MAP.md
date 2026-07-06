# DEPENDENCY MAP

**Date:** 2026-06-19
**Auditor:** Technical Auditor — code evidence only
**Method:** Direct import-level inspection of all source files
**Scope:** Runtime import dependencies. Not call-graph dependencies.

---

## Legend

```
──►  Hard import  (fails to start if missing; ImportError exits)
- ->  Soft import  (try/except; system degrades but does not crash)
···►  Internal package import (within same package)
[D]   Disk artifact dependency (file on filesystem, not import)
```

Status tags used inline:
- `[ACTIVE]` — imported and called in the runtime pipeline
- `[OPTIONAL]` — imported with try/except; system runs without it
- `[ORPHAN]` — defined but never imported by any other module
- `[DEAD]` — imported by an orphan, transitively unreachable at runtime

---

## Section 1 — Architecture A Dependency Graph

**Entry point:** `Mergen.py` → `brain.py` → all downstream

```
Mergen.py
    ──► brain.py (MergenBrain_v7)
            ──► mergen_vocab.py       [ACTIVE]  hard — exits on failure
            ──► intent_analyzer.py   [ACTIVE]  hard — exits on failure
            ──► broca_area.py        [ACTIVE]  hard — exits on failure
                    ··· MergenConfig           (internal class)
                    ··· MergenBrain            (internal class)
                    ··· BrocaArea              (internal class)
                            ──► language_engine.py  [ACTIVE — fallback only]
            ──► conversation_memory.py    [ACTIVE]  unconditional import
            ──► response_generator.py     [ACTIVE]  unconditional import
            ──► mergen_brain_wrapper.py   [ACTIVE]  unconditional import
                    ──► broca_area.py     [ACTIVE]  (already loaded)
                    - -> wernicke_area.py [OPTIONAL] try/except at init time
                            - -> sentence_transformers  [OPTIONAL — MISSING]
            - -> rag_engine.py       [OPTIONAL]
                    ──► bio_vectorizer.py   [ACTIVE if RAG loads]
                    - -> htm_retriever.py   [OPTIONAL within RAG]
                    ──► chromadb           [ACTIVE if RAG loads — external]
            - -> data_loader.py      [OPTIONAL]  (part of RAG group)
            - -> hebbian_rag_bridge.py [OPTIONAL] (part of RAG group)
            - -> turkish_morph.py    [OPTIONAL]
                    - -> zeyrek        [OPTIONAL — external]
```

**External packages consumed by Architecture A:**

| Package | Required by | Install status |
|---------|-------------|----------------|
| `torch` | `broca_area.py`, `mergen_brain_wrapper.py`, `language_engine.py` | In `requirements.txt` |
| `chromadb` | `rag_engine.py` | In `requirements.txt` |
| `numpy` | `bio_vectorizer.py`, `htm_retriever.py` | In `requirements.txt` |
| `zeyrek` | `turkish_morph.py` | In `requirements.txt` |
| `requests` | `data_loader.py` | In `requirements.txt` |
| `sentence_transformers` | `wernicke_area.py` | **ABSENT from `requirements.txt`** |
| `matplotlib` | `dream.py` only (standalone) | In `requirements.txt` |

---

## Section 2 — Architecture B Dependency Graph

**Entry point:** `main.py`

```
main.py
    ──► config.py                              [ACTIVE]
    ──► anatomy.cortical_sheet (CorticalLayer) [ACTIVE]
            ──► engine.tensor_ops (fft_convolve2d)  [ACTIVE]
                    ──► torch                        [ACTIVE]
            ──► config                              [ACTIVE]
    ──► anatomy.hippocampus (Hippocampus)      [ACTIVE]
            ──► torch                               [ACTIVE]
    ──► connectivity.kernels (MexicanHatKernel) [ACTIVE]
            ──► torch                               [ACTIVE]
    ──► connectivity.global_workspace (GlobalWorkspace) [ACTIVE]
            ──► torch                               [ACTIVE]
    ──► utils.encoder (SpikeEncoder)           [ACTIVE]
            ──► torch                               [ACTIVE]
            ──► numpy                              [ACTIVE]
    ──► utils.stability (HomeostaticRegulator)  [ACTIVE]
    ──► utils.telemetry (TelemetryBox)          [ACTIVE]
    ──► datasets.generators.math_teacher (MathTeacher) [ACTIVE]
```

**Package `__init__.py` re-exports (loaded when package is imported):**

```
anatomy/__init__.py  exports:
    ··► CorticalLayer   [ACTIVE — used by main.py]
    ··► Hippocampus     [ACTIVE — used by main.py]
    ··► BasalGanglia    [ORPHAN — exported but never consumed]
    ··► Cerebellum      [ORPHAN — exported but never consumed]

connectivity/__init__.py  exports:
    ··► MexicanHatKernel              [ACTIVE]
    ··► GaborKernel                   [ORPHAN — exported but never consumed]
    ··► create_sparse_projection      [ORPHAN]
    ··► create_topological_projection [ORPHAN]
    ··► GlobalWorkspace               [ACTIVE]

engine/__init__.py  exports:
    ··► fft_convolve2d     [ACTIVE — used by CorticalLayer]
    ··► normalize_tensor   [ORPHAN — exported, not used by main.py directly]
    ··► EulerSolver        [ORPHAN]
    ··► RungeKutta4Solver  [ORPHAN]
    ··► DelayBuffer        [ORPHAN]

learning/__init__.py  exports:
    ··► SurrogateSpike     [ORPHAN — package never imported by main.py]
    ··► STDPMechanism      [ORPHAN]
    ··► DopamineModulator  [ORPHAN]
```

**External packages consumed by Architecture B:**

| Package | Required by | Install status |
|---------|-------------|----------------|
| `torch` | All anatomy/connectivity/engine/utils | In `requirements.txt` |
| `numpy` | `utils/encoder.py` | In `requirements.txt` |

Architecture B has the smallest external dependency surface of all three architectures.

---

## Section 3 — Architecture C Dependency Graph

**Entry point:** `auto_evolution.py <API_KEY>` (or via `run_mergen_evolution.bat`)

```
auto_evolution.py
    ──► openrouter_client.py (OpenRouterClient, FREE_MODELS)  [ACTIVE]
            ──► requests                                       [ACTIVE — external]
    ──► code_evolution.py (CodeEvolutionEngine)               [ACTIVE]
            ──► ast                                           [stdlib]
            ──► json, re, sys, pathlib                        [stdlib]
    ──► curriculum.py (CURRICULUM dict)                       [ACTIVE]
    - -> telegram_bot.py (TelegramBot)  [OPTIONAL — if token arg provided]
```

**Hidden filesystem dependency:** `code_evolution.py` calls `self.project_root.rglob("*.py")` at runtime, which reads all Python files in the working directory. This is not an import dependency but a **hidden runtime dependency on the entire source tree**. Any `.py` file present at runtime becomes a potential target for overwriting.

**External packages consumed by Architecture C:**

| Package | Required by | Install status |
|---------|-------------|----------------|
| `requests` | `openrouter_client.py` | In `requirements.txt` |

Architecture C makes no PyTorch or scientific computing imports. It is independent of the cognitive infrastructure.

---

## Section 4 — Orphaned Dependency Graphs

These subsystems import from each other or from active modules, but are never imported by any entry point (`Mergen.py`, `main.py`, `auto_evolution.py`).

### 4A — `limbic_executive_layer.py`

```
limbic_executive_layer.py  [ORPHAN — never imported]
    ──► torch               [already in env]
    ──► json, threading, time, pathlib, base64, hashlib  [stdlib]
    [no imports of other project modules]
```

`limbic_executive_layer.py` has zero inbound imports from the rest of the project. It imports only stdlib and PyTorch. It is fully self-contained and fully disconnected.

### 4B — `dream.py`

```
dream.py  [ORPHAN — never imported; standalone CLI]
    ──► torch, numpy                              [active packages]
    ──► os, json, time, pathlib, datetime         [stdlib]
    ──► language_engine.py (LanguageEngine)       [ACTIVE in Arch A — reachable]
    [D] ./mergen.mx                               [DISK — PATH MISMATCH]
    [D] ./logs/*.npz                              [DISK]
    [D] ./dream_log.npz                           [DISK]
```

`dream.py` calls `language_engine.LanguageEngine.dream_consolidate()`, which is defined and active. However, `dream.py` itself is never called from any runtime pipeline. This is the only cross-architecture import connection in the repository: a standalone CLI script calling a method on a module that is loaded by Architecture A.

**Path mismatch:** `dream.py` defaults to `./mergen.mx`. `brain.py` (Architecture A) saves to `./mergen_weights.mx`. These are different files.

### 4C — `learning/` package

```
learning/__init__.py  [ORPHAN — package never imported]
    ··► learning/gradients.py  (SurrogateSpike, SpikingActivation)
    ··► learning/stdp.py       (STDPMechanism)
    ··► learning/rl_agent.py   (DopamineModulator)

learning/hebbian_engine.py  [ORPHAN — not exported from __init__]
    ──► torch
    ──► learning.gradients     (SurrogateSpike, SpikingActivation)
    ──► learning.stdp          (STDPMechanism)
    ──► learning.rl_agent      (DopamineModulator)
```

`HybridHebbianLearner` is defined in `learning/hebbian_engine.py` but is **not exported** from `learning/__init__.py`. It is therefore unreachable even if someone imports the `learning` package. This is a silent omission — the most sophisticated learning engine in the repository is not only orphaned from the runtime but also omitted from its own package's public interface.

### 4D — `response_synthesizer.py`

```
response_synthesizer.py  [ORPHAN — never imported]
    ──► re, random, collections   [stdlib]
    [no project module imports]
```

### 4E — `mergen_logic.py`

```
mergen_logic.py  [ORPHAN — never imported]
    ──► re, random, collections   [stdlib]
    [no project module imports]
```

### 4F — `monitor_server.py`

```
monitor_server.py  [ORPHAN — standalone; never imported]
    ──► json, http.server, pathlib   [stdlib]
    [D] ./index.html                 [DISK — not confirmed present]
    [D] ./status.json                [DISK — written by auto_evolution.py]
```

`monitor_server.py` reads `status.json`, which is written by `auto_evolution.py`. This is the only runtime coupling between Architecture C and any other component — and it is a filesystem coupling, not an import dependency.

### 4G — `sentences.py`

```
sentences.py  [ORPHAN — never imported]
    ──► re, random   [stdlib]
    [no project module imports]
```

`TurkishSentenceBuilder` is implemented and exports grammar composition methods. No other module imports it.

### 4H — `context_manager.py`

```
context_manager.py  [ORPHAN — never imported]
    ──► re, collections   [stdlib]
    [no project module imports]
```

`ContextManager` stores and summarizes text context. No other module imports it. This is distinct from `ConversationMemory` (`conversation_memory.py`), which is active in Architecture A.

---

## Section 5 — Circular Dependency Analysis

**No circular imports were found.**

Evidence: The dependency graph across all files is a directed acyclic graph (DAG). The key absence of cycles is guaranteed by the following structural properties:

- `brain.py` imports `broca_area.py`. `broca_area.py` imports `language_engine.py`. `language_engine.py` imports no project modules.
- `mergen_brain_wrapper.py` imports `broca_area.py` and optionally `wernicke_area.py`. Neither imports back to `mergen_brain_wrapper.py`.
- The `anatomy/`, `connectivity/`, `engine/`, `utils/`, `learning/`, `datasets/` packages import only from `torch`, `numpy`, `stdlib`, or sibling modules within the same package. None import upward to `main.py` or `brain.py`.
- `hebbian_rag_bridge.py` imports no project modules (accepts `brain` and `vocab` as constructor arguments, not import-time dependencies).

---

## Section 6 — Hidden Dependencies

Hidden dependencies are runtime requirements not expressed as Python imports.

### 6A — Disk File Dependencies

| File required at runtime | Consumer | Consequence if absent |
|---|---|---|
| `mergen_weights.mx` | `brain.py` (line 121) | Brain starts with empty weights and empty KB |
| `mergen_vocab.json` | `mergen_vocab.py` | Vocab starts from hardcoded seed words only |
| `mergen_matrix_memory.json` | `intent_analyzer.py` | IntentAnalyzer starts with zero telemetry history |
| `mergen_conversation_memory.json` | `conversation_memory.py` | Conversation history starts empty |
| `mergen_rag_db/` (ChromaDB) | `rag_engine.py` | RAG returns empty results; `count()` returns 0 |
| `mergen.mx` | `dream.py` | Dream loads empty weights (path mismatch with active brain) |
| `logs/*.npz` | `dream.py` | NREM phase has no episodic patterns to replay |
| `learning_progress.json` | `code_evolution.py` | Evolution starts from grade 1 |
| `status.json` | `monitor_server.py` | Monitor returns default empty status |
| `index.html` | `monitor_server.py` | Monitor returns 404 on root request |
| `config.py` | `main.py`, `anatomy/cortical_sheet.py` | Hard failure — not in try/except |

### 6B — Network Dependencies

| Network target | Consumer | Condition |
|---|---|---|
| `api.github.com` (raw) | `data_loader.py` | When `rag:yukle` command is invoked |
| `openrouter.ai/api/v1/chat/completions` | `openrouter_client.py` | When `auto_evolution.py` is running |
| HuggingFace model hub | `wernicke_area.py` (first load) | If `sentence-transformers` installed and Wernicke enabled |

### 6C — Implicit Package Version Dependencies

`brain.py` calls `chromadb.PersistentClient(path=...)`. The `PersistentClient` API was introduced in ChromaDB `0.4.0`. `requirements.txt` specifies `chromadb>=0.4.0` correctly. However, `chromadb` API is not stable across minor versions — a `chromadb>=0.5.0` install may behave differently for collection creation and `upsert` batch handling.

---

## Section 7 — Unused Imports

### 7A — Imports within `anatomy/__init__.py` that are never consumed

```python
# anatomy/__init__.py
from .basal_ganglia import BasalGanglia   # Never used by main.py or any caller
from .cerebellum import Cerebellum        # Never used by main.py or any caller
```

`main.py` imports `from anatomy.cortical_sheet import CorticalLayer` and `from anatomy.hippocampus import Hippocampus` directly — bypassing `anatomy/__init__.py` entirely. This means `BasalGanglia` and `Cerebellum` are loaded into memory when any code does `import anatomy`, but they are never referenced.

### 7B — Imports within `connectivity/__init__.py` that are never consumed

```python
# connectivity/__init__.py
from .kernels import GaborKernel                       # Never used anywhere
from .projections import create_sparse_projection      # Never used anywhere
from .projections import create_topological_projection # Never used anywhere
```

`main.py` imports `from connectivity.kernels import MexicanHatKernel` and `from connectivity.global_workspace import GlobalWorkspace` directly. `GaborKernel` and both `projections` utilities are never called.

### 7C — `HybridHebbianLearner` not exported from `learning/__init__.py`

```python
# learning/__init__.py — current state
from .gradients import SurrogateSpike
from .stdp import STDPMechanism
from .rl_agent import DopamineModulator
# HybridHebbianLearner from hebbian_engine.py is NOT exported
```

`HybridHebbianLearner` is the primary class of the `learning/` package by file size (265 lines, ~44% of the package). It is the only class that composes the other three (`SurrogateSpike`, `STDPMechanism`, `DopamineModulator`) into a complete learning engine. It is absent from the package's public interface.

### 7D — Unused symbol: `normalize_tensor` in `engine/__init__.py`

```python
from .tensor_ops import fft_convolve2d, normalize_tensor
```

`normalize_tensor` is exported but not called by `main.py` or any active code path. `fft_convolve2d` is called by `anatomy/cortical_sheet.py`.

---

## Section 8 — Disconnected Subsystems

The following subsystems form independent clusters with no import edges connecting them to any active entry point:

### Cluster 1 — Autonomy / Metacognition

```
limbic_executive_layer.py ──► [stdlib + torch only]
```
No inbound edges. No outbound edges to project modules.

### Cluster 2 — Offline Consolidation

```
dream.py ──► language_engine.py  (inbound from Arch A pipeline — read-only)
         ──► [torch, numpy, stdlib]
         [D] mergen.mx (MISMATCHED PATH)
```
No inbound edges. One outbound call to `language_engine.py.dream_consolidate()` which is defined but never triggered at runtime.

### Cluster 3 — Advanced Learning Engine

```
learning/
    hebbian_engine.py ──► learning/gradients.py
                      ──► learning/stdp.py
                      ──► learning/rl_agent.py
```
The `learning/` package is never imported by `main.py`, `brain.py`, or any other module. Internally self-consistent; externally fully disconnected.

### Cluster 4 — Standalone Language Utilities

```
sentences.py       ──► [stdlib only]
context_manager.py ──► [stdlib only]
mergen_logic.py    ──► [stdlib only]
```
No inbound or outbound edges to project modules.

### Cluster 5 — Evolution Monitor

```
monitor_server.py ──► [stdlib only]
                  [D] status.json (written by auto_evolution.py)
                  [D] index.html
```
Connected to Architecture C only via a filesystem artifact (`status.json`). No import-level connection.

---

## Section 9 — Full Dependency Matrix

`I` = imports, `O` = optionally imports, `-` = no dependency

| Consumer → | `mergen_vocab` | `broca_area` | `intent_analyzer` | `conv_memory` | `response_gen` | `brain_wrapper` | `wernicke_area` | `rag_engine` | `bio_vectorizer` | `htm_retriever` | `hebbian_rag` | `lang_engine` | `turkish_morph` | `data_loader` |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `Mergen.py` | - | - | - | - | - | - | - | - | - | - | - | - | - | - |
| `brain.py` | I | I | I | I | I | I | - | O | - | - | O | - | O | O |
| `brain_wrapper` | - | I | - | - | - | - | O | - | - | - | - | - | - | - |
| `broca_area` | - | - | - | - | - | - | - | - | - | - | - | I | - | - |
| `rag_engine` | - | - | - | - | - | - | - | - | I | O | - | - | - | - |
| `hebbian_rag` | - | - | - | - | - | - | - | - | - | - | - | - | - | - |
| `dream.py` | - | - | - | - | - | - | - | - | - | - | - | I | - | - |

| Consumer → | `config` | `anatomy.*` | `connectivity.*` | `engine.*` | `utils.*` | `datasets.*` | `learning.*` |
|---|---|---|---|---|---|---|---|
| `main.py` | I | I (partial) | I (partial) | - | I | I | - |
| `cortical_sheet` | I | - | - | I | - | - | - |
| `anatomy/__init__` | - | I (all 4) | - | - | - | - | - |
| `connectivity/__init__` | - | - | I (all) | - | - | - | - |
| `engine/__init__` | - | - | - | I (all) | - | - | - |
| `hebbian_engine` | - | - | - | - | - | - | I (3/3) |

---

## Section 10 — Summary of Dependency Classification

### Active Runtime Dependencies (Architecture A)

```
Mergen.py → brain.py → mergen_vocab → [json, pathlib]
                     → intent_analyzer → [re, json, math]
                     → broca_area → language_engine → [re, torch]
                     → conversation_memory → [re, json, collections]
                     → response_generator → [re, random]
                     → mergen_brain_wrapper → broca_area (cached)
```

### Optional Runtime Dependencies (Architecture A)

```
brain.py -→ mergen_brain_wrapper -→ wernicke_area -→ sentence_transformers [MISSING]
         -→ rag_engine → bio_vectorizer → [hashlib, numpy]
                       → htm_retriever → [numpy]
                       → chromadb [external]
         -→ hebbian_rag_bridge → [re, threading]
         -→ turkish_morph → zeyrek [external, optional]
         -→ data_loader → requests [external]
```

### Active Runtime Dependencies (Architecture B)

```
main.py → config
        → anatomy.cortical_sheet → engine.tensor_ops → torch
        → anatomy.hippocampus → torch
        → connectivity.kernels → torch
        → connectivity.global_workspace → torch
        → utils.encoder → torch, numpy
        → utils.stability → torch
        → utils.telemetry → torch, numpy
        → datasets.generators.math_teacher → random
```

### Orphaned (No Inbound Imports from Any Entry Point)

| Module | Inbound import count | Status |
|--------|---------------------|--------|
| `limbic_executive_layer.py` | 0 | ORPHAN |
| `learning/hebbian_engine.py` | 0 | ORPHAN |
| `learning/__init__.py` | 0 | ORPHAN |
| `dream.py` | 0 | ORPHAN (standalone CLI) |
| `response_synthesizer.py` | 0 | DEAD CODE |
| `mergen_logic.py` | 0 | DEAD CODE |
| `sentences.py` | 0 | ORPHAN |
| `context_manager.py` | 0 | ORPHAN |
| `monitor_server.py` | 0 | ORPHAN (standalone CLI) |
| `anatomy/basal_ganglia.py` | 1 (from `anatomy/__init__`) | DEAD — re-exported but unconsumed |
| `anatomy/cerebellum.py` | 1 (from `anatomy/__init__`) | DEAD — re-exported but unconsumed |
| `connectivity/projections.py` | 1 (from `connectivity/__init__`) | DEAD — re-exported but unconsumed |

### Most Critical Missing Connection

`learning/hebbian_engine.py` (`HybridHebbianLearner`) has **zero inbound imports** and is also **absent from its own package's `__init__.py`**. It imports from three sibling modules (`gradients`, `stdp`, `rl_agent`), which are themselves only transitively orphaned because the package that contains them is never loaded. This is the deepest disconnection in the repository: a complete learning subsystem that is invisible both to the runtime and to the package system.
