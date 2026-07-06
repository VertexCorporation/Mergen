# MERGEN DEVELOPER ONBOARDING GUIDE

**Audience:** New developers joining the Mergen project.
**Goal:** Practical orientation to the codebase as it exists today, not as it was planned.
**Author:** Principal AI Systems Architect

---

## 1. What the System Actually Does

Mergen is officially described as a "biological cognitive AI." In reality, the repository contains three completely disconnected projects:

1.  **A Turkish Conversational Chatbot (Active Production):** Takes Turkish text, classifies intent via regex/keywords, looks up relevant sentences from a Python list or ChromaDB (RAG), and returns a string. It has two frozen PyTorch Linear layers that do no real learning.
2.  **A Headless SNN Math Engine (Experimental):** A biophysical spiking neural network loop that trains itself to classify simple addition problems (e.g., "3 + 5" → 8).
3.  **An Autonomous Code Rewriter (Standalone Tool):** A script that sends the project's own `.py` files to an LLM API and overwrites them locally.

When you run `python Mergen.py`, you are running only the **Chatbot**. It has no neural reasoning, no world model, and no biological learning. It is a keyword-indexer wrapped in a chatbot UI.

## 2. Active Runtime Path

The only path a user actually interacts with is the Chatbot (Architecture A). Here is what happens when you type a message:

1.  **Entry:** `Mergen.py` → `brain.py:MergenBrain_v7.run()`.
2.  **Intent:** `intent_analyzer.py` checks your text against hardcoded lists of Turkish words to guess your intent.
3.  **"Neural" Pass:** `broca_area.py:MergenBrain` passes your text through two PyTorch Linear layers that are essentially frozen, outputting a tensor that is largely ignored.
4.  **Learning:** If you typed a declarative sentence (not a question), it appends your exact string to a Python list called `knowledge_base` and saves it to disk (`mergen_weights.mx`).
5.  **Recall:** If you asked a question, it searches the `knowledge_base` list and ChromaDB (`rag_engine.py`) for matching substrings.
6.  **Response:** `response_generator.py` glues the retrieved strings together into a response.

## 3. Most Important Modules

If you want to understand the *production* system:
- `brain.py`: The massive 1000-line God Object that runs the chatbot loop.
- `broca_area.py`: Contains `MergenBrain`, the neural core (which is actually just data storage + 2 linear layers).
- `rag_engine.py`: The cleanest, most functional part of the production system (ChromaDB retrieval).

If you want to understand the *scientific/biological* future of the project:
- `main.py`: The training loop for the biophysical spiking neural network.
- `learning/hebbian_engine.py`: The most advanced code in the repo. A state-of-the-art biologically plausible learning engine. *(Currently completely disconnected from the rest of the project).*
- `limbic_executive_layer.py`: An autonomous metacognitive loop designed to let the AI think while idle. *(Currently completely disconnected).*

## 4. Modules That Can Be Ignored Initially

- `auto_evolution.py` / `code_evolution.py`: This is a dangerous code-rewriting script. Ignore it until you need to automate refactoring. Do not run it.
- `response_synthesizer.py`: Dead code. 24KB of unused logic.
- `mergen_logic.py`: Dead code.
- `anatomy/basal_ganglia.py` and `anatomy/cerebellum.py`: Written but never used anywhere.
- `monitor_server.py`: A standalone web server that just serves a static JSON file.

## 5. Learning Order for the Codebase

Do not read top-to-bottom. Read in this order:

**Phase 1: The Production Reality**
1.  `Mergen.py` (Entry point)
2.  `brain.py` (Specifically `respond()`, to see the actual pipeline)
3.  `broca_area.py` (Specifically `MergenBrain.learn_from_text()` to see that "learning" means appending to a list)

**Phase 2: The Biological Architecture (The Future)**
4.  `main.py` (To see how a spiking neural network is wired)
5.  `anatomy/cortical_sheet.py` (The actual biophysical neuron model)
6.  `connectivity/global_workspace.py` (How different brain regions talk)

**Phase 3: The Disconnected Masterpieces**
7.  `learning/hebbian_engine.py` (The true biological learning engine)
8.  `limbic_executive_layer.py` (The autonomy/daydreaming engine)
9.  `dream.py` (Offline sleep consolidation)

## 6. Technical Debt Hotspots

1.  **The `brain.py` God Object:** 1000+ lines doing initialization, routing, I/O, file parsing, and loop management. It relies heavily on `except: pass`, so failures are silent.
2.  **File Naming Chaos:** `broca_area.py` contains the core `MergenBrain` class. `main.py` is not the main entry point; `Mergen.py` is.
3.  **Path Conflicts:** The production brain saves its state to `./mergen_weights.mx`. The offline sleep consolidator (`dream.py`) tries to load `./mergen.mx`. They are looking at different files.
4.  **Missing Dependency:** `WernickeArea` (the semantic text processor) fails silently because `sentence-transformers` was removed from `requirements.txt`.

## 7. Known Broken/Orphaned Systems

1.  **Orphaned Learning Engine:** `learning/hebbian_engine.py` is the crown jewel of the repo but has zero inbound imports. It is not even exported in `learning/__init__.py`.
2.  **Orphaned Autonomy:** `limbic_executive_layer.py` has no inbound imports. It was built to wrap the learning engine, but the wiring was never completed.
3.  **Broken Sleep:** `dream.py` calls a method on `LanguageEngine` to consolidate memories, but `dream.py` itself is never executed by any part of the system.

## 8. Recommended First Modifications

If you want to start cleaning up without breaking anything:

1.  **Expose the Learning Engine:** Add `from .hebbian_engine import HybridHebbianLearner` to `learning/__init__.py`.
2.  **Fix the MX File Path:** Change `dream.py` and `main.py` to point to `mergen_weights.mx` so that all systems agree on where the brain's state is stored.
3.  **Delete Dead Code:** Delete `response_synthesizer.py` and `mergen_logic.py`. They are demonstrably unused.
4.  **Fix Wernicke:** Add `sentence-transformers` back to `requirements.txt` so `WernickeArea` can actually load.

## 9. Recommended First Debugging Targets

If you want to understand how the system fails:
1.  Run `Mergen.py`. Type gibberish. Watch how `IntentAnalyzer` defaults to `UNKNOWN` and `broca_generate` kicks in with a canned response.
2.  Search `brain.py` for `except: pass`. Place `print(e)` statements in those blocks to see how many errors the system is silently swallowing during normal operation.

## 10. Estimated Architecture Maturity

**Production (Chatbot): Medium Maturity, Low Ceiling.**
It works, it persists data, and it handles Turkish morphological quirks. However, its core is string-matching and list-appending. It cannot evolve into a reasoning engine without a total rewrite.

**Biological (SNN): Low Maturity, High Ceiling.**
The biophysical models (`anatomy/`, `learning/`) are mathematically correct and architecturally sound. They are currently isolated, headless, and lack a natural language interface. 

The primary engineering task for the next year is not building new features, but wiring the disconnected biological engine (`HybridHebbianLearner` + `LimbicExecutiveLayer`) into the production pipeline to replace the string-matching chatbot.
