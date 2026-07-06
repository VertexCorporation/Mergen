# COGNITIVE CORE INTEGRATION FEASIBILITY REPORT

**Target Modules:**
1. `learning/hebbian_engine.py` (HybridHebbianLearner)
2. `limbic_executive_layer.py` (LimbicExecutiveLayer)
3. `connectivity/global_workspace.py` (GlobalWorkspace)
4. `anatomy/cortical_sheet.py` (CorticalLayer)

**Date:** 2026-06-20
**Auditor:** Principal AI Systems Architect

---

## 1. Were these modules designed to work together?

Yes. Code evidence strongly indicates a latent, uncompleted plan to unify these specific components into a single biologically plausible cognitive architecture.

*   **Limbic + Hebbian:** `LimbicExecutiveLayer` contains a `save_state()` method that attempts to serialize `self.engine.eligibility`, `self.engine.trace_pre`, and `self.engine.firing_rate_ema`. These exact attribute names exist **only** in `HybridHebbianLearner`. `LimbicExecutiveLayer` was undeniably written specifically to act as the metacognitive driver for the Hebbian engine.
*   **Cortex + Workspace:** `GlobalWorkspace` and `CorticalLayer` already interact cleanly inside the headless `main.py` loop. The workspace is designed to route continuous or flattened representations of the discrete spiking output from the cortical sheets.
*   **Hebbian + Cortex:** `HybridHebbianLearner` accepts discrete `pre_spikes` and `post_spikes` and implements surrogate gradients. `CorticalLayer` generates exactly these 2D discrete spike matrices.

## 2. Existing Integration Points

*   **State Serialization:** `LimbicExecutiveLayer.save_state()` and `load_state()` perfectly map to the state tensors of `HybridHebbianLearner` (Layer 2) and `BrocaArea` (Layer 3).
*   **Reward Pipeline:** `LimbicExecutiveLayer` evaluates text, calculates a heuristic reward, and calls `self.engine.apply_dopamine(reward)`. `HybridHebbianLearner` has exactly this method signature.
*   **Routing Loop:** `main.py` demonstrates the functional pipeline: `Sensory Cortex → Global Workspace → Motor Cortex`. 

## 3. Missing Integration Points

While they were designed for each other, they are not currently wired together. The critical missing piece is an **Orchestrator/Wrapper**.

*   `LimbicExecutiveLayer` expects its `self.engine` to accept text embeddings in a single `forward(tensor)` call.
*   However, the biophysical modules (`CorticalLayer`, `GlobalWorkspace`) and the learning engine (`HybridHebbianLearner`) operate on spikes, not text. 
*   **Missing Link:** There is no single class that wraps the Cortex/Workspace/Hebbian triad and exposes a clean `forward(embeddings)` interface that the Limbic layer expects. 

## 4. Required Data Flow

To bridge the gap without modifying the modules, the data must flow as follows:

1.  **Input:** User text is received by `LimbicExecutiveLayer.respond()`.
2.  **Encoding:** The text is mapped to currents via `SpikeEncoder`.
3.  **Sensory Processing:** Currents drive `CorticalLayer (Sensory)`. Output is `spikes_s`.
4.  **Routing:** `spikes_s` enters `GlobalWorkspace`. Output is `gw_drive`.
5.  **Motor Processing:** `gw_drive` triggers `CorticalLayer (Motor)`. Output is `spikes_m`.
6.  **Decoding:** `spikes_m` is read out into a text response via `BrocaArea` or a linear readout.
7.  **Evaluation:** `LimbicExecutiveLayer` evaluates the outcome and calculates a `reward`.
8.  **Learning:** The wrapper calls `HybridHebbianLearner.update_traces(spikes_s, spikes_m)` and `apply_dopamine(reward)` to adjust the synaptic weights connecting the cortical layers.

## 5. Which module should be the entry point?

The entry point for the user application (e.g., `Mergen.py`) should be `LimbicExecutiveLayer`. It provides the high-level `respond()` method for active chat, handles threading for the background DMN (Default Mode Network), and manages state persistence (`.mx` files).

## 6. Which module should drive the main loop?

**`LimbicExecutiveLayer`** must drive the main cognitive loop. 

It is the only module capable of autonomous pacing. Its `wake_up()` method spawns the `_dmn_loop()`, which runs continuously in the background at 2Hz. This loop allows the system to think, consolidate memories, and apply spontaneous Hebbian updates even when the user is not typing. It is the "heartbeat" of the biological architecture.

## 7. Feasibility of a Minimal Working Cycle

**Is it feasible to create a minimal working cycle without modifying the architecture?**

**Yes, absolutely.**

A minimal working cycle can be achieved by creating a new file (e.g., `cognitive_core.py`) containing a wrapper class (e.g., `IntegratedBrainWrapper`). 

This wrapper would:
1.  Instantiate `CorticalLayer`, `GlobalWorkspace`, and `HybridHebbianLearner` internally.
2.  Implement the duck-typed interface expected by the Limbic layer: `forward()`, `update_traces()`, `apply_dopamine()`.
3.  Handle the internal translation between continuous embeddings (from the Limbic layer) and spikes (for the Cortical layers).

You can then instantiate `LimbicExecutiveLayer(engine=IntegratedBrainWrapper())`. This entirely fulfills the architectural vision without requiring a single line of code to be modified in the existing modules.
