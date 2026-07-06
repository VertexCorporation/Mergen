# MWCC V2 ARCHITECTURE DESIGN

**Date:** 2026-06-20
**Auditor:** Principal AI Systems Architect
**Scope:** Resolution of tensor dimensionality conflicts via adapter integration within the `IntegratedBrainWrapper`.

---

## 1. Optimal Dimensions
To satisfy the rigid boundaries of the frozen layers without truncation or PyTorch crashes, the dimensions must be decoupled using adapter layers:

*   **Sensory Input (Wernicke):** $384$ (Fixed by `all-MiniLM-L6-v2`)
*   **Cortical Grid (Processing):** $32 \times 32 = 1024$ (Provides a spacious, squarable topological medium for the biological cortex and workspace)
*   **Motor Output (Broca):** $668$ (Fixed by the default concept vocabulary size)

## 2. Adapter Placement & Strategy
The wrapper will internally manage two distinct adapters to funnel data through the biological core:

*   **Sensory Adapter ($384 \rightarrow 1024$):** 
    *   **Implementation:** A frozen, randomly initialized `nn.Linear(384, 1024, bias=False)`. 
    *   **Justification:** Biologically, the retina/cochlea maps to the primary sensory cortex via fixed topographical projections. A fixed random sparse projection acts as a spatial scatter-gather, converting dense 1D semantic embeddings into a 2D spatial pattern for the `CorticalLayer` without requiring backpropagation.
*   **Motor Adapter ($1024 \rightarrow 668$):**
    *   **Implementation:** The `HybridHebbianLearner`.

## 3. The HybridHebbianLearner as the Motor Adapter
**Should the `HybridHebbianLearner` be used as the motor adapter? YES.**

This brilliantly solves both the dimensionality mismatch and the "Ghost Learner" flaw identified in the risk assessment. By setting the learner's `n_pre = 1024` and `n_post = 668`, its internal weight matrix acts directly as the $1024 \rightarrow 668$ Motor Adapter.

**The Crucial Trick:** `LimbicExecutiveLayer._spontaneous_fire()` expects `engine.n_pre` to equal $384$ so it can generate random sensory noise. The wrapper must expose `@property def n_pre(self): return 384`, while internally passing `n_pre=1024` to the Hebbian learner. During `forward()`, the wrapper caches the 1024-dimensional motor cortex output. When `LimbicExecutiveLayer` calls `wrapper.update_traces(pre_384, post_668)`, the wrapper intercepts it and calls `learner.update_traces(cached_motor_1024, post_668)`. The Hebbian learning now directly associates *internal cortical states* with *spoken words*.

## 4. Updated Data Flow (Pipeline)

1.  **Input:** `pre_1d` (Shape: 384) arrives from Wernicke or DMN.
2.  **Sensory Projection:** `sensory_drive = pre_1d @ sensory_adapter.weight.T` (Shape: 1024)
3.  **Sensory Cortex:** `pre_2d = sensory_drive.view(32, 32)` $\rightarrow$ `CorticalLayer` $\rightarrow$ `spikes_s_2d` (Shape: 32x32)
4.  **Global Workspace:** `spikes_s_1d` (Shape: 1x1024) $\rightarrow$ `GlobalWorkspace` $\rightarrow$ `gw_out_1d` (Shape: 1x1024)
5.  **Motor Cortex:** `gw_out_2d = gw_out_1d.view(32, 32)` $\rightarrow$ `CorticalLayer` $\rightarrow$ `motor_spikes_2d` (Shape: 32x32)
6.  **State Caching:** `motor_spikes_1d = motor_spikes_2d.flatten()` (Shape: 1024) $\rightarrow$ Save to `self._cached_motor_spikes`.
7.  **Hebbian Motor Adapter:** `motor_spikes_1d` $\rightarrow$ `HybridHebbianLearner.forward()` $\rightarrow$ `post_1d` (Shape: 668).
8.  **Output:** Return `post_1d` to Limbic/Broca.

## 5. Revised Milestone 2 Implementation Plan

Based on the MWCC V2 Architecture, Milestone 2 will now proceed as follows:

1.  **Update `cognitive_core.py` Initialization:**
    *   Instantiate `self.sensory_cortex` and `self.motor_cortex` as `CorticalLayer(32, 32)`.
    *   Instantiate `self.workspace` as `GlobalWorkspace(1024)`.
    *   Update `self.learner` to `HybridHebbianLearner(n_pre=1024, n_post=668)`.
    *   Create `self.sensory_adapter` (Linear 384 $\rightarrow$ 1024, `requires_grad=False`).
2.  **Override Property Passthroughs:**
    *   Hardcode `@property def n_pre(self): return 384` to prevent DMN crashes.
    *   Hardcode `@property def n_post(self): return 668`.
3.  **Implement the Forward Pipeline:**
    *   Code the exact tensor flow outlined in Section 4, ensuring all batch dimensions `unsqueeze(0)` and `squeeze(0)` match the `GlobalWorkspace` requirements.
4.  **Implement the Trace Updater:**
    *   `def update_traces(self, pre_1d, post_1d)`: Ignore `pre_1d` (384). Instead, use `self._cached_motor_spikes` (1024) to update the Hebbian traces.
