# INTEGRATED BRAIN WRAPPER SPECIFICATION

**Date:** 2026-06-20
**Auditor:** Principal AI Systems Architect

This document details the exact integration parameters for the `IntegratedBrainWrapper` class, designed to unify the Mergen cognitive architecture without modifying any existing source files.

---

## 1. Method Verification

**Do all required methods already exist?**
Yes. The required integration methods map cleanly across the modules:
*   `LimbicExecutiveLayer` requires its injected engine to have: `forward()`, `update_traces()`, `apply_dopamine()`.
*   `HybridHebbianLearner` natively implements `forward()`, `update_traces()`, `apply_dopamine()`.
*   `CorticalLayer` and `GlobalWorkspace` both natively implement standard `forward()` passes.

## 2. Shape Compatibility

**Are tensor shapes compatible?**
*No, direct passing will fail. The wrapper must explicitly handle shape mutations.*

*   **Limbic Input:** `LimbicExecutiveLayer.respond()` loops over time $t$ and passes `pre_spikes` as a 1D tensor of shape `(n_pre,)`.
*   **Cortical Input:** `CorticalLayer` explicitly requires a 2D tensor of shape `(H, W)`.
*   **Workspace Input:** `GlobalWorkspace` requires a batched 1D tensor of shape `(Batch, N)`.
*   **Resolution:** The wrapper must reshape the 1D Limbic input into `(H, W)` for the Sensory Cortex, flatten it to `(1, N)` for the Workspace, and reshape the Workspace output back to `(H, W)` for the Motor Cortex. It must then flatten the Motor Cortex output back to `(n_post,)` to return to the Limbic layer. Therefore, `n_pre` and `n_post` must exactly equal `H * W`.

## 3. Data Type Compatibility

**Are data types compatible?**
Yes. All components use PyTorch tensors. `CorticalLayer` outputs discrete float spikes (0.0 or 1.0) via `(v > theta).to(dtype)`. `HybridHebbianLearner` explicitly expects float spikes, and `GlobalWorkspace` uses continuous float tensors. The `LimbicExecutiveLayer` also expects PyTorch tensors to accumulate `neural_intent`.

## 4. Required Wrapper Interface

To satisfy `LimbicExecutiveLayer`'s rigid `.mx` serialization and runtime expectations, the wrapper MUST expose the following:

**Methods:**
*   `forward(pre: torch.Tensor) -> torch.Tensor`
*   `update_traces(pre: torch.Tensor, post: torch.Tensor)`
*   `apply_dopamine(reward: float)`

**Properties (For `save_state` / `load_state`):**
*   `self.weights` (must be a tensor, e.g., the Hebbian learner's weights)
*   `self.eligibility`
*   `self.trace_pre`
*   `self.trace_post`
*   `self.firing_rate_ema`
*   `self._step_count`
*   `self._da_event_count`
*   `self.n_post`
*   `self.device`

## 5. Non-Destructive Feasibility

**Can the wrapper be implemented without modifying any existing files?**
Yes. By utilizing Python's `@property` decorators, the wrapper can expose the internal `HybridHebbianLearner`'s state variables directly to the `LimbicExecutiveLayer`. Because dependency injection is used (`limbic = LimbicExecutiveLayer(mergen_engine=wrapper)`), the limbic layer will unknowingly serialize the internal Hebbian learner's state.

## 6. Likely Integration Errors

1.  **Shape Mismatch Exception:** If `LimbicExecutiveLayer` feeds `(n_pre,)` to `CorticalLayer(external_input)` without reshaping.
2.  **Attribute Error on Serialization:** If the wrapper fails to expose `.data` on the weights tensor. `LimbicExecutiveLayer` calls `self.engine.weights.data.cpu().tolist()`. The property must return a tensor that has a `.data` attribute.
3.  **Device Mismatch (CUDA/CPU):** `LimbicExecutiveLayer` does `post.detach().cpu()`, but if the wrapper initializes components on different devices without explicitly moving inputs, PyTorch will throw a device mismatch error during `CorticalLayer` computation.

---

## 7. Pseudocode Specification

```python
import torch
import torch.nn as nn

class IntegratedBrainWrapper(nn.Module):
    def __init__(self, H, W, device='cpu'):
        super().__init__()
        self.H = H
        self.W = W
        self.device = device
        self.n_neurons = H * W
        
        # Expose attributes expected by LimbicExecutiveLayer
        self.n_post = self.n_neurons
        
        # 1. Biological Cortex
        self.sensory_cortex = CorticalLayer(H, W, ...)
        self.motor_cortex = CorticalLayer(H, W, ...)
        
        # 2. Global Router
        self.workspace = GlobalWorkspace(input_dim=self.n_neurons, ...)
        
        # 3. Hebbian Learner (learns connections across the workspace)
        self.learner = HybridHebbianLearner(n_pre=self.n_neurons, n_post=self.n_neurons, ...)

    # ==========================================
    # PASSTHROUGH PROPERTIES FOR LIMBIC LAYER .mx
    # ==========================================
    @property
    def weights(self): return self.learner.weights
    @property
    def eligibility(self): return self.learner.eligibility
    @property
    def trace_pre(self): return self.learner.trace_pre
    @property
    def trace_post(self): return self.learner.trace_post
    @property
    def firing_rate_ema(self): return self.learner.firing_rate_ema
    @property
    def _step_count(self): return self.learner._step_count
    @property
    def _da_event_count(self): return self.learner._da_event_count

    # ==========================================
    # RUNTIME INTERFACE
    # ==========================================
    def forward(self, pre_1d: torch.Tensor) -> torch.Tensor:
        # 1. Reshape 1D -> 2D
        pre_2d = pre_1d.view(self.H, self.W)
        
        # 2. Sensory processing
        spikes_s_2d = self.sensory_cortex(pre_2d)
        
        # 3. Routing (flatten for Workspace)
        gw_drive = self.workspace(spikes_s_2d.flatten().unsqueeze(0))
        gw_drive_2d = gw_drive.view(self.H, self.W)
        
        # 4. Motor processing
        spikes_m_2d = self.motor_cortex(gw_drive_2d)
        
        # 5. Return 1D for Broca/Limbic
        return spikes_m_2d.flatten()

    def update_traces(self, pre_1d: torch.Tensor, post_1d: torch.Tensor):
        # Delegate trace update to the Hebbian learner
        self.learner.update_traces(pre_1d, post_1d)

    def apply_dopamine(self, reward: float):
        # Apply RPE and update weights
        self.learner.apply_dopamine(reward)
```
