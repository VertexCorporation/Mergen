# RISK VALIDATION REPORT

**Date:** 2026-06-20
**Auditor:** Principal AI Systems Architect
**Scope:** Verification of pre-implementation risks against actual source code.

This report cross-references the theoretical risks identified in the `INTEGRATION_RISK_ASSESSMENT.md` with hard code evidence from the repository.

---

## 1. Ghost Learner Flaw
**Status: Confirmed**

*   **Evidence:** `HybridHebbianLearner` tracks and updates a dense weight matrix (`self.weights = nn.Parameter(...)` in `learning/hebbian_engine.py:99`). However, in a wrapped architecture that uses `GlobalWorkspace` for routing, there is no automatic path for `self.weights` to influence the `CorticalLayer`. If the wrapper simply delegates `update_traces()` to the learner but does not manually inject `pre_spikes @ self.weights` into the motor cortex drive, the learned weights are mathematically isolated from the output.

## 2. Amnesiac Workspace
**Status: Confirmed**

*   **Evidence:** `connectivity/global_workspace.py` lines 35-37 randomize routing weights on instantiation: `nn.init.orthogonal_(self.W_up.weight, gain=0.5)`. 
*   **Evidence:** `limbic_executive_layer.py` lines 206-216 (`save_state`) strictly hardcode the engine attributes to save: `weights`, `eligibility`, `trace_pre`, `trace_post`, `firing_rate_ema`. 
*   **Conclusion:** The Limbic layer has zero awareness of the `GlobalWorkspace` internal state. The routing topology will completely scramble on every reboot, inducing permanent amnesia.

## 3. JSON Serialization OOM
**Status: Confirmed**

*   **Evidence:** `limbic_executive_layer.py` line 207 executes `self.engine.weights.data.cpu().tolist()`. Line 240 calls `json.dumps(state)`.
*   **Conclusion:** A PyTorch tensor converted to a Python list of floats and stringified via JSON is highly inefficient. If the grid is 100x100 (`H*W = 10,000`), the dense matrix is 100 million floats. This will cause catastrophic latency or OOM crashes during the `shutdown()` hook.

## 4. Shape Crash in `view()`
**Status: Confirmed**

*   **Evidence:** `limbic_executive_layer.py` line 591 passes a 1D slice of the spike train: `pre = spike_train[t]; post = self.engine.forward(pre)`.
*   **Conclusion:** If the `WernickeArea` outputs a dimension that does not perfectly equal `H * W`, any `pre.view(H, W)` command in the wrapper will trigger an immediate `RuntimeError` from PyTorch.

## 5. DMN Thread Race Conditions
**Status: NOT CONFIRMED (Mitigated)**

*   **Evidence:** The assumption that the background Default Mode Network and the foreground chat loop would collide is incorrect. `limbic_executive_layer.py` correctly implements a `threading.RLock()`:
    *   Line 377 (`_spontaneous_fire`): `with self._lock:`
    *   Line 589 (`respond`): `with self._lock:`
*   **Conclusion:** Access to `self.engine` is strictly serialized. Race conditions mutating membrane potentials will not occur.

## 6. Cortical State Bleed-Over
**Status: Confirmed**

*   **Evidence:** `anatomy/cortical_sheet.py` lines 85-88 demonstrate continuous Euler integration of membrane voltage `self.v`. 
*   **Evidence:** `limbic_executive_layer.py` never calls a `reset_state()` or `clear_voltage()` method on the engine between separate chat messages. 
*   **Conclusion:** Membrane potentials and refractory periods from a previous user interaction will bleed seamlessly into the start of the next interaction, regardless of how much real-time has passed.

## 7. Silent Dopamine Exceptions
**Status: Confirmed**

*   **Evidence:** `limbic_executive_layer.py` lines 545-548 explicitly swallow exceptions during critical negative reward application:
    ```python
    try:
        self.engine.apply_dopamine(reward=-0.8)
    except Exception:
        pass
    ```
*   **Conclusion:** If the wrapper's `apply_dopamine` method fails (e.g., due to a detached tensor or device mismatch), the Limbic layer will silently ignore the failure. The user will think Mergen was corrected, but no synaptic updates will have occurred.

## 8. Missing sentence-transformers
**Status: Confirmed**

*   **Evidence:** Known missing dependency from `requirements.txt`.
*   **Evidence:** If `WernickeArea` fails to load, `limbic_executive_layer.py` falls back to generating random noise: `neural_intent = torch.rand(self.engine.n_post) * 0.5` (line 597). Mergen will function, but it will be hallucinating random semantic inputs.
