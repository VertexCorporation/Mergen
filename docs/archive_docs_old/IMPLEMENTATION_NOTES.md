# IMPLEMENTATION NOTES: MWCC Milestone 1

**Date:** 2026-06-20
**Target:** `cognitive_core.py` (`IntegratedBrainWrapper`)

---

## 1. Objective Achieved
Successfully implemented the skeleton for `IntegratedBrainWrapper` fulfilling Milestone 1. 

## 2. Serialization Compatibility Details
The most critical aspect of Milestone 1 is establishing seamless state persistence (`save_state` and `load_state`) with the `LimbicExecutiveLayer`, without requiring any modifications to `limbic_executive_layer.py`.

*   **Property Delegation:** The wrapper uses Python `@property` decorators to expose the internal state of the `HybridHebbianLearner`.
*   **Setter Requirements:** `limbic_executive_layer.py` explicitly overwrites references during loading (e.g., `self.engine.eligibility = e`). To support this, `@property.setter` methods were implemented for `eligibility`, `trace_pre`, `trace_post`, `_step_count`, and `_da_event_count`.
*   **`.data` Assignment:** `limbic.load_state()` restores the weight matrix via `self.engine.weights.data = w`. This mutates the tensor in place rather than reassigning the variable. Therefore, the `weights` property only requires a getter that returns the `nn.Parameter` from the internal learner.
*   **Read-Only EMA:** `limbic.save_state()` reads `firing_rate_ema`, but `limbic.load_state()` does not restore it. Therefore, `firing_rate_ema` only requires a getter.

## 3. Next Steps (Milestone 2)
The wrapper is currently a hollow shell that directly delegates `forward()` to the Hebbian learner. 
In the next milestone, we will:
1.  Initialize the `CorticalLayer` and `GlobalWorkspace` modules inside the constructor.
2.  Rewrite `forward()` to accept a 1D tensor, reshape it to 2D, pass it through the Cortex and Workspace, and flatten it back to 1D.
3.  Inject the Hebbian learner's weights into the Motor Cortex drive to resolve the Ghost Learner flaw.
