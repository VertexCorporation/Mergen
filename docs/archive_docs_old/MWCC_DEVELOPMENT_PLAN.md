# MINIMUM WORKING COGNITIVE CORE (MWCC) DEVELOPMENT PLAN

**Date:** 2026-06-20
**Target:** `IntegratedBrainWrapper` (Composition Only)
**Constraints:** NO modification to existing source files.

---

## 1. Milestones

**Milestone 1: Interface Stubbing & Serialization Compliance**
*   Create `cognitive_core.py`.
*   Implement `IntegratedBrainWrapper` with empty `forward`, `update_traces`, and `apply_dopamine` methods.
*   Instantiate `HybridHebbianLearner` internally.
*   Implement all `@property` getters (e.g., `self.weights`, `self.eligibility`) to pass through to the internal learner.
*   *Validation:* `LimbicExecutiveLayer` can instantiate the wrapper, call `wake_up()`, and call `save_state()` without crashing.

**Milestone 2: Sub-Component Instantiation & Shape Alignment**
*   Instantiate Sensory `CorticalLayer`, Motor `CorticalLayer`, and `GlobalWorkspace`.
*   Define the exact mapping between `vocab_size` ($N$) and cortical grid dimensions ($H \times W$). E.g., if vocab is 1024, $H=32, W=32$.
*   *Validation:* All components initialize successfully on the same `device` without shape mismatches.

**Milestone 3: Data Plumbing & The Ghost Learner Fix**
*   Implement the `forward()` method:
    1. Reshape 1D input into 2D for Sensory Cortex.
    2. Route 2D spikes through Global Workspace.
    3. **Fix Ghost Learner:** Calculate `direct_drive = pre_spikes @ self.learner.weights`.
    4. Pass `workspace_drive + direct_drive` into Motor Cortex.
    5. Return flattened 1D motor spikes.
*   Implement `update_traces()` by delegating to `self.learner`.
*   *Validation:* Dummy `(N,)` tensor correctly passes through the entire pipeline and returns an `(N,)` output.

**Milestone 4: State Management & Reset Logic**
*   Implement a `reset_cortical_state()` method to clear $V$ (membrane voltage) and $A$ (adaptation) in both Cortical Layers.
*   Develop a heuristic to auto-reset state at the beginning of a new sequence (e.g., if a new chat message arrives).
*   *Validation:* Traces do not bleed across separate input sequences.

---

## 2. Development Order

1.  **Serialization Layer:** Build the wrapper shell and connect it to `LimbicExecutiveLayer`. Test `.mx` loading and saving.
2.  **Forward Pass Pipeline:** Assemble the Cortex -> Workspace -> Cortex flow. Test with dummy tensors.
3.  **Hebbian Integration:** Attach the `HybridHebbianLearner` to the forward pass (`direct_drive`) and trace updates.
4.  **Dopamine Loop:** Connect the reward pathway and test if DMN cycles actually strengthen weights.
5.  **State Isolation:** Add reset mechanics for the cortical sheet voltages.

---

## 3. Test Strategy

**A. Shape Consistency Tests:**
Pass random 1D tensors of size $N$ into the wrapper. Assert that the output is exactly size $N$ and that no PyTorch `view()` errors are raised.

**B. Serializability Tests:**
Run `limbic.save_state()` with a small grid ($10 \times 10$). Assert that the `.mx` file is created. Run `limbic.load_state()` and assert that the loaded weights match the saved weights.

**C. Learning Verification (Ghost Learner Test):**
1. Pass input $A$. Record output $O_1$.
2. Pass input $A$ again. Call `apply_dopamine(reward=1.0)`.
3. Pass input $A$ again. Record output $O_2$.
4. Assert that $O_2 \neq O_1$ (proving that the learned weights are actively influencing the forward pass).

**D. State Bleed Tests:**
Pass input $A$. Verify membrane voltage $V > 0$. Call `reset_cortical_state()`. Assert $V == 0$.

---

## 4. Success Criteria

*   **Zero File Modifications:** The implementation strictly uses dependency injection and composition. No lines in `hebbian_engine.py`, `cortical_sheet.py`, `global_workspace.py`, or `limbic_executive_layer.py` are altered.
*   **Operational DMN:** The background Default Mode Network runs, generates spontaneous activity, and updates synaptic traces without crashing.
*   **Active Learning:** User corrections (negative dopamine) successfully depress firing rates for penalized outputs over subsequent turns.
*   **Stability:** The system can handle 100 consecutive dialogue turns without shape errors, race conditions, or memory leaks.

---

## 5. Failure Criteria

*   **OOM on Shutdown:** The JSON serialization of the weight matrix causes the application to consume >4GB RAM or freeze for >10 seconds during `shutdown()`.
*   **Amnesia:** The system fails to remember learned associations across reboots because the wrapper fails to expose the correct properties to the `LimbicExecutiveLayer` serializer.
*   **Silent Exceptions:** The wrapper throws tensor mismatched device/shape errors that are swallowed by `LimbicExecutiveLayer`'s `try/except` blocks, resulting in a zombie state where the engine "runs" but does no math.
