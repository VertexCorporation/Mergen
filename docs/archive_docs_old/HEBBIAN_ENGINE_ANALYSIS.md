# HEBBIAN ENGINE ANALYSIS

**Module:** `learning/hebbian_engine.py`
**Class:** `HybridHebbianLearner`
**Role:** Principal AI Systems Architect
**Date:** 2026-06-19

---

## 1. Inputs

The `HybridHebbianLearner` processes three types of inputs during its learning loop:

1.  **`pre_spikes` (torch.Tensor):** The activity of the pre-synaptic neurons (e.g., sensory input or lower-level cortex). Shape: `(batch_size, n_pre)` or `(n_pre,)`.
2.  **`post_spikes` (torch.Tensor):** The activity of the post-synaptic neurons (e.g., motor output or higher-level cortex). Shape: `(batch_size, n_post)` or `(n_post,)`.
3.  **`reward` (float):** An external scalar reward signal representing the success or failure of an action.
4.  **`new_value_estimate` (Optional[float]):** The updated expected future reward, used for Temporal Difference learning in the Critic.

## 2. Outputs

The learner does not output a continuous action space directly. Its primary "outputs" are state updates to the synaptic weights and a telemetry dictionary.

1.  **Forward Pass (`forward()`):** Returns a discrete spike tensor `(0.0 or 1.0)` of shape `(n_post,)` or `(batch, n_post)`. This is the inference output.
2.  **Learning Step (`learning_step()`):** Returns a telemetry dictionary containing learning metrics (RPE, LTP/LTD magnitudes, delta W, sparsity). It modifies `self.weights` in place.
3.  **Dopamine Modulation (`apply_dopamine()`):** Returns the `delta_w` tensor representing the final weight changes applied in that step.

## 3. Internal Learning Mechanisms

The class implements a sophisticated "Three-Factor Learning Rule" combining local unsupervised Hebbian updates with global reinforcement learning:

1.  **Surrogate Gradients (`forward()`):** Uses a custom `SpikingActivation` to pass discrete spikes forward while allowing continuous gradients to flow backward, bridging spiking networks with PyTorch autograd.
2.  **STDP Traces (`update_traces()`):** Maintains exponentially decaying memory traces of pre- and post-synaptic spikes (`trace_pre`, `trace_post`).
3.  **Eligibility Accumulation (`update_traces()`):** Calculates local Hebbian changes (LTP/LTD) via soft-bounded STDP, but *does not apply them immediately*. Instead, it accumulates them in an `eligibility` trace buffer that decays slowly.
4.  **Dopamine Gating (`apply_dopamine()`):** Computes a Reward Prediction Error (RPE) via `DopamineModulator`. It scales the accumulated `eligibility` trace by the RPE.
    *   `RPE > 0` (Good surprise): Eligibility trace becomes a positive weight update.
    *   `RPE < 0` (Bad surprise): Eligibility trace becomes a negative weight update.
5.  **Homeostatic Normalization (`_homeostatic_normalization()`):** Prevents runaway excitation and silent neurons through three mechanisms:
    *   Hard bounding: Clamping weights to `[w_min, w_max]`.
    *   Synaptic Scaling: Normalizing the column sums of the weight matrix to a `target_input_sum`.
    *   Firing Rate Regulation: Adjusting weights to maintain a long-term `target_firing_rate` (using EMA).

## 4. Dependencies

**Internal Imports:**
*   `from .gradients import SurrogateSpike, SpikingActivation`
*   `from .stdp import STDPMechanism`
*   `from .rl_agent import DopamineModulator`

**External Imports:**
*   `torch`, `torch.nn`
*   `typing.Optional`, `typing.Dict`

## 5. Architecture B Modules That Can Directly Use It

Because `HybridHebbianLearner` abstracts away specific network topologies and requires only `pre_spikes` and `post_spikes`, it is highly compatible with Architecture B's biophysical models.

1.  **`anatomy/cortical_sheet.py` (`CorticalLayer`):** Can use `HybridHebbianLearner` to replace or supplement its simple STDP trace logic for learning the `W_forward` weights between layers. The `CorticalLayer` generates the requisite discrete spike tensors.
2.  **`connectivity/global_workspace.py` (`GlobalWorkspace`):** Could theoretically use this learner for its input/output projections, allowing the workspace to learn which cortical areas to attend to based on reward.
3.  **`main.py`:** The main training loop currently manually calculates readouts and STDP. It can be refactored to delegate learning entirely to an instance of `HybridHebbianLearner`.

## 6. Architecture A Modules That Could Potentially Use It

Architecture A does not generate discrete spikes; it passes continuous embeddings through frozen `nn.Linear` layers. Direct integration is impossible without architectural changes.

1.  **`broca_area.py` (`MergenBrain`):** This is the only theoretical candidate. To use `HybridHebbianLearner`, `MergenBrain` would have to be completely rewritten to operate on spike trains rather than float vectors. The `hebbian_trace` array would be replaced by the engine's internal traces.
2.  **`limbic_executive_layer.py`:** This is designed to wrap an engine that conforms to a specific duck-typed interface. It expects `engine.forward()`, `engine.update_traces()`, and `engine.apply_dopamine()`. `HybridHebbianLearner` *perfectly* matches this expected interface.

## 7. Required Integration Points

To activate `HybridHebbianLearner`, the following steps are necessary:

1.  **Fix Package Export:** The class `HybridHebbianLearner` must be exported in `learning/__init__.py`. Currently, it is completely hidden.
2.  **Integration with `limbic_executive_layer.py`:** `LimbicExecutiveLayer` was clearly built to wrap this engine. It needs to be instantiated with a real `HybridHebbianLearner` instead of the `MockEngine` currently used in its test block.
3.  **Integration with Architecture B (`main.py`):** The `main.py` loop must instantiate `HybridHebbianLearner` for the connections between `SpikeEncoder` → `CorticalLayer` → `Readout`. The manual STDP logic in `main.py` should be stripped out and delegated to the engine.

## 8. Missing Dependencies

There are no missing external dependencies. It relies purely on PyTorch.

## 9. Runtime Requirements

*   **Compute:** Requires PyTorch. Designed to run on either CPU or CUDA (accepts a `device` argument).
*   **Memory:** Creates several state buffers (`trace_pre`, `trace_post`, `eligibility`, `firing_rate_ema`) of size `O(N)` or `O(N*M)`. For very large layers, `eligibility` (a dense matrix) will consume significant VRAM.
*   **Statefulness:** The engine is highly stateful. It must be reset (`reset_traces()`) between independent episodes to prevent trace bleed-over, and its state (`weights`, `eligibility`, `firing_rate_ema`) must be serialized to disk for persistent learning across sessions.
