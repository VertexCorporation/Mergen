# PRE-IMPLEMENTATION RISK AUDIT: INTEGRATED BRAIN WRAPPER

**Date:** 2026-06-20
**Auditor:** Principal AI Systems Architect
**Target:** `IntegratedBrainWrapper` (Limbic + Hebbian + Cortex + Workspace)

This document outlines the critical risks associated with implementing the `IntegratedBrainWrapper` as previously specified.

---

## 1. Hidden Assumptions

*   **The "Ghost Learner" Flaw:** The pseudocode delegates `update_traces()` and `apply_dopamine()` to the `HybridHebbianLearner`, meaning the learner successfully updates its internal weights based on reward. However, the learner's weights are never actually multiplied against the sensory spikes to drive the motor cortex during `forward()`. The learner is learning, but it is a "ghost" with no influence over the output. The wrapper must explicitly add a `direct_drive = pre_spikes @ self.learner.weights` step to the Motor Cortex input.
*   **Wernicke / Broca Vocabulary Mapping:** `LimbicExecutiveLayer` expects `n_post` to correspond to `BrocaArea`'s vocabulary size. `CorticalLayer` requires `n_neurons = H * W`. Therefore, the 2D grid dimensions (`H * W`) MUST exactly equal the vocabulary size, or the semantic mapping will be completely misaligned.

## 2. Shape Inconsistencies

*   **Wernicke Mismatch:** If `SpikeEncoder.perceive()` outputs a `(T, N_INPUT)` tensor where `N_INPUT != H * W`, the wrapper's `pre_1d.view(self.H, self.W)` call will immediately throw a `RuntimeError: shape invalid for input size`.
*   **Workspace Flattening:** `GlobalWorkspace` expects `(Batch, N)`. `spikes_s_2d.flatten().unsqueeze(0)` correctly forms a `(1, H*W)` batch, but `GlobalWorkspace` outputs `(1, H*W)`. The `gw_drive_2d = gw_drive.view(self.H, self.W)` call will fail if the batch dimension is not squeezed out first (`gw_drive.squeeze(0).view(...)`).

## 3. Serialization Risks

*   **Amnesiac Workspace (CRITICAL):** `LimbicExecutiveLayer.save_state()` hardcodes the variables it saves: `weights`, `eligibility`, `traces`, etc. It knows nothing about `GlobalWorkspace`'s `W_up`, `W_lat`, and `W_down` matrices. Since `GlobalWorkspace` randomly initializes these matrices with `nn.init.orthogonal_` on creation, **all workspace routing will be randomized on every reboot**, instantly destroying the network's structural integrity.
*   **State Desynchronization:** `CorticalLayer` maintains critical biophysical state vectors (`v`, `a`, `theta`). These are also completely ignored by the `.mx` serialization protocol and will reset to zero/baseline on restart.
*   **JSON OOM Latency:** The `.mx` protocol saves state by calling `.tolist()` and running it through `json.dumps()`. If `H*W = 10,000` (a small vocabulary), the Hebbian weight matrix is `10,000 x 10,000` (100 million floats). Serializing 100M floats to JSON will cause massive UI freezes (minutes) or outright Out-Of-Memory (OOM) crashes on shutdown.

## 4. Training Stability Risks

*   **DMN Race Conditions:** `LimbicExecutiveLayer` spawns a background thread for the Default Mode Network (`_dmn_loop`) which occasionally triggers spontaneous firings. If the user chats at the exact same moment the DMN fires, concurrent access to `CorticalLayer`'s non-thread-safe state (`self.v`, `self.a`) will corrupt the membrane potentials.
*   **Trace Bleed-Over:** `LimbicExecutiveLayer.respond()` loops over time $t$, accumulating `neural_intent`. But nowhere does it call `reset_state()` on the cortical layers between *different* user chats. The residual membrane voltage (`v`) and adaptation (`a`) from the previous chat will bleed into the next one.

## 5. Runtime Performance Risks

*   **FFT Convolution Overhead:** `CorticalLayer` uses `fft_convolve2d` for lateral interactions on every time step $t$. If `SpikeEncoder` generates 50 time steps per word, and the user types a 10-word sentence, the wrapper performs 500 FFT convolutions sequentially. On CPU, this will cause severe response lag.

## 6. Missing Dependencies

*   The integration does not technically require new PyPI packages, but it absolutely requires `sentence-transformers` (which is missing from `requirements.txt`) so that `WernickeArea` can generate the initial spike train that `LimbicExecutiveLayer` feeds into the wrapper. Without it, the Limbic layer feeds random noise to the wrapper.

---

## 7. Top 10 Likely Failure Points

| # | Failure Point | Severity | Likelihood | Mitigation |
|---|---|---|---|---|
| 1 | **Ghost Learner (No forward drive)** | CRITICAL | 100% | Add `direct_drive = pre_1d @ self.learner.weights` to `gw_drive` in the wrapper's `forward()` method. |
| 2 | **Amnesiac Workspace (Weights lost on boot)** | CRITICAL | 100% | The wrapper must intercept `save_state()` somehow, or `LimbicExecutiveLayer` must be modified. Alternatively, inject deterministic seeds for `GlobalWorkspace`. |
| 3 | **JSON Serialization OOM / Freeze** | HIGH | 90% | If vocab > 3000, JSON serialization will choke. The architecture must switch to `torch.save()`, which violates the strict "no modification" constraint of the Limibic layer. |
| 4 | **Shape Crash in `view()`** | HIGH | 80% | Dynamically pad or pool `pre_1d` in the wrapper if `len(pre_1d) != H * W`. |
| 5 | **DMN Thread Race Conditions** | HIGH | 50% | Add a `threading.Lock()` inside the wrapper's `forward()` and `update_traces()` methods to serialize access. |
| 6 | **Cortical State Bleed-Over** | MEDIUM | 100% | The wrapper must detect the start of a new spike train (e.g., if `t=0` or via a timeout heuristic) and call a `reset_cortical_state()` method internally. |
| 7 | **Workspace Batch Dimension Crash** | HIGH | 100% | Fix the pseudocode: `gw_drive_2d = gw_drive.squeeze(0).view(self.H, self.W)`. |
| 8 | **Silent Dopamine Exceptions** | MEDIUM | 40% | `Limbic` swallows dopamine exceptions. Ensure the wrapper's `apply_dopamine` perfectly matches the signature and handles detached tensors properly. |
| 9 | **Missing `sentence-transformers`** | HIGH | 100% | Add to `requirements.txt`. Without it, `wernicke` falls back to `None` and the Limibic layer feeds random noise vectors. |
| 10 | **`theta` Threshold Saturation** | LOW | 30% | `CorticalLayer`'s threshold `theta` might climb too high if DMN runs infinitely. The wrapper should enforce `theta` clamping internally. |
