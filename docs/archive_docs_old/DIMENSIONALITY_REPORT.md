# DIMENSIONALITY REPORT: SENSORY-MOTOR INTEGRATION

**Date:** 2026-06-20
**Auditor:** Principal AI Systems Architect
**Scope:** Dimensionality alignment between Wernicke (Input), Limbic/Engine (Processing), and Broca (Output).

---

## 1. Exact Dimensionality Expected by LimbicExecutiveLayer
*   **Input (`pre`):** A 1D tensor of size $N_{pre}$. This size is dynamically inherited from whatever `wernicke.perceive()` returns.
*   **Engine Forward:** `LimbicExecutiveLayer` loops over time $T$ and feeds the 1D slice into `engine.forward(pre)`.
*   **Output (`post`):** Expects `engine.forward()` to return a 1D tensor of size $N_{post}$. It accumulates these into a `neural_intent` tensor of the same size.

## 2. Exact Dimensionality Produced by WernickeArea
*   **Shape:** 2D tensor `(time_window, n_neurons)`.
*   **Default Values:** `time_window = 50`. `n_neurons = 384`.
*   **Constraint:** The $N_{pre}$ of Wernicke defaults to 384 because it is hard-coupled to the semantic embedding dimension of the `all-MiniLM-L6-v2` transformer model.

## 3. Exact Dimensionality Expected by BrocaArea
*   **Input (`neural_intent`):** A 1D tensor.
*   **Default Values:** `n_neurons = 668` (which defaults to a `vocab_size` of 668 concepts).
*   **Behavior:** `BrocaArea.generate()` has a hard-coded safety net (lines 867-876). If the input `neural_intent` does not match `vocab_size`, it will forcefully **pad with zeros** or **truncate**.

## 4. Must H*W Equal Vocabulary Size?
*   **To prevent PyTorch crashes?** NO. `BrocaArea` will silently pad or truncate the tensor to prevent shape mismatch exceptions.
*   **To maintain semantic integrity?** YES. If $H \times W = 384$ (to match Wernicke) and `vocab_size = 668`, the `BrocaArea` will pad the 384-length vector with 284 zeros. Because those 284 neurons will never receive activation from the engine, **Mergen will be permanently unable to speak 42% of its vocabulary**. Conversely, if Wernicke output is truncated to fit a smaller cortex, sensory input is destroyed.

## 5. Is an Adapter Layer Required?
**YES. An Adapter Layer is absolutely mandatory within the `IntegratedBrainWrapper`.**

Currently, the system is suffering from a massive dimensional misalignment:
*   Wernicke pushes **384** dimensions.
*   Cortex expects **H × W** (e.g., 1024 for a 32x32 grid).
*   Broca expects **668** dimensions.

If the wrapper simply reshapes tensors, it will crash. If it truncates/pads, it will induce cognitive blindness and aphasia. 

**Proposed Adapter Topology for `IntegratedBrainWrapper`:**
1.  **Sensory Adapter:** `nn.Linear(384, H*W)` — Projects the dense semantic embedding into a larger, sparse cortical grid.
2.  **Biological Processing:** `CorticalLayer(H, W) -> GlobalWorkspace -> CorticalLayer(H, W)`
3.  **Motor Adapter:** `nn.Linear(H*W, 668)` — Compresses the cortical grid back down to the specific conceptual vocabulary expected by `BrocaArea`. 

*Note: The `HybridHebbianLearner` could potentially serve as the Motor Adapter if its $n_{pre} = H*W$ and its $n_{post} = 668$, perfectly solving the "Ghost Learner" flaw by wiring it directly into the output path.*
