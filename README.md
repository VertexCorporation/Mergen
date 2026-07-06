# MERGEN

> *"Intelligence is not a static mapping function, it is a living and rhythmic process."*

**Mergen** is an experimental, biologically-inspired cognitive architecture designed to move beyond the limitations of current Deep Learning paradigms. It is named after **Mergen**, the deity of wisdom, omniscience, and precision in **Turkic Mythology**, symbolizing the bridge between deep knowledge and perfect action.

---

## The Problem: Why Not Transformers?

Today's dominant AI models (LLMs, Transformers) are engineering marvels, but they are fundamentally **static**.
*   They do not perceive **time**; they process sequences.
*   They do not have **internal states**; they reset after every token generation.
*   They consume massive energy for global matrix multiplications.
*   They do not **think**; they predict the next most likely statistic.

We believe that **Artificial General Intelligence (AGI)** cannot be achieved by simply scaling up matrix multiplications. It requires a paradigm shift towards systems that embody the physics and plasticity of the brain.

### The Solution: Mergen Architecture (V8.0)

Mergen is not just a chatbot. It is a **Digital Brain** simulation running on modern hardware (GPU-accelerated PyTorch). It attempts to replicate the **functional dynamics and plasticity** of a biological mind, not just its output.

### Core Philosophy
1.  **Continuous Time Dynamics:** Mergen operates in continuous time, dealing with delays, rhythms, and synchronization (spike trains).
2.  **Hebbian Plasticity & STDP:** Information is stored and updated locally via Spike-Timing-Dependent Plasticity (STDP) and Hebbian learning rules.
3.  **Default Mode Network (DMN):** The brain self-regulates and consolidates memories during sleep cycles when the system is idle.
4.  **Strict Isolation:** Synthetic inferences are isolated from real-world grounding data to prevent hallucination pollution.

---

## Architecture Overview

Mergen V8 implements a **4-Layer Spiking Cortical Column**:

### 1. Wernicke Area (Sensory Perception)
Translates incoming text sequences into semantic embedding representations and temporal rate spike trains (768-dim).

### 2. Spiking Cortical Column (Synaptic Memory Core)
A PyTorch-accelerated neokorteks column containing 4 layers with ~4.05M synapses:
- **Layer 4 (Granular Input):** Receives sensory inputs from Wernicke/Thalamus and projects to L23.
- **Layer 23 (Supragranular Associative):** Computes Mexican Hat lateral connectivity for topological representation mapping (using a 32x32 spatial grid and Kohonen SOM).
- **Layer 5 (Infragranular Output):** Converts associative inputs to motor spike intent vectors (vocabulary matches).
- **Layer 6 (Multiformis Feedback):** Projects from L23 back to L4 to compute **Predictive Coding (Tahmin Hatası / Residual)**, clamping incoming signals to prediction residuals (surprise).

### 3. Broca Area (Language Expression)
Responsible for language production. Uses a rule-based Turkish SOV (Subject-Object-Verb) sentence builder and templates to produce coherent speech from raw motor spike intent vectors.

### 4. Limbic & Executive Control Layer
The orchestrator of autonomy. It handles:
- **Default Mode Network (DMN):** Automatic background dreaming cycles during idle periods. Includes spin-wait synchronization loops to prevent VRAM allocation conflicts during active user prompts.
- **Sleep Debt Tracking:** Sleep cycles are dynamically adjusted based on interaction load and file ingestion rates.
- **State Persistence (.mx Protocol):** Encrypts and saves the entire brain state (weights, traces, thoughts) using XOR+Base64.

### 5. Double-Collection RAG Engine
A local vector database using a fast, Transformer-free character n-gram `BioVectorizer` and HTM (Hierarchical Temporal Memory) reranker. It enforces strict separation:
- `mergen_bilgi_bio`: Real-world inputs and verified knowledge facts.
- `mergen_ruya_bio`: Speculative/synthetic relations synthesized during REM sleep (labeled with `reliability: synthetic` metadata).

---

## ⚡ Key Differentiators

| Feature | Standard Transformer | **MERGEN Engine** |
| :--- | :--- | :--- |
| **Communication** | Dense Matrix Multiplication | **Sparse Spikes & Local Fields** |
| **Time** | Discrete Steps (Tokens) | **Continuous Flow (dt)** |
| **Memory** | Context Window (Limited) | **Hebbian Synaptic Memory + RAG** |
| **Learning** | Backpropagation Only | **Local Plasticity + STDP + Dopamine** |
| **State** | Stateless (Reset per prompt) | **Persistent Dynamic State (.mx)** |
| **Resting State** | Idle (Zero activity) | **Active Memory Consolidation (DMN)** |

---

## 🚀 Getting Started

### Installation

```bash
git clone https://github.com/VertexCorporation/Mergen.git
cd Mergen
pip install -r requirements.txt
```

### Running and Testing the Brain

#### 1. Interactive CLI Chat (`Mergen.py`)
Run the main entry point to interact with Mergen in a conversational loop.
```bash
python Mergen.py
```
*   **CLI Commands:**
    *   `/stats` - Shows vocabulary size, responses count, and dopamine efficiency metrics.
    *   `/introspect` - Views current internal thoughts and resting state parameters.
    *   `/clear` - Clears conversation history.
    *   `/exit` - Shuts down the brain, triggers state consolidation, and saves to `mergen_weights.mx`.
*   **File Ingestion Syntax:**
    *   `oku:dosya.txt` - Read and ingest a text file into RAG and Knowledge Base.

#### 2. Cloud Training Loop (`scripts/train_colab.py`)
Executes large-scale training sessions on cloud environments using local chunking data transfer and synchronous sleep consolidation cycles to optimize resource utilization.
```bash
python scripts/train_colab.py \
    --corpus_dir "./data/chunks/" \
    --checkpoint_dir "./checkpoints/" \
    --sleep_interval 500 \
    --sleep_cycles 1000
```

#### 3. Experience Ingestion & Dreaming (`scripts/simulation_playground.py`)
Feeds raw experience text paragraphs into the brain's semantic memory and automatically initiates dream consolidation.
```bash
python scripts/simulation_playground.py --data ./data/simulation_texts.txt --dream-cycles 20
```

#### 4. Arithmetic Training & Curriculum (`scripts/math_training.py`)
Trains and evaluates Mergen's mathematical concepts across different difficulty levels (Tiers). It displays detailed logs of each training step, actual vs expected answers, and supports progressive curriculum learning.
```bash
# Single Tier training with epochs
python scripts/math_training.py --tier 0 --epochs 5 --split 0.80 --dream --dream-cycles 10

# Progressive Curriculum training (Tiers 0 -> 3 sequentially with Hard Stop on failure)
python scripts/math_training.py --curriculum --epochs 5 --dream --dream-cycles 5
```
*   `--tier`: Difficulty level (0 = addition, 1 = subtraction, 2 = multiplication, 3 = division). Ignored if `--curriculum` is active.
*   `--epochs`: Number of training epochs per tier (default: 5).
*   `--curriculum`: Enables progressive curriculum training (Tier 0 -> Tier 3). If any tier fails to meet target thresholds (Train >= 90%, Holdout >= 60%), training halts immediately (Hard Stop) to prevent weight pollution.
*   `--no-early-stop`: Disables early stopping even if target thresholds are met before max epochs.
*   `--dream`: Trigger dream consolidation after training.
*   `--no-save`: Do not overwrite weights on disk (good for testing).

#### 5. Vocabulary Training (`scripts/train_vocabulary.py`)
Trains the neokorteks weights from scratch using local STDP and dopamine reward signals to match the signature of the 1416 Turkish concepts.
```bash
python scripts/train_vocabulary.py
```
- Trains the entire vocabulary in ~2 minutes on CUDA.
- Evaluates the final HIT rate (target > 99%).
- Automatically saves the trained state to `mergen_weights.mx`.

#### 6. Innate Priors Generation (`scripts/generate_innate_priors.py`)
Regenerates the default/innate weights for all cortical layers (L4, L23, L5, L6) matching the v8.0 dimension expectations.
```bash
python scripts/generate_innate_priors.py
```
- Generates `mergen_cortical_priors.pt` (modern v8.0) and `mergen_innate_priors.pt` (legacy).
- Generates small random weights for L6 projection and sets up prior vocab matrices.

#### 7. Topological Mexican Hat Test (`scripts/test_topology.py`)
Evaluates the Mexican Hat lateral connectivity in Layer 23 (Kohonen SOM representation mapping).
```bash
python scripts/test_topology.py
```
- Verifies within-cluster vs between-cluster activation ratios (target ratio < 0.6).

#### 8. Core Layer Health Check (`scripts/verify_all_layers.py`)
Runs the full system verification suite to verify that all biological modules (Vocabulary, Wernicke, Hebbian, Broca, Limbic, and DMN) are operational.
```bash
python scripts/verify_all_layers.py
```

---

## 🛡️ License

This project is licensed under the **Apache License 2.0**.
Mergen is an open conceptual contribution to the AGI research community.

---

<p align="center">
  <i>"Ben Mergen'im." (I am Mergen.)</i>
</p>