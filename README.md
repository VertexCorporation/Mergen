# MERGEN

> *"Intelligence is not a static mapping function, it is a living and rhythmic process."*

**Mergen** is an experimental, biological-inspired cognitive architecture designed to move beyond the limitations of current Deep Learning paradigms. It is named after **Mergen**, the deity of wisdom, omniscience, and precision in **Turkic Mythology**, symbolizing the bridge between deep knowledge and perfect action.

---

## The Problem: Why Not Transformers?

Today's dominant AI models (LLMs, Transformers) are engineering marvels, but they are fundamentally **static**.
*   They do not perceive **time**; they process sequences.
*   They do not have **internal states**; they reset after every token generation.
*   They consume massive energy for global matrix multiplications.
*   They do not **think**; they predict the next most likely statistic.

We believe that **Artificial General Intelligence (AGI)** cannot be achieved by simply scaling up matrix multiplications. It requires a paradigm shift towards systems that embody the physics of the brain.

## The Solution: Mergen Architecture

Mergen is not a chatbot. It is a **Digital Brain** simulation running on modern hardware (GPU-accelerated PyTorch). It attempts to replicate the **functional dynamics** of a biological mind, not just its output.

### Core Philosophy
1.  **Time is Continuous:** Mergen operates in continuous time (dt), dealing with delays, rhythms, and synchronization.
2.  **Spikes over Floats:** Information is carried by sparse events (spikes) and energy waves, drastically reducing computational waste.
3.  **Homeostasis:** The system self-regulates. If it's too excited, it calms down; if it's too silent, it lowers its threshold. It strives for stability.
4.  **Emergence:** Behaviors like memory recall, attention, and decision-making emerge from the interaction of local cortical fields and global workspaces.

---

## Architecture Overview

Mergen implements a **Hybrid Spiking Cognitive Architecture**:

### 1. Cortical Sheets (The Tissue)
Instead of dense layers, Mergen uses 2D topographic maps of neurons.
*   **Mexican Hat Kernels:** Neurons interact via local excitation and lateral inhibition, creating "bubbles of attention" and traveling waves.
*   **FFT Accelerated:** Using Fast Fourier Transforms to simulate millions of local connections efficiently.

### 2. The Global Workspace (The Consciousness)
Based on the **Global Neuronal Workspace Theory (GNWT)**.
*   Local cortical areas (Sensory, Motor) compete for access to a central "router."
*   Only strong, relevant signals ignite the workspace and are broadcasted back to the entire brain.
*   This creates a unified "moment of thought."

### 3. Hippocampus (Fast Episodic Memory)
*   **One-Shot Learning:** Unlike backpropagation which takes epochs, the Hippocampus captures snapshots of cortical activity instantly.
*   **Pattern Completion:** It can retrieve a full memory from a partial cue (associative recall) using high-speed vector similarity.

### 4. Sparse Direct Pathways
*   Like the brain's white matter tracts, Mergen uses long-range sparse connections to bypass local processing for fast reflexes and learned habits.

---

## ⚡ Key Differentiators

| Feature | Standard Transformer | **MERGEN Engine** |
| :--- | :--- | :--- |
| **Communication** | Dense Matrix Multiplication | **Sparse Spikes & Local Fields** |
| **Time** | Discrete Steps (Tokens) | **Continuous Flow (dt)** |
| **Memory** | Context Window (Limited) | **Episodic Store (Associative)** |
| **Learning** | Backpropagation Only | **Local Plasticity + Hebbian + RL** |
| **State** | Stateless (Reset per prompt) | **Persistent Dynamic State** |

---

## 🚀 Getting Started

Mergen is a research framework. It is currently set up to demonstrate **symbolic learning** through biological dynamics.

### Installation

```bash
git clone https://github.com/VertexCorporation/Mergen.git
cd Mergen
pip install -r requirements.txt
```

### Running the Engine

To start the training loop where Mergen learns simple arithmetic concepts via spiking dynamics:

```bash
python examples/teach_math.py
```

*Note: You will see telemetry logs showing Sensory/Motor firing rates, Workspace activation levels, and memory retrieval statuses.*

---

## 🛡️ License

This project is licensed under the **Apache License 2.0**.
Mergen is an open conceptual contribution to the AGI research community.

---

<p align="center">
  <i>"Ben Mergen'im." (I am Mergen.)</i>
</p>
```