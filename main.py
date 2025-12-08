# main.py

"""
MERGEN V3 - TRAINING LOOP (HEADLESS + TELEMETRY)

This script wires together the full MERGEN V3 cognitive architecture and
runs an open-ended training loop on a simple symbolic math task
(e.g., "3 + 5" → 8).

Key components used here:
- CorticalLayer (sensory + motor): 2D spiking cortical sheets
- GlobalWorkspace: low-rank global routing / attention
- Sparse Direct Pathway: fast point-to-point motor drive
- Hippocampus: fast episodic key–value memory
- HomeostaticRegulator: keeps firing rates near a target regime
- SpikeEncoder: turns text into spatiotemporal input currents
- MathTeacher: generates arithmetic questions and labels
- TelemetryBox: logs activity, performance, and snapshots for analysis

All plotting is disabled in this loop. Instead, telemetry is written to
`logs/` and can be analyzed offline via `visualization/analyze_logs.py`.
"""

import os
import time
from typing import Tuple, Optional

import numpy as np
import torch

import config as cfg
from anatomy.cortical_sheet import CorticalLayer
from anatomy.hippocampus import Hippocampus
from connectivity.kernels import MexicanHatKernel
from connectivity.global_workspace import GlobalWorkspace
from utils.encoder import SpikeEncoder
from utils.stability import HomeostaticRegulator
from utils.telemetry import TelemetryBox
from datasets.generators.math_teacher import MathTeacher


class MergenCognitiveArchitecture:
    """
    High-level container for the MERGEN cognitive engine.

    This class:
    - Instantiates all major brain components.
    - Holds long-lived state traces (sensory + motor).
    - Implements a single forward step and learning update.
    - Handles saving/loading of the "brain state" (weights + memory).
    """

    def __init__(self) -> None:
        print(f"🧠 Assembling MERGEN V3 on device: {cfg.DEVICE}")
        print(f"📐 Cortex resolution: {cfg.H_CORTEX} x {cfg.W_CORTEX} "
              f"({cfg.N_NEURONS} neurons per layer)")

        # 1. CORTICAL LAYERS (Sensory + Motor)
        kernel_t = MexicanHatKernel.create(
            cfg.H_CORTEX,
            cfg.W_CORTEX,
            exc_sigma=cfg.EXC_SIGMA,
            inh_sigma=cfg.INH_SIGMA,
            exc_gain=cfg.EXC_GAIN,
            inh_gain=cfg.INH_GAIN,
            device=cfg.DEVICE,
        )

        self.sensory = CorticalLayer(
            cfg.H_CORTEX,
            cfg.W_CORTEX,
            cfg.DT,
            kernel_t,
            device=cfg.DEVICE,
        )

        self.motor = CorticalLayer(
            cfg.H_CORTEX,
            cfg.W_CORTEX,
            cfg.DT,
            kernel_t,
            device=cfg.DEVICE,
        )

        # 2. GLOBAL WORKSPACE
        self.workspace = GlobalWorkspace(
            input_dim=cfg.N_NEURONS,
            workspace_dim=cfg.N_BUNDLES,
            output_dim=cfg.N_NEURONS,
            device=cfg.DEVICE,
        )

        # 3. DIRECT SPARSE PATHWAY (Fast sensorimotor shortcut)
        print("🔗 Building sparse direct pathway...")
        n_synapses = cfg.N_NEURONS * cfg.SYNAPSES_PER_NEURON

        # Each neuron gets SYNAPSES_PER_NEURON outgoing synapses
        rows = torch.arange(
            cfg.N_NEURONS,
            device=cfg.DEVICE,
        ).repeat_interleave(cfg.SYNAPSES_PER_NEURON)

        # Random destinations for each synapse
        cols = torch.randint(
            0,
            cfg.N_NEURONS,
            (n_synapses,),
            device=cfg.DEVICE,
        )

        # (2, n_synapses) index tensor for a sparse COO matrix
        self.sm_indices = torch.stack([rows, cols])

        # Initial synaptic weights for the sparse pathway
        self.sm_values = torch.randn(
            n_synapses,
            device=cfg.DEVICE,
        ) * 0.5

        # 4. HIPPOCAMPUS (Episodic memory: sensory trace → class vector)
        self.hippocampus = Hippocampus(
            input_dim=cfg.N_NEURONS,
            value_dim=cfg.N_CLASSES,
            memory_capacity=cfg.MEMORY_CAPACITY,
            threshold=cfg.SIMILARITY_THRESHOLD,
            device=cfg.DEVICE,
        )

        # 5. READOUT & STABILITY
        # Motor trace → class logits
        self.W_readout = torch.randn(
            (cfg.N_NEURONS, cfg.N_CLASSES),
            device=cfg.DEVICE,
        ) * 0.1

        # Global homeostatic regulator (shared threshold tuning)
        self.regulator = HomeostaticRegulator(
            n_neurons=cfg.N_NEURONS,
            target_rate=cfg.TARGET_RATE,
            adjustment_speed=cfg.HOMEOSTATIC_LR,
            device=cfg.DEVICE,
        )

        # 6. TRACES (slow state summaries over time)
        self.trace_sensory = torch.zeros(cfg.N_NEURONS, device=cfg.DEVICE)
        self.trace_motor = torch.zeros(cfg.N_NEURONS, device=cfg.DEVICE)

    # --------------------------------------------------------------------- #
    # STATE MANAGEMENT
    # --------------------------------------------------------------------- #

    def reset_state(self) -> None:
        """
        Clears fast dynamical state between episodes.

        This acts a bit like a "sleep reset":
        - Clears voltages and spikes in both cortical layers.
        - Clears workspace activation.
        - Resets trace vectors.
        """
        self.sensory.v.fill_(cfg.V_RESET)
        self.motor.v.fill_(cfg.V_RESET)

        self.sensory.spikes.zero_()
        self.motor.spikes.zero_()

        self.trace_sensory.zero_()
        self.trace_motor.zero_()

        self.workspace.state.zero_()

    # --------------------------------------------------------------------- #
    # FORWARD DYNAMICS
    # --------------------------------------------------------------------- #

    def forward_pass(
        self,
        input_frame: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], float]:
        """
        Single simulation step.

        Pipeline:
        1. Feed input into sensory cortex.
        2. Update sensory trace.
        3. Query hippocampus with sensory trace.
        4. Propagate through global workspace.
        5. Add sparse direct pathway drive.
        6. Update motor cortex + motor trace.
        7. Occasionally apply homeostatic threshold updates.

        Args:
            input_frame: Tensor of shape (N_NEURONS,) or (H, W).

        Returns:
            mem_vec: Retrieved value vector from hippocampus, or None.
            mem_conf: Similarity score for the best memory match.
        """
        # Ensure 2D shape
        input_2d = input_frame.view(cfg.H_CORTEX, cfg.W_CORTEX)

        # 1) Sensory cortex dynamics
        spikes_s = self.sensory(input_2d * 2.0)  # input gain = 2.0
        self.trace_sensory = (
            0.95 * self.trace_sensory + 0.05 * spikes_s.flatten()
        )

        # 2) Episodic memory query
        mem_vec, mem_conf = self.hippocampus.retrieve(self.trace_sensory)

        # 3) Global Workspace broadcast
        gw_out = self.workspace(spikes_s.flatten().unsqueeze(0))
        gw_drive = gw_out.view(cfg.H_CORTEX, cfg.W_CORTEX) * cfg.BUNDLE_GAIN

        # 4) Sparse direct sensorimotor pathway (O(#synapses), no big sparse matrix)
        if spikes_s.sum() > 0:
            # Flatten sensory spikes
            flat_spikes = spikes_s.flatten()

            # sm_indices: (2, n_synapses) -> rows = pre, cols = post
            rows, cols = self.sm_indices

            # Pre-synaptic activity at each synapse
            pre_act = flat_spikes[rows]  # (n_synapses,)

            # Contribution of each synapse: pre_act * weight
            contrib = pre_act * self.sm_values  # (n_synapses,)

            # Accumulate into post-synaptic neurons
            direct_signal = torch.zeros(cfg.N_NEURONS, device=cfg.DEVICE)
            direct_signal.index_add_(0, cols, contrib)

            direct_drive = direct_signal.view(
                cfg.H_CORTEX,
                cfg.W_CORTEX,
            ) * 3.0  # direct gain
        else:
            direct_drive = torch.zeros(
                (cfg.H_CORTEX, cfg.W_CORTEX),
                device=cfg.DEVICE,
            )

        # 5) Motor cortex dynamics
        spikes_m = self.motor(gw_drive + direct_drive)
        self.trace_motor = (
            0.95 * self.trace_motor + 0.05 * spikes_m.flatten()
        )

        # 6) Homeostasis (sampled update to avoid overhead)
        if np.random.random() < 0.01:
            # Flatten thresholds, update, then reshape to 2D
            new_theta_sens = self.regulator.update(
                current_spikes=spikes_s.flatten(),
                current_thresholds=self.sensory.theta.flatten(),
            ).view(cfg.H_CORTEX, cfg.W_CORTEX)

            new_theta_motor = self.regulator.update(
                current_spikes=spikes_m.flatten(),
                current_thresholds=self.motor.theta.flatten(),
            ).view(cfg.H_CORTEX, cfg.W_CORTEX)

            self.sensory.theta = new_theta_sens
            self.motor.theta = new_theta_motor

        return mem_vec, float(mem_conf)

    # --------------------------------------------------------------------- #
    # READOUT + LEARNING
    # --------------------------------------------------------------------- #

    def predict(self) -> torch.Tensor:
        """
        Computes class probabilities from the current motor trace.

        Returns:
            probs: Tensor of shape (N_CLASSES,) (softmax over logits).
        """
        logits = torch.matmul(self.trace_motor, self.W_readout)
        return torch.softmax(logits, dim=0)

    def learn(self, target_class: int, use_memory: bool = True) -> None:
        """
        Updates:
        - Readout weights (motor_trace → class).
        - Sparse direct pathway weights (Hebbian style).
        - Episodic memory (sensory trace → one-hot class).

        Args:
            target_class: Correct class index in [0, N_CLASSES).
            use_memory: If True, store the episode in hippocampus.
        """
        # --- 1) Readout learning (simple delta rule) ---
        target_vec = torch.zeros(cfg.N_CLASSES, device=cfg.DEVICE)
        target_vec[target_class] = 1.0

        pred = self.predict()
        error = target_vec - pred

        self.W_readout += cfg.LEARNING_RATE_CORTEX * torch.outer(
            self.trace_motor,
            error,
        )

        # --- 2) Sparse Hebbian update on direct pathway ---
        target_motor = self.W_readout[:, target_class]
        motor_error = target_motor - self.trace_motor

        rows, cols = self.sm_indices
        hebbian_update = self.trace_sensory[rows] * motor_error[cols]
        self.sm_values += 0.05 * hebbian_update  # fixed local learning rate

        # --- 3) Episodic storage ---
        if use_memory:
            self.hippocampus.store(self.trace_sensory, target_vec)

    # --------------------------------------------------------------------- #
    # PERSISTENCE
    # --------------------------------------------------------------------- #

    def save_brain(self, filepath: str = "mergen.mx") -> None:
        """
        Saves core brain parameters (readout, sparse pathway, memory).
        """
        state = {
            "W_readout": self.W_readout,
            "sm_values": self.sm_values,
            "hippo_keys": self.hippocampus.keys,
            "hippo_vals": self.hippocampus.values,
            "hippo_size": self.hippocampus.size,
        }
        torch.save(state, filepath)
        print(f"💾 Brain snapshot saved: {filepath}")

    def load_brain(self, filepath: str = "mergen.mx") -> None:
        """
        Loads brain parameters if a snapshot exists.
        """
        if not os.path.exists(filepath):
            print("ℹ️ No existing brain snapshot found. Starting fresh.")
            return

        print(f"📂 Loading brain snapshot from: {filepath} ...")
        s = torch.load(filepath, map_location=cfg.DEVICE)

        if "W_readout" in s and s["W_readout"].shape == self.W_readout.shape:
            self.W_readout = s["W_readout"].to(cfg.DEVICE)
            self.sm_values = s["sm_values"].to(cfg.DEVICE)
            self.hippocampus.keys = s["hippo_keys"].to(cfg.DEVICE)
            self.hippocampus.values = s["hippo_vals"].to(cfg.DEVICE)
            self.hippocampus.size = int(s["hippo_size"])
            print("✅ Brain state successfully restored.")
        else:
            print("⚠️ Snapshot shape mismatch. Starting from scratch.")


# ------------------------------------------------------------------------- #
# TRAINING LOOP
# ------------------------------------------------------------------------- #


def run_training() -> None:
    """
    Main training loop.

    At each episode:
    1. Reset fast state.
    2. Sample a math problem from the teacher.
    3. Encode text to spike-like input currents.
    4. Run the cortical+workspace+memory dynamics over time.
    5. Decide the answer (hippocampus if confident, else cortex).
    6. Apply learning updates.
    7. Log telemetry (activity, accuracy, memory usage, latency).

    Training continues until interrupted (Ctrl+C).
    """
    # Disable autograd everywhere (we use local plasticity, no backprop)
    torch.set_grad_enabled(False)

    brain = MergenCognitiveArchitecture()
    brain.load_brain()

    encoder = SpikeEncoder(cfg.N_NEURONS, device=cfg.DEVICE)
    teacher = MathTeacher(cfg.N_NEURONS)

    telemetry = TelemetryBox(run_name="training_session_v3")

    print("\n⚡ MERGEN V3 TRAINING STARTED (log-only mode)")
    print("   Press Ctrl+C to stop and save.\n")

    episode = 0
    history = []

    try:
        while True:
            episode += 1
            brain.reset_state()
            ep_start = time.time()

            # 1) Sample a problem
            _, target_class, problem_text = teacher.generate_sample()

            # 2) Encode as input stream (Time x N_NEURONS)
            input_stream = encoder.encode_text(
                text=problem_text,
                duration_steps=500,  # shorter episodes for faster training
            )
            T = len(input_stream)
            learn_phase = int(T * 0.9)

            final_pred = -1
            mem_active = False

            # 3) Simulate episode
            for t in range(T):
                mem_vec, mem_conf = brain.forward_pass(input_stream[t])

                # Decision point (just before learning window)
                if t == learn_phase - 1:
                    cortical_probs = brain.predict()

                    use_memory = mem_vec is not None 

                    if use_memory:
                        final_pred = torch.argmax(mem_vec).item()
                        mem_active = True
                    else:
                        final_pred = torch.argmax(cortical_probs).item()

                    last_cortical_pred = torch.argmax(cortical_probs).item()
                    last_memory_pred = torch.argmax(mem_vec).item() if mem_vec is not None else None

                # Learning window (end of episode)
                if t >= learn_phase:
                    brain.learn(target_class)

            # 4) Episode statistics
            is_correct = (final_pred == target_class)
            history.append(1 if is_correct else 0)
            if len(history) > 100:
                history.pop(0)

            avg_acc = float(np.mean(history) * 100.0)
            ep_duration = time.time() - ep_start

            # 5) Telemetry logging (only final state of episode)
            telemetry.log_step(
                step=episode,
                sensory=brain.sensory.spikes,
                motor=brain.motor.spikes,
                workspace=brain.workspace.state,
                acc=avg_acc,
                mem_hit=mem_active,
                duration=ep_duration,
            )

            # 6) Periodic snapshot + console status
            if episode <= 5 or episode % 50 == 0:
                telemetry.take_snapshot(
                    episode,
                    brain.sensory.spikes,
                    brain.motor.spikes,
                )
                brain.save_brain()

                sensory_rate = brain.sensory.spikes.float().mean().item()
                motor_rate = brain.motor.spikes.float().mean().item()
                workspace_level = brain.workspace.state.float().mean().item()

                status = "✅" if is_correct else "❌"
                src = "MEM" if mem_active else "CTX"

                print(
                    "\n"
                    f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                    f"EPISODE {episode}\n"
                    f"Question: {problem_text}\n"
                    f"True Answer: {target_class}\n"
                    f"Cortex Prediction: {last_cortical_pred}\n"
                    f"Memory Prediction: {last_memory_pred if last_memory_pred is not None else '—'}\n"
                    f"Final Decision: {final_pred} ({src}) {status}\n"
                    f"Accuracy (last 100 ep): {avg_acc:.2f}%\n"
                    f"Sensory Rate: {sensory_rate:.4f}\n"
                    f"Motor Rate: {motor_rate:.4f}\n"
                    f"Workspace Activation: {workspace_level:.4f}\n"
                    f"⏱Episode Time: {ep_duration:.4f} s\n"
                    f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n",
                    flush=True,
                )

    except KeyboardInterrupt:
        print("\n🛑 Training interrupted by user.")
        brain.save_brain()
        telemetry.save_report()
        print("📊 Telemetry saved. You can inspect it with:")
        print("   python -m visualization.analyze_logs training_session_v3")


if __name__ == "__main__":
    run_training()