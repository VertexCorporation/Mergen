"""
MERGEN V3 - TRAINING LOOP
The "School" where the Cognitive Engine learns.
"""

import torch
import numpy as np
import time
from engine.integrators import EulerSolver
from anatomy.cortical_sheet import CorticalLayer
from anatomy.hippocampus import Hippocampus
from connectivity.kernels import MexicanHatKernel
from connectivity.global_workspace import GlobalWorkspace
from utils.encoder import SpikeEncoder
from datasets.generators.math_teacher import MathTeacher
import config as cfg

def main():
    print(f"🚀 Initializing MERGEN V3 (PyTorch + GPU: {cfg.DEVICE})...")
    print(f"🧠 Cortex Size: {cfg.N_NEURONS} Neurons x 2 Layers")
    
    # --- 1. BUILD THE BRAIN (ANATOMY) ---
    # A. Sensory Cortex (L4)
    kernel_tensor = MexicanHatKernel.create(cfg.H_CORTEX, cfg.W_CORTEX, device=cfg.DEVICE)
    sensory_cortex = CorticalLayer(cfg.H_CORTEX, cfg.W_CORTEX, cfg.DT, kernel_tensor, device=cfg.DEVICE)
    
    # B. Motor Cortex (L5/6)
    motor_cortex = CorticalLayer(cfg.H_CORTEX, cfg.W_CORTEX, cfg.DT, kernel_tensor, device=cfg.DEVICE)
    
    # C. Global Workspace (The Router)
    workspace = GlobalWorkspace(cfg.N_NEURONS, cfg.N_BUNDLES, cfg.N_NEURONS, device=cfg.DEVICE)
    
    # D. Hippocampus (Fast Memory)
    hippocampus = Hippocampus(cfg.N_NEURONS, cfg.MEMORY_CAPACITY, cfg.SIMILARITY_THRESHOLD, device=cfg.DEVICE)
    
    # E. Tools
    encoder = SpikeEncoder(cfg.N_NEURONS, device=cfg.DEVICE)
    teacher = MathTeacher(cfg.N_NEURONS, duration_ms=1000) # Re-use your generator
    
    # Readout Weights (Motor -> Output Class)
    # Simple linear readout for now
    W_readout = torch.zeros((cfg.N_NEURONS, 20), device=cfg.DEVICE, requires_grad=False)
    
    print("✅ Brain Assembly Complete.")
    print("\n⚡ Starting Training Loop...")
    print("-----------------------------------")
    
    episode = 0
    recent_acc = []
    
    try:
        while True:
            episode += 1
            
            # --- EPISODE START: RESET ---
            # Clear voltages (Sleep)
            sensory_cortex.v.fill_(cfg.V_RESET)
            motor_cortex.v.fill_(cfg.V_RESET)
            # Clear traces is CRITICAL to prevent ghosting
            sensory_trace = torch.zeros(cfg.N_NEURONS, device=cfg.DEVICE)
            motor_trace = torch.zeros(cfg.N_NEURONS, device=cfg.DEVICE)
            
            # --- GET TASK ---
            # Teacher gives: "3 + 5", Answer: 8
            # Note: MathTeacher currently returns numpy, we might need to adapt it slightly
            # for now let's assume it gives text and int target.
            _, target_class, problem_text = teacher.generate_sample()
            
            # Convert text to PyTorch input signal
            input_signal = encoder.encode_text(problem_text, duration_steps=1000)
            
            # --- SIMULATION LOOP ---
            used_memory = False
            final_probs = None
            
            # Learn at the end (Consolidation phase)
            learn_start = int(len(input_signal) * 0.9)
            
            for t in range(len(input_signal)):
                # 1. Input (Sensory L4)
                inp_t = input_signal[t].view(cfg.H_CORTEX, cfg.W_CORTEX)
                spikes_s = sensory_cortex(inp_t)
                
                # Update Sensory Trace
                sensory_trace = (1 - 0.05) * sensory_trace + 0.05 * spikes_s.flatten()
                
                # 2. Hippocampal Query
                # "Have I seen this sensory pattern before?"
                mem_val, mem_sim = hippocampus.retrieve(sensory_trace)
                
                # 3. Global Workspace (Broadcast)
                # Compresses Sensory info and broadcasts to Motor
                gw_feedback = workspace(spikes_s.flatten().unsqueeze(0)).view(cfg.H_CORTEX, cfg.W_CORTEX)
                
                # 4. Motor Output (L5/6)
                # Input = GW_Feedback + Memory_Injection
                motor_drive = gw_feedback * cfg.BUNDLE_GAIN
                spikes_m = motor_cortex(motor_drive)
                
                # Update Motor Trace
                motor_trace = (1 - 0.05) * motor_trace + 0.05 * spikes_m.flatten()
                
                # 5. Decision & Learning (At end of episode)
                if t == learn_start:
                    # Prediction: Readout from Motor Trace
                    logits = torch.matmul(motor_trace, W_readout)
                    cortical_probs = torch.softmax(logits, dim=0)
                    
                    # Integration: If memory is strong, trust it
                    if mem_val is not None:
                        # Memory returns a 'value pattern' (target class vector)
                        # We simulate this as a confidence boost
                        used_memory = True
                        final_probs = mem_val # Trust memory 100%
                    else:
                        final_probs = cortical_probs
                    
                    # --- LEARNING (PLASTICITY) ---
                    # 1. Update Readout (Slow Cortex)
                    target_vec = torch.zeros(20, device=cfg.DEVICE)
                    target_vec[target_class] = 1.0
                    error = target_vec - final_probs
                    
                    # dW = lr * input * error
                    delta = cfg.LEARNING_RATE_CORTEX * torch.outer(motor_trace, error)
                    W_readout += delta
                    
                    # 2. Update Hippocampus (Fast One-Shot)
                    # Store Sensory_Pattern -> Target_Class_Vector
                    hippocampus.store(sensory_trace, target_vec)

            # --- EVALUATION ---
            pred_class = torch.argmax(final_probs).item()
            is_correct = (pred_class == target_class)
            
            recent_acc.append(1 if is_correct else 0)
            if len(recent_acc) > 50: recent_acc.pop(0)
            avg = np.mean(recent_acc) * 100
            
            status = "✅" if is_correct else "❌"
            mem_str = "🧠(Mem)" if used_memory else "🔹(New)"
            
            print(f"Ep {episode:03d} | Q: {problem_text} | Tgt: {target_class} | Pred: {pred_class} | Acc: {avg:.1f}% | {status} {mem_str}")
            
            if episode % 25 == 0:
                print("-" * 60)

    except KeyboardInterrupt:
        print("\n🛑 Training stopped.")

if __name__ == "__main__":
    main()