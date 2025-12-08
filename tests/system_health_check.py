# tests/system_health_check.py
import sys
import os
import torch
import time
import numpy as np

# Proje kök dizinini ekle
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import config as cfg
from anatomy.cortical_sheet import CorticalLayer
from anatomy.hippocampus import Hippocampus
from connectivity.global_workspace import GlobalWorkspace
from connectivity.kernels import MexicanHatKernel
from utils.telemetry import TelemetryBox

def run_diagnostics(steps=500):
    print("🩺 STARTING SYSTEM DIAGNOSTICS (MERGEN V3)...")
    
    # 1. INIT
    device = cfg.DEVICE
    print(f"⚙️  Hardware: {device}")
    
    try:
        kernel = MexicanHatKernel.create(cfg.H_CORTEX, cfg.W_CORTEX, device=device)
        sensory = CorticalLayer(cfg.H_CORTEX, cfg.W_CORTEX, cfg.DT, kernel, device=device)
        motor = CorticalLayer(cfg.H_CORTEX, cfg.W_CORTEX, cfg.DT, kernel, device=device)
        workspace = GlobalWorkspace(cfg.N_NEURONS, cfg.N_BUNDLES, cfg.N_NEURONS, device=device)
        
        print("✅ Anatomy Init Success")
    except Exception as e:
        print(f"❌ CRITICAL ERROR during Init: {e}")
        return

    # 2. TELEMETRY INIT
    telemetry = TelemetryBox(run_name="diagnostic_run_01")
    
    # 3. STRESS TEST LOOP
    print(f"🚀 Running {steps} simulation steps...")
    
    input_noise = torch.randn((steps, cfg.H_CORTEX, cfg.W_CORTEX), device=device) * 10.0
    
    start_global = time.time()
    
    for t in range(steps):
        step_start = time.time()
        
        # A. Sensory Dynamics
        spikes_s = sensory(input_noise[t])
        
        # B. Global Workspace
        gw_out = workspace(spikes_s.flatten().unsqueeze(0)).view(cfg.H_CORTEX, cfg.W_CORTEX)
        
        # C. Motor Dynamics
        spikes_m = motor(gw_out * cfg.BUNDLE_GAIN) 
        
        step_end = time.time()
        duration = step_end - step_start
        
        # D. Log Data
        telemetry.log_step(
            step=t, 
            sensory=spikes_s, 
            motor=spikes_m, 
            workspace=workspace.state, # State activation
            acc=0.0, # Dummy acc
            mem_hit=False, 
            duration=duration
        )
        
        # E. Take Snapshot every 50 steps
        if t % 50 == 0:
            telemetry.take_snapshot(t, spikes_s, spikes_m)
            # Canlılık belirtisi
            print(f"   Step {t}/{steps} | Sensory Rate: {spikes_s.mean():.4f} | Motor Rate: {spikes_m.mean():.4f} | Time: {duration*1000:.2f}ms")

    total_time = time.time() - start_global
    print(f"✅ Diagnostics Complete in {total_time:.2f}s")
    print(f"⚡ Average FPS: {steps/total_time:.1f}")
    
    # Save Data
    telemetry.save_report()

if __name__ == "__main__":
    run_diagnostics()