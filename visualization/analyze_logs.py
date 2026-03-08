# visualization/analyze_logs.py
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

def analyze(run_name="diagnostic_run_01"):
    path = f"logs/{run_name}_data.npz"
    if not os.path.exists(path):
        print(f"❌ Log file not found: {path}")
        return

    print(f"📂 Loading telemetry: {path}")
    data = np.load(path, allow_pickle=True)
    history = data['history'].item()
    snapshots = data['snapshots'].item()
    
    # --- PLOT 1: ACTIVITY RATES ---
    fig, axs = plt.subplots(3, 1, figsize=(12, 12))
    
    # Panel 1: Firing Rates (Epilepsy Check)
    axs[0].plot(history['sensory_activity'], label='Sensory', color='orange', alpha=0.7)
    axs[0].plot(history['motor_activity'], label='Motor', color='cyan', alpha=0.7)
    axs[0].set_title("Neuron Firing Rates")
    axs[0].set_ylabel("Active Neuron Ratio (0.0 - 1.0)")
    axs[0].axhline(y=0.05, color='r', linestyle='--', label='Target (Homeostasis)')
    axs[0].legend()
    axs[0].grid(True, alpha=0.2)
    
    # Panel 2: Global Workspace Activation
    axs[1].plot(history['workspace_activity'], color='magenta')
    axs[1].set_title("Global Workspace Activation")
    axs[1].set_ylabel("Activity")
    axs[1].grid(True, alpha=0.2)
    
    # Panel 3: Performance Latency
    axs[2].plot(history['computation_time'], color='lime')
    axs[2].set_title("Computation Time (ms)")
    axs[2].set_ylabel("Seconds")
    axs[2].set_xlabel("Step")
    axs[2].grid(True, alpha=0.2)
    
    plt.tight_layout()
    plt.savefig(f"logs/{run_name}_report.png")
    print(f"🖼️  Report saved to logs/{run_name}_report.png")
    plt.show()

    # --- PLOT 2: SNAPSHOTS (HEATMAPS) ---
    # Plot the first and last snapshots
    if len(snapshots) > 0:
        steps = sorted(snapshots.keys())
        first_step = steps[0]
        last_step = steps[-1]
        
        fig2, axes = plt.subplots(2, 2, figsize=(10, 10))
        
        axes[0,0].imshow(snapshots[first_step]['sensory'], cmap='inferno', vmin=0, vmax=1)
        axes[0,0].set_title(f"Sensory (Step {first_step})")
        
        axes[0,1].imshow(snapshots[first_step]['motor'], cmap='viridis', vmin=0, vmax=1)
        axes[0,1].set_title(f"Motor (Step {first_step})")
        
        axes[1,0].imshow(snapshots[last_step]['sensory'], cmap='inferno', vmin=0, vmax=1)
        axes[1,0].set_title(f"Sensory (Step {last_step})")
        
        axes[1,1].imshow(snapshots[last_step]['motor'], cmap='viridis', vmin=0, vmax=1)
        axes[1,1].set_title(f"Motor (Step {last_step})")
        
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    analyze()