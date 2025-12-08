# utils/telemetry.py
import torch
import numpy as np
import time
import os

class TelemetryBox:
    def __init__(self, run_name="test_run"):
        self.history = {
            'step': [],
            'sensory_activity': [], # Ortalama ateşleme hızı
            'motor_activity': [],
            'workspace_activity': [],
            'accuracy': [],
            'memory_hits': [], # Hafıza kullanıldı mı?
            'computation_time': []
        }
        self.snapshots = {} # Detaylı ısı haritaları için
        self.run_name = run_name
        self.start_time = time.time()

    def log_step(self, step, sensory, motor, workspace, acc, mem_hit, duration):
        # GPU'dan CPU'ya sadece "Scalar" (tek sayı) çekiyoruz, bu hızlıdır.
        self.history['step'].append(step)
        self.history['sensory_activity'].append(sensory.float().mean().item())
        self.history['motor_activity'].append(motor.float().mean().item())
        self.history['workspace_activity'].append(workspace.float().mean().item())
        self.history['accuracy'].append(acc)
        self.history['memory_hits'].append(1 if mem_hit else 0)
        self.history['computation_time'].append(duration)

    def take_snapshot(self, step, sensory_spikes, motor_spikes):
        # Her adımda değil, sadece belirli aralıklarda tam resim alıyoruz
        # Veriyi sıkıştırıp (Downsample) saklayalım yoksa disk dolar.
        # 100x100 -> 50x50 gibi
        s_img = torch.nn.functional.interpolate(sensory_spikes.unsqueeze(0).unsqueeze(0), scale_factor=0.5).squeeze().cpu().numpy()
        m_img = torch.nn.functional.interpolate(motor_spikes.unsqueeze(0).unsqueeze(0), scale_factor=0.5).squeeze().cpu().numpy()
        
        self.snapshots[step] = {'sensory': s_img, 'motor': m_img}

    def save_report(self, output_dir="logs"):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        filename = f"{output_dir}/{self.run_name}_data.npz"
        np.savez(filename, history=self.history, snapshots=self.snapshots)
        print(f"📊 Telemetry saved to {filename}")