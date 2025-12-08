# config.py
import torch

# --- 1. HARDWARE ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32

# --- 2. ANATOMY SCALE ---
H_CORTEX = 320   
W_CORTEX = 320  
N_NEURONS = H_CORTEX * W_CORTEX  # ~102k Nöron
# Chat için kelime haznesi kadar sınıf lazım, şimdilik 50 diyelim
N_CLASSES = 50  

# --- 3. MEMORY SYSTEM ---
MEMORY_CAPACITY = 5000  
SIMILARITY_THRESHOLD = 0.85 # Biraz daha esnek olsun, hemen hatırlasın

# --- 4. PHYSICS (TIME) ---
DT = 1.0           
TAU_MEM = 5.0      # Çok hızlı tepki
TAU_ADAPT = 50.0   # Çabuk toparlanma
TAU_SYN = 5.0      

# --- 5. CONNECTIVITY ---
EXC_SIGMA = 2.0    
INH_SIGMA = 5.0    
EXC_GAIN = 50.0    # 10 -> 50 (Sinyali patlat!)
INH_GAIN = 5.0     

# --- 6. GLOBAL WORKSPACE ---
N_BUNDLES = 256    
BUNDLE_GAIN = 200.0 # Global yayın çok güçlü olmalı

# --- 7. DIRECT PATHWAY (SPARSE) ---
SYNAPSES_PER_NEURON = 100 # VRAM dostu

# --- 8. NEURON DYNAMICS ---
THETA_BASE = 0.05  # Eşik çok düşük, hemen ateşlesin
GLOBAL_INH = 0.0   
K_ADAPT = 1.0      
V_RESET = 0.0      
THETA_DECAY = 0.95 

# --- 9. LEARNING ---
LEARNING_RATE_CORTEX = 0.1  
LEARNING_RATE_MEMORY = 1.0   

# --- 10. STABILITY ---
TARGET_RATE = 0.05 
HOMEOSTATIC_LR = 0.01