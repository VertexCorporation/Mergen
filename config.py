# config.py
import torch

# --- 1. HARDWARE ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32

# --- 2. ANATOMY SCALE ---
H_CORTEX = 320   
W_CORTEX = 320  
N_NEURONS = H_CORTEX * W_CORTEX  # ~102k Neurons
# Need as many classes as vocabulary for chat, let's say 50 for now
N_CLASSES = 50  

# --- 3. MEMORY SYSTEM ---
MEMORY_CAPACITY = 5000  
SIMILARITY_THRESHOLD = 0.85 # Make it a bit more flexible, so it remembers quickly

# --- 4. PHYSICS (TIME) ---
DT = 1.0           
TAU_MEM = 5.0      # Very fast response
TAU_ADAPT = 50.0   # Quick recovery
TAU_SYN = 5.0      

# --- 5. CONNECTIVITY ---
EXC_SIGMA = 2.0    
INH_SIGMA = 5.0    
EXC_GAIN = 50.0    # 10 -> 50 (Amplify the signal!)
INH_GAIN = 5.0     

# --- 6. GLOBAL WORKSPACE ---
N_BUNDLES = 256    
BUNDLE_GAIN = 200.0 # Global broadcast should be very strong

# --- 7. DIRECT PATHWAY (SPARSE) ---
SYNAPSES_PER_NEURON = 100 # VRAM-friendly

# --- 8. NEURON DYNAMICS ---
THETA_BASE = 0.05  # Threshold very low, so it fires immediately
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