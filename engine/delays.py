import torch

class DelayBuffer:
    """
    A Ring-Buffer mechanism to simulate axonal transmission delays.
    
    In the brain, a signal from the visual cortex takes ~10ms to reach the frontal lobe.
    This class manages that history.
    """
    def __init__(self, n_neurons: int, max_delay_steps: int, device='cpu'):
        """
        Args:
            n_neurons: Total number of neurons in the layer.
            max_delay_steps: The maximum memory depth (in time steps).
                             If dt=1ms and max delay=100ms, this is 100.
        """
        self.max_delay = max_delay_steps
        self.n_neurons = n_neurons
        self.device = device
        
        # The Buffer: [Time, Neurons]
        # We initialized it with zeros (silence).
        self.buffer = torch.zeros((max_delay_steps, n_neurons), device=device)
        
        # Pointer to the "Current Time" inside the buffer
        self.cursor = 0

    def write(self, current_activity: torch.Tensor):
        """
        Records the activity happening NOW into the buffer.
        Overwrites the oldest data (Ring structure).
        """
        self.buffer[self.cursor] = current_activity

    def read(self, delay_indices: torch.Tensor) -> torch.Tensor:
        """
        Retrieves signals from the past.
        
        Args:
            delay_indices: A tensor of shape (n_neurons,) where each value
                           is how many steps back we should look for that neuron.
                           
        Returns:
            The delayed signal for each neuron.
        """
        # Calculate the past position: (Current - Delay) % Max_Size
        # The modulo (%) operator creates the "Ring" effect.
        read_indices = (self.cursor - delay_indices) % self.max_delay
        
        # We need to gather the specific time-slices for specific neurons.
        # This uses advanced PyTorch indexing.
        # buffer[:, i] is the history of neuron i.
        # We want buffer[read_indices[i], i].
        
        neuron_indices = torch.arange(self.n_neurons, device=self.device)
        delayed_signal = self.buffer[read_indices, neuron_indices]
        
        return delayed_signal

    def tick(self):
        """Moves the clock forward by 1 step."""
        self.cursor = (self.cursor + 1) % self.max_delay