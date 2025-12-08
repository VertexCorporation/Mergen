"""
MERGEN V3 - MATH TEACHER
Generates symbolic arithmetic problems.

Unlike V1/V2, this module DOES NOT generate signals.
It only generates the abstract problem (Text + Target).
The Brain's own 'SpikeEncoder' handles the signal conversion.
"""

import random

class MathTeacher:
    def __init__(self, n_neurons=None, duration_ms=1000):
        # n_neurons is kept for compatibility but not used here anymore
        # duration_ms is also handled by the brain's encoder
        self.operations = ['+'] 
        
    def generate_sample(self):
        """
        Generates a random addition problem.
        
        Returns:
            input_signal: None (Handled by Brain's Encoder)
            target_class: int (The correct answer, e.g., 8)
            problem_text: str (The question, e.g., "3 + 5")
        """
        # Generate numbers (0-9)
        # Result will be between 0 and 18.
        # Mergen is configured for 20 classes, so this fits perfectly.
        a = random.randint(0, 9)
        b = random.randint(0, 9)
        
        op = '+'
        result = a + b
        
        # Format: "3 + 5"
        # The encoder will turn this string into neural spikes.
        problem_text = f"{a} {op} {b}"
        
        # The target class is the integer result
        target_class = int(result)
        
        # We return None for the first argument because V3 main.py 
        # expects 3 values but generates the signal itself.
        return None, target_class, problem_text

    def get_batch(self, batch_size=1):
        """Helper for batch testing if needed."""
        targets = []
        texts = []
        for _ in range(batch_size):
            _, tar, txt = self.generate_sample()
            targets.append(tar)
            texts.append(txt)
        return None, targets, texts