import torch
from typing import Callable

class BaseIntegrator:
    """Base class for time-stepping algorithms."""
    def step(self, state: torch.Tensor, dynamics_func: Callable, dt: float) -> torch.Tensor:
        raise NotImplementedError

class EulerSolver(BaseIntegrator):
    """
    First-order Euler integration.
    Fast, simple, good for standard spiking neurons (LIF).
    Formula: X_new = X_old + (dX/dt * dt)
    """
    def step(self, state: torch.Tensor, dynamics_func: Callable, dt: float) -> torch.Tensor:
        d_state_dt = dynamics_func(state)
        return state + (d_state_dt * dt)

class RungeKutta4Solver(BaseIntegrator):
    """
    Fourth-order Runge-Kutta (RK4).
    
    Why this is legendary:
    Biological oscillators (like Hodgkin-Huxley or Resonators) are very sensitive.
    Euler method makes them unstable. RK4 is extremely precise.
    It samples the slope at 4 different points to predict the future perfectly.
    
    Essential for generating stable Gamma/Theta rhythms.
    """
    def step(self, state: torch.Tensor, dynamics_func: Callable, dt: float) -> torch.Tensor:
        k1 = dynamics_func(state)
        k2 = dynamics_func(state + 0.5 * dt * k1)
        k3 = dynamics_func(state + 0.5 * dt * k2)
        k4 = dynamics_func(state + dt * k3)
        
        return state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)