"""Clohessy-Wiltshire-Hill system."""

import torch
import numpy as np
from torch.utils.data import IterableDataset
import os
import sys
from dataclasses import dataclass

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from function_encoder.model.neural_ode import rk4_step

@dataclass
class CWHParams:
    """Clohessy-Wiltshire-Hill system parameters."""

    orbital_rate: float = 0.001078 # rad/s; sqrt(mu/a3), a~7000
    """Orbital rate."""


def cwh(
    # t: float,
    x: np.ndarray,
    u: np.ndarray,
    params: CWHParams,
    *args,
    **kwargs,
):
    """Clohessey-Wiltshire-Hill (CWH) system.

    The CWH equations are given by

        x_ddot = 3 * n**2 * x + 2 * n * y_dot
        y_ddot = -2 * n * x_dot
        z_ddot = -n**2 * z

    where n is the orbital rate.

    We define the state as [x, y, z, x_dot, y_dot, z_dot]^T and augment the
    system with a control input [u_x, u_y, u_z]^T.

    """

    n = params.orbital_rate

    dx = np.array(
        [
            x[3],
            x[4],
            x[5],
            3 * n**2 * x[0] + 2 * n * x[4] + u[0],
            -2 * n * x[3] + u[1],
            -(n**2) * x[2] + u[2],
        ]
    )

    return dx


def cwh_torch(
    t: torch.Tensor,
    x: torch.Tensor,
    u: torch.Tensor,
    params: CWHParams,
    *args,
    **kwargs,
):
    """PyTorch-compatible version of the CWH system for use with rk4_step."""
    
    n = params.orbital_rate
    
    # Handle both batched and unbatched inputs
    if x.dim() == 1:
        # Single state vector
        dx = torch.stack([
            x[3],
            x[4],
            x[5],
            3 * n**2 * x[0] + 2 * n * x[4] + u[0],
            -2 * n * x[3] + u[1],
            -(n**2) * x[2] + u[2],
        ])
    else:
        # Batched state vectors
        dx = torch.stack([
            x[:, 3],
            x[:, 4],
            x[:, 5],
            3 * n**2 * x[:, 0] + 2 * n * x[:, 4] + u[:, 0],
            -2 * n * x[:, 3] + u[:, 1],
            -(n**2) * x[:, 2] + u[:, 2],
        ], dim=1)
    
    return dx


def _cwh_closed_form(t, x0, params: CWHParams):
    """Closed-form solution of the CWH dynamics."""

    n = params.orbital_rate

    sol = np.array(
        [
            (4 - 3 * np.cos(n * t)) * x0[0]
            + (np.sin(n * t) / n) * x0[3]
            + (2 / n) * (1 - np.cos(n * t)) * x0[4],
            6 * (np.sin(n * t) - n * t) * x0[0]
            + x0[1]
            - (2 / n) * (1 - np.cos(n * t)) * x0[3]
            + ((4 * n - 3 * np.sin(n * t)) / n) * x0[4],
            np.cos(n * t) * x0[2] + (np.sin(n * t) / n) * x0[5],
        ]
    )

    return sol


class CWHDataset(IterableDataset):
    def __init__(
        self,
        n_points: int = 1000,
        n_example_points: int = 100,
        dt=20, # 20s
        pos_range =(0.001, 0.01), # 1e-2 km, v: 1e-4 km/s
        vel_range = (0.00001, 0.0001),
        params: CWHParams = None  
    ):
        super().__init__()
        self.n_points = n_points
        self.n_example_points = n_example_points
        self.pos_range = pos_range
        self.vel_range = vel_range
        self.dt = dt
        self.params = params if params is not None else CWHParams()  # Use default if None


    def __iter__(self):
        while True:
            total_points = self.n_example_points + self.n_points
            
            # Generate random initial conditions (6D state: x, y, z, x_dot, y_dot, z_dot)
            positions = torch.empty(total_points, 3).uniform_(*self.pos_range)
            velocities = torch.empty(total_points, 3).uniform_(*self.vel_range)
            _y0 = torch.cat([positions, velocities], dim=1)
            _dt = torch.full((total_points,), self.dt, dtype=torch.float32)
            u = torch.zeros_like(_y0[:, :3])  # Control input for x, y, z directions

            _y1 = rk4_step(cwh_torch, _y0, _dt, u=u, params=self.params)

            # Split the data
            y0_example = _y0[: self.n_example_points]
            dt_example = _dt[: self.n_example_points]
            y1_example = _y1[: self.n_example_points]

            y0 = _y0[self.n_example_points :]
            dt = _dt[self.n_example_points :]
            y1 = _y1[self.n_example_points :]

            # Store orbital rate as a tensor instead of CWHParams object
            orbital_rate_tensor = torch.tensor([self.params.orbital_rate], dtype=torch.float32)
            
            yield orbital_rate_tensor, y0, dt, y1, y0_example, dt_example, y1_example


# Randomize orbital rate

