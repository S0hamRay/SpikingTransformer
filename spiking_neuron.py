from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn


class SurrogateHeaviside(torch.autograd.Function):
    """Binary spike function with a piecewise-linear surrogate gradient."""

    @staticmethod
    def forward(ctx, membrane: Tensor, threshold: Tensor) -> Tensor:
        """Emit binary spikes in the forward pass.

        Args:
            membrane: Membrane potential tensor.
            threshold: Spike threshold tensor.

        Returns:
            Tensor with values in {0, 1}.
        """
        ctx.save_for_backward(membrane, threshold)
        return (membrane >= threshold).to(membrane.dtype)

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> tuple[Tensor, Tensor]:
        """Backpropagate with a triangular surrogate gradient.

        Args:
            grad_output: Upstream gradient.

        Returns:
            Gradient with respect to the membrane potential and threshold.
        """
        membrane, threshold = ctx.saved_tensors
        surrogate_grad = (1.0 - (membrane - threshold).abs()).clamp(0.0, 1.0)
        membrane_grad = grad_output * surrogate_grad
        threshold_grad = (-membrane_grad).sum_to_size(threshold.shape)
        return membrane_grad, threshold_grad


class SurrogateTernarySpike(torch.autograd.Function):
    """Ternary spike function with a piecewise-linear surrogate gradient."""

    @staticmethod
    def forward(ctx, membrane: Tensor, threshold: Tensor) -> Tensor:
        """Emit ternary spikes in the forward pass.

        Args:
            membrane: Membrane potential tensor.
            threshold: Magnitude threshold tensor.

        Returns:
            Tensor with values in {-1, 0, 1}.
        """
        ctx.save_for_backward(membrane, threshold)
        magnitude_spike = (membrane.abs() >= threshold).to(membrane.dtype)
        return membrane.sign() * magnitude_spike

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> tuple[Tensor, Tensor]:
        """Backpropagate with a symmetric triangular surrogate gradient.

        Args:
            grad_output: Upstream gradient.

        Returns:
            Gradient with respect to the membrane potential and threshold.
        """
        membrane, threshold = ctx.saved_tensors
        surrogate_grad = (1.0 - (membrane.abs() - threshold).abs()).clamp(0.0, 1.0)
        membrane_grad = grad_output * surrogate_grad
        threshold_grad = (-membrane_grad).sum_to_size(threshold.shape)
        return membrane_grad, threshold_grad


class LIFNeuron(nn.Module):
    """Leaky integrate-and-fire neuron with persistent hidden state."""

    def __init__(
        self,
        vth: float = 0.5,
        beta: float = 0.25,
        vreset: float = 0.0,
        learnable_threshold: bool = False,
    ) -> None:
        """Initialize the stateful LIF neuron.

        Args:
            vth: Firing threshold.
            beta: Membrane decay factor after non-spiking updates.
            vreset: Reset value applied after a spike.
            learnable_threshold: Whether the threshold is trainable.
        """
        super().__init__()
        threshold = torch.tensor(float(vth), dtype=torch.float32)
        if learnable_threshold:
            self.vth = nn.Parameter(threshold)
        else:
            self.register_buffer('vth', threshold)
        self.register_buffer('beta', torch.tensor(float(beta), dtype=torch.float32))
        self.register_buffer('vreset', torch.tensor(float(vreset), dtype=torch.float32))
        self.hidden_state: Optional[Tensor] = None

    def reset(self) -> None:
        """Clear the persistent hidden state between batches."""
        self.hidden_state = None

    def forward(self, x: Tensor, reset: bool = False) -> Tensor:
        """Update the membrane potential for one timestep.

        Args:
            x: Input current for the current timestep.
            reset: Whether to clear the hidden state before processing `x`.

        Returns:
            Binary spike tensor with the same shape as `x`.
        """
        if reset:
            self.reset()

        if (
            self.hidden_state is None
            or self.hidden_state.shape != x.shape
            or self.hidden_state.device != x.device
            or self.hidden_state.dtype != x.dtype
        ):
            self.hidden_state = torch.zeros_like(x)

        threshold = self.vth.to(device=x.device, dtype=x.dtype)
        beta = self.beta.to(device=x.device, dtype=x.dtype)
        vreset = self.vreset.to(device=x.device, dtype=x.dtype)

        membrane = self.hidden_state + x
        spikes = SurrogateHeaviside.apply(membrane, threshold)
        self.hidden_state = vreset * spikes + beta * membrane * (1.0 - spikes)
        return spikes


class TernaryLIFNeuron(nn.Module):
    """Ternary LIF neuron that emits negative, zero, or positive spikes."""

    def __init__(
        self,
        vth: float = 0.5,
        beta: float = 0.25,
        vreset: float = 0.0,
        learnable_threshold: bool = False,
    ) -> None:
        """Initialize the ternary LIF neuron.

        Args:
            vth: Magnitude threshold for spiking.
            beta: Membrane decay factor after non-spiking updates.
            vreset: Reset value applied after a non-zero spike.
            learnable_threshold: Whether the threshold is trainable.
        """
        super().__init__()
        threshold = torch.tensor(float(vth), dtype=torch.float32)
        if learnable_threshold:
            self.vth = nn.Parameter(threshold)
        else:
            self.register_buffer('vth', threshold)
        self.register_buffer('beta', torch.tensor(float(beta), dtype=torch.float32))
        self.register_buffer('vreset', torch.tensor(float(vreset), dtype=torch.float32))
        self.hidden_state: Optional[Tensor] = None

    def reset(self) -> None:
        """Clear the persistent hidden state between batches."""
        self.hidden_state = None

    def forward(self, x: Tensor, reset: bool = False) -> Tensor:
        """Update the membrane potential for one timestep.

        Args:
            x: Input current for the current timestep.
            reset: Whether to clear the hidden state before processing `x`.

        Returns:
            Ternary spike tensor with values in {-1, 0, 1}.
        """
        if reset:
            self.reset()

        if (
            self.hidden_state is None
            or self.hidden_state.shape != x.shape
            or self.hidden_state.device != x.device
            or self.hidden_state.dtype != x.dtype
        ):
            self.hidden_state = torch.zeros_like(x)

        threshold = self.vth.to(device=x.device, dtype=x.dtype)
        beta = self.beta.to(device=x.device, dtype=x.dtype)
        vreset = self.vreset.to(device=x.device, dtype=x.dtype)

        membrane = self.hidden_state + x
        spikes = SurrogateTernarySpike.apply(membrane, threshold)
        spike_gate = spikes.abs()
        self.hidden_state = vreset * spikes + beta * membrane * (1.0 - spike_gate)
        return spikes
