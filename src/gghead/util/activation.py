import torch


def mip_sigmoid(x: torch.tensor, overshoot: float = 0.001, clamp: bool = False) -> torch.Tensor:
    values = torch.sigmoid(x) * (1 + 2 * overshoot) - overshoot
    if clamp:
        values = values.clamp(0, 1)
    return values


def mip_tanh(x, overshoot: float = 0.001, clamp: bool = False) -> torch.Tensor:
    values = torch.tanh(x) * (1 + 2 * overshoot) - overshoot
    if clamp:
        values = values.clamp(-1, 1)
    return values


def mip_tanh2(x, overshoot: float = 0.001, clamp: bool = False) -> torch.Tensor:
    values = torch.tanh(x) * (1 + overshoot)
    if clamp:
        values = values.clamp(-1, 1)
    return values
