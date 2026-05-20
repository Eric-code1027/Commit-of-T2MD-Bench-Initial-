import numpy as np
import torch


class FastEmaModelUpdater:
    """
    Simplified EMA model updater, adapted from TurboDiffusion's FastEmaModelUpdater.
    Maintains a separate target model (EMA) updated from a source model (student).
    Does not register EMA weights as buffers; expects two separate modules with the same architecture.
    """

    @torch.no_grad()
    def copy_to(self, src_model: torch.nn.Module, tgt_model: torch.nn.Module) -> None:
        for tgt_params, src_params in zip(tgt_model.parameters(), src_model.parameters()):
            tgt_params.data.copy_(src_params.data)

    @torch.no_grad()
    def update_average(self, src_model: torch.nn.Module, tgt_model: torch.nn.Module, beta: float = 0.9999) -> None:
        target_list = []
        source_list = []
        for tgt_params, src_params in zip(tgt_model.parameters(), src_model.parameters()):
            target_list.append(tgt_params)
            source_list.append(src_params.data.float().to(tgt_params.device))
        torch._foreach_mul_(target_list, beta)
        torch._foreach_add_(target_list, source_list, alpha=1.0 - beta)


def compute_power_ema_exp(s: float) -> float:
    """
    Compute the PowerEMA exponent from the rate parameter s.
    See EDM2 paper for details.
    """
    return np.roots([1, 7, 16 - s ** -2, 12 - s ** -2]).real.max()


def power_ema_beta(iteration: int, exp: float) -> float:
    """
    Compute the PowerEMA beta (decay rate) for a given iteration.
    beta = (1 - 1/(iteration+1))^(exp+1)
    """
    if iteration < 1:
        return 0.0
    i = iteration + 1
    return (1 - 1 / i) ** (exp + 1)
