import torch
from jaxtyping import Bool, Float, Int


def cross_entropy(
    inputs: Float[torch.Tensor, " batch_size vocab_size"],targets: Int[torch.Tensor, " batch_size"]
) -> Float[torch.Tensor, ""]:
    # 1. Subtract the largest element for numerical stability
    shifted = inputs - inputs.max(dim=-1, keepdim=True).values

    # 2. Cancel out log and exp whenever possible
    log_probs = shifted - torch.log(torch.exp(shifted).sum(dim=-1, keepdim=True))

    target_log_probs = log_probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    return -target_log_probs.mean()


class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)