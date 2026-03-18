from typing import Iterable, Optional

import torch
from torch.optim import Adam


class RiemannianAdam(Adam):
    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        betas=(0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        amsgrad: bool = False,
    ):
        super().__init__(
            params=params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
        )

    @torch.no_grad()
    def step(self, closure: Optional[callable] = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for param in group["params"]:
                if param.grad is None:
                    continue

                if getattr(param, "manifold", None) is None:
                    continue

                grad = param.rgrad if getattr(param, "rgrad", None) is not None else param.grad
                if grad is None:
                    continue

                state = self.state[param]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(param)
                    state["exp_avg_sq"] = torch.zeros_like(param)
                    if group["amsgrad"]:
                        state["max_exp_avg_sq"] = torch.zeros_like(param)

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                state["step"] += 1

                if group["weight_decay"] != 0:
                    grad = grad.add(param, alpha=group["weight_decay"])
                    if hasattr(param.manifold, "proj"):
                        grad = param.manifold.proj(param, grad)

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                if hasattr(param.manifold, "proj"):
                    exp_avg.copy_(param.manifold.proj(param, exp_avg))
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                if group["amsgrad"]:
                    max_exp_avg_sq = state["max_exp_avg_sq"]
                    torch.maximum(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    denom = max_exp_avg_sq.sqrt().add_(group["eps"])
                else:
                    denom = exp_avg_sq.sqrt().add_(group["eps"])

                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]
                step_size = group["lr"] * (bias_correction2 ** 0.5) / bias_correction1
                old_point = param.detach().clone()
                direction = exp_avg / denom
                if hasattr(param.manifold, "proj"):
                    direction = param.manifold.proj(old_point, direction)
                new_point = param.manifold.retr(old_point, -step_size * direction)
                param.copy_(new_point)
                if hasattr(param.manifold, "transp"):
                    exp_avg.copy_(param.manifold.transp(old_point, new_point, exp_avg))
                elif hasattr(param.manifold, "proj"):
                    exp_avg.copy_(param.manifold.proj(new_point, exp_avg))

                param.grad = None

        for group in self.param_groups:
            euclidean = [param for param in group["params"] if getattr(param, "manifold", None) is None]
            if not euclidean:
                continue
            shadow_group = dict(group)
            shadow_group["params"] = euclidean
            shadow_optimizer = Adam(
                shadow_group["params"],
                lr=shadow_group["lr"],
                betas=shadow_group["betas"],
                eps=shadow_group["eps"],
                weight_decay=shadow_group["weight_decay"],
                amsgrad=shadow_group["amsgrad"],
            )
            shadow_optimizer.state = {param: self.state.setdefault(param, {}) for param in euclidean}
            shadow_optimizer.step()
            for param in euclidean:
                param.grad = None

        return loss
