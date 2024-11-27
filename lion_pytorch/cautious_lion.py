from __future__ import annotations
from typing import Tuple, Callable

import torch
from torch.optim.optimizer import Optimizer

# functions

def exists(val):
    return val is not None

# class

class Lion(Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas: Tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0.0,
        cautious_factor: float = 0.,
        decoupled_weight_decay: bool = False,
    ):
        assert lr > 0.
        assert all([0. <= beta <= 1. for beta in betas])
        assert 0. <= cautious_factor <= 1.

        self._init_lr = lr
        self.decoupled_wd = decoupled_weight_decay

        defaults = dict(
            lr = lr,
            betas = betas,
            weight_decay = weight_decay,
            cautious_factor = cautious_factor
        )

        super().__init__(params, defaults)

    @torch.no_grad()
    def step(
        self,
        closure: Callable | None = None
    ):

        loss = None
        if exists(closure):
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in filter(lambda p: exists(p.grad), group['params']):

                grad, lr, wd, cautious_factor, beta1, beta2, state, decoupled_wd, init_lr = p.grad, group['lr'], group['weight_decay'], group['cautious_factor'], *group['betas'], self.state[p], self.decoupled_wd, self._init_lr

                # maybe decoupled weight decay

                if decoupled_wd:
                    wd /= init_lr

                # init state - exponential moving average of gradient values

                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p)

                exp_avg = state['exp_avg']

                # stepweight decay

                p.data.mul_(1. - lr * wd)

                # weight update

                update = exp_avg.clone().mul_(beta1).add(grad, alpha = 1. - beta1).sign_()

                # maybe cautious update - algorithm 2 in https://arxiv.org/abs/2411.16085

                if cautious_factor < 1.:
                    align_mask = (update * grad) > 0
                    scale = torch.where(align_mask, torch.ones_like(grad), cautious_factor)
                    scale /= scale.mean().clamp(min = 1e-5)
                    update.mul_(scale)

                # update params

                p.add_(update, alpha = -lr)

                # decay the momentum running average coefficient

                exp_avg.mul_(beta2).add_(grad, alpha = 1. - beta2)

        return loss
