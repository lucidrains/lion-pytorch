from typing import Tuple, Optional, Callable
import sys
sys.path.append('../geoopt/')
import geoopt

import torch
from torch.optim.optimizer import Optimizer

# functions

def exists(val):
    return val is not None

# update functions

def update_fn(p, grad, exp_avg, lr, wd, beta1, beta2):
    # stepweight decay

    p.data.mul_(1 - lr * wd)

    # weight update

    update = exp_avg.clone().mul_(beta1).add(grad, alpha = 1 - beta1).sign_()
    p.add_(update, alpha = -lr)

    # decay the momentum running average coefficient

    exp_avg.mul_(beta2).add_(grad, alpha = 1 - beta2)

# class
class RLion(geoopt.optim.mixin.OptimMixin,Optimizer):
  r"""Implements Lion algorithm."""

  def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
    """Initialize the hyperparameters.

    Args:
      params (iterable): iterable of parameters to optimize or dicts defining
        parameter groups
      lr (float, optional): learning rate (default: 1e-4)
      betas (Tuple[float, float], optional): coefficients used for computing
        running averages of gradient and its square (default: (0.9, 0.99))
      weight_decay (float, optional): weight decay coefficient (default: 0)
    """

    if not 0.0 <= lr:
      raise ValueError('Invalid learning rate: {}'.format(lr))
    if not 0.0 <= betas[0] < 1.0:
      raise ValueError('Invalid beta parameter at index 0: {}'.format(betas[0]))
    if not 0.0 <= betas[1] < 1.0:
      raise ValueError('Invalid beta parameter at index 1: {}'.format(betas[1]))
    defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
    super().__init__(params, defaults)

  @torch.no_grad()
  def step(self, closure=None):
    """Performs a single optimization step.

    Args:
      closure (callable, optional): A closure that reevaluates the model
        and returns the loss.

    Returns:
      the loss.
    """
    loss = None
    if closure is not None:
      with torch.enable_grad():
        loss = closure()

    for group in self.param_groups:
      for p in group['params']:
        
        if p.grad is None:
          continue
        
        if isinstance(p, (geoopt.tensor.ManifoldParameter, geoopt.tensor.ManifoldTensor)):
            manifold = p.manifold
        else:
            manifold = self._default_manifold
            
        learning_rate = group["lr"]    
        # Perform stepweight decay
        p.data.mul_(1 - group['lr'] * group['weight_decay'])

        grad = p.grad
        
        grad = manifold.egrad2rgrad(p, grad) 
        
        
        state = self.state[p]
        # State initialization
        if len(state) == 0:
          # Exponential moving average of gradient values
          state['exp_avg'] = torch.zeros_like(p)

        exp_avg = state['exp_avg']
        beta1, beta2 = group['betas']

        # Weight update
        #update = exp_avg * beta1 + grad * (1 - beta1)
        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
        direction = torch.sign(exp_avg)
        new_point, exp_avg_new = manifold.retr_transp(
                        p, -learning_rate * direction, exp_avg
                    )
        
        
        
#         p.add_(torch.sign(update), alpha=-group['lr'])
        p.copy_(new_point)
#         exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)
        exp_avg.copy_(exp_avg_new)

    return loss
