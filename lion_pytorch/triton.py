import torch
from torch import Tensor

try:
    import triton
    import triton.language as tl
except ImportError as e:
    print('triton is not installed, please install by running `pip install triton -U --pre`')
    exit()

# helper functions

def calc_num_warps(block_size):
    num_warps = 4
    if block_size >= 2048:
        num_warps = 8
    if block_size >= 4096:
        num_warps = 16
    return num_warps

# triton cuda kernel

@triton.jit
def update_fn_kernel(
    p_ptr,
    grad_ptr,
    exp_avg_ptr,
    lr,
    wd,
    beta1,
    beta2,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis = 0)

    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    mask = offsets < n_elements

    # offsetted pointers

    offset_p_ptr = p_ptr + offsets
    offset_grad_ptr = grad_ptr + offsets
    offset_exp_avg_ptr = exp_avg_ptr + offsets

    # load

    p = tl.load(offset_p_ptr, mask = mask)
    grad = tl.load(offset_grad_ptr, mask = mask)
    exp_avg = tl.load(offset_exp_avg_ptr, mask = mask)

    # stepweight decay

    p = p * (1 - lr * wd)

    # diff between momentum running average and grad

    diff = exp_avg - grad

    # weight update

    update = diff * beta1 + grad

    # torch.sign

    can_update = update != 0
    update_sign = tl.where(update > 0, -lr, lr)

    p = p + update_sign * can_update

    # decay the momentum running average coefficient

    exp_avg = diff * beta2 + grad

    # store new params and momentum running average coefficient

    tl.store(offset_p_ptr, p, mask = mask)
    tl.store(offset_exp_avg_ptr, exp_avg, mask = mask)

def update_fn(
    p: Tensor,
    grad: Tensor,
    exp_avg: Tensor,
    lr: float,
    wd: float,
    beta1: float,
    beta2: float,
    inplace: bool = True,
    BLOCK_SIZE: int = 1024
):
    assert all([t.is_cuda for t in (p, grad, exp_avg)])

    n_elements = p.numel()

    block_size = triton.next_power_of_2(BLOCK_SIZE)
    num_warps = calc_num_warps(block_size)
    n_rows = triton.cdiv(n_elements, block_size)

    # call triton cuda kernel

    update_fn_kernel[(n_rows,)](
        p,
        grad,
        exp_avg,
        lr,
        wd,
        beta1,
        beta2,
        n_elements,
        num_warps = num_warps,
        BLOCK_SIZE = BLOCK_SIZE
    )
