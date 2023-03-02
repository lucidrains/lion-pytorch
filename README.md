<img src="./lion.png" width="500px"></img>

## 🦁 Lion - Pytorch

<a href="https://arxiv.org/abs/2302.06675">🦁 Lion</a>, Evo**L**ved S**i**gn M**o**me**n**tum, new optimizer discovered by Google Brain that is purportedly better than Adam(w), in Pytorch. This is nearly a straight copy from <a href="https://github.com/google/automl/blob/master/lion/lion_pytorch.py">here</a>, with few minor modifications.

It is so simple, we may as well get it accessible and used asap by everyone to train some great models, if it really works 🤞

In regards to learning rate and weight decay, the authors write in section 5 - `Based on our experience, a suitable learning rate for Lion is typically 3-10x smaller than that for AdamW. Since the effective weight decay is lr * λ, the value of decoupled weight decay λ used for Lion is 3-10x larger than that for AdamW in order to maintain a similar strength.`

The authors use the same learning rate schedule for Lion as AdamW in the paper. Nevertheless, they observe a larger gain when using a cosine decay schedule, compared to a reciprocal square-root schedule.

Update: seems to work for my local enwik8 autoregressive language modeling

Update 2: <a href="https://api.wandb.ai/links/lucidrains/d4v6c8sl">experiments</a>, seems much worse than Adam if learning rate held constant

Update 3: Dividing the learning rate by 3, seeing better early results than Adam. Maybe Adam has been dethroned, after nearly a decade.

Update 4: using the 10x smaller learning rate rule of thumb from the paper resulted in the worst run. so I guess it still takes a bit of tuning

Update 5: so far hearing all positive results for language modeling, when done right. also heard positive results for significant text-to-image training, although it takes a bit of tuning. the negative results seem to be with problems and architectures outside of what was evaluated in the paper - RL, feedforward networks, weird hybrid architectures with LSTMs + convolutions etc. negative anecdata also confirms this technique is sensitive to batch size, amount of data / augmentation. tbd what optimal learning rate schedule is, and whether cooldown affects results. also interestingly have a positive result at open-clip, which became negative as the model size was scaled up (but may be resolvable)

## Install

```bash
$ pip install lion-pytorch
```

## Usage

```python
# toy model

import torch
from torch import nn

model = nn.Linear(10, 1)

# import Lion and instantiate with parameters

from lion_pytorch import Lion

opt = Lion(model.parameters(), lr = 1e-4, weight_decay = 1e-2)

# forward and backwards

loss = model(torch.randn(10))
loss.backward()

# optimizer step

opt.step()
opt.zero_grad()
```

To use a fused kernel for updating the parameters, first `pip install triton -U --pre`, then

```python
opt = Lion(
    model.parameters(),
    lr = 1e-4,
    weight_decay = 1e-2,
    use_triton = True # set this to True to use cuda kernel w/ Triton lang (Tillet et al)
)
```

## Appreciation

- <a href="https://stability.ai/">Stability.ai</a> for the generous sponsorship to work and open source cutting edge artificial intelligence research

## Citations

```bibtex
@misc{https://doi.org/10.48550/arxiv.2302.06675,
    url     = {https://arxiv.org/abs/2302.06675},
    author  = {Chen, Xiangning and Liang, Chen and Huang, Da and Real, Esteban and Wang, Kaiyuan and Liu, Yao and Pham, Hieu and Dong, Xuanyi and Luong, Thang and Hsieh, Cho-Jui and Lu, Yifeng and Le, Quoc V.},
    title   = {Symbolic Discovery of Optimization Algorithms},
    publisher = {arXiv},
    year = {2023}
}
```

```bibtex
@article{Tillet2019TritonAI,
    title   = {Triton: an intermediate language and compiler for tiled neural network computations},
    author  = {Philippe Tillet and H. Kung and D. Cox},
    journal = {Proceedings of the 3rd ACM SIGPLAN International Workshop on Machine Learning and Programming Languages},
    year    = {2019}
}
```
