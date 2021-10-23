# Implicit Q-Learning (IQL) in PyTorch
This repository houses a minimal PyTorch implementation of [Implicit Q-Learning (IQL)](https://arxiv.org/abs/2110.06169), an offline reinforcement learning algorithm, along with a script to run IQL on tasks from the [D4RL](https://github.com/rail-berkeley/d4rl) benchmark.

To install the dependencies, use `pip install -r requirements.txt`.

You can run the script like so:
```
python main.py --log-dir /path/where/results/will/go --env-name hopper-medium-v2 --tau 0.7 --beta 3.0
```

Note that the paper's authors have published [their official implementation](https://github.com/ikostrikov/implicit_q_learning), which is based on JAX. My implementation is intended to be an alternative for PyTorch users, and my general recommendation is to use the authors' code unless you specifically want/need PyTorch for some reason.

I am validating my implementation against the results stated in the paper as compute permits.
Below are results for the MuJoCo locomotion tasks, normalized return at the end of training, averaged (+/- standard deviation) over 3 seeds:

| Environment | This implementation | Official implementation |
| ----------- | ------------------- | ----------------------- |
| halfcheetah-medium-v2 | 47.7 +/- 0.2 | 47.4 |
| hopper-medium-v2 | 61.2 +/- 6.4 | 66.3 |
| walker2d-medium-v2 | 78.7 +/- 4.5 | 78.3 |
| halfcheetah-medium-replay-v2 | 42.9 +/- 1.7 | 44.2 |
| hopper-medium-replay-v2 | 86.8 +/- 15.5 | 94.7 |
| walker2d-medium-replay-v2 | 68.3 +/- 6.4 | 73.9 |
| halfcheetah-medium-expert-v2 | 88.3 +/- 2.8 | 86.7 |
| hopper-medium-expert-v2 | 76.6 +/- 34.9 | 91.5 |
| walker2d-medium-expert-v2 | 108.7 +/- 2.2 | 109.6 |

We can see that the performance is mostly similar to what is stated in the paper, but slightly worse on a few tasks. Note that these results were obtained using a small simplification (deterministic policy and least-squares loss rather than a Gaussian distribution and negative log likelihood), which may explain the discrepancy.
