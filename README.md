# IQL-PyTorch
This repository houses a minimal PyTorch implementation of [Implicit Q-Learning (IQL)](https://arxiv.org/abs/2110.06169), an offline reinforcement learning algorithm, along with a script to run IQL on tasks from the [D4RL](https://github.com/rail-berkeley/d4rl) benchmark.

Note that the paper's authors have published [their official implementation](https://github.com/ikostrikov/implicit_q_learning), which is based on JAX. My implementation is intended to be an alternative for PyTorch users, and my general recommendation is to use the authors' code unless you specifically want/need PyTorch for some reason. I am planning to validate my implementation against the results stated in the paper once I have some spare compute.
