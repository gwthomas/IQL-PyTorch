from pathlib import Path

import gym
import d4rl
from d4rl import ope
import numpy as np
import torch
from tqdm import trange

from src.iql import ImplicitQLearning
from src.policy import GaussianPolicy
from src.value_functions import TwinQ, ValueFunction
from src.util import sample_batch, evaluate_policy, torchify, set_seed, Log


def main(args):
    log = Log(Path(args.log_dir)/args.env_name)
    log(f'Log dir: {log.dir}')
    env = gym.make(f'{args.env_name}-v2')
    dataset = d4rl.qlearning_dataset(env)
    for k, v in dataset.items():
        dataset[k] = torchify(v)
    obs_dim = dataset['observations'].shape[1]
    act_dim = dataset['actions'].shape[1]   # this assume continuous actions
    set_seed(args.seed, env=env)

    policy = GaussianPolicy(obs_dim, act_dim, hidden_dim=args.hidden_dim, n_hidden=args.n_hidden)
    def eval_policy():
        eval_returns = np.array([evaluate_policy(env, policy, args.max_episode_steps) \
                                 for _ in range(args.n_eval_episodes)])
        log(f'Return: {eval_returns.mean()} +/- {eval_returns.std()}')
        normalized_returns = ope.normalize(args.env_name, eval_returns)
        log(f'Normalized return: {normalized_returns.mean()} +/- {normalized_returns.std()}')

    iql = ImplicitQLearning(
        qf=TwinQ(obs_dim, act_dim, hidden_dim=args.hidden_dim, n_hidden=args.n_hidden),
        vf=ValueFunction(obs_dim, hidden_dim=args.hidden_dim, n_hidden=args.n_hidden),
        policy=policy,
        optimizer_factory=lambda params: torch.optim.Adam(params, lr=args.learning_rate),
        max_steps=args.n_steps,
        tau=args.tau,
        beta=args.beta,
        alpha=args.alpha,
        discount=args.discount
    )
    def debug_info():
        print(f'Current policy LR: {iql.policy_lr_schedule.get_lr()}')

        q_means, v_means, policy_q_means = [], [], []
        for _ in range(100):
            batch = sample_batch(dataset, args.batch_size)
            obs, actions = batch['observations'], batch['actions']
            with torch.no_grad():
                policy_actions = policy(obs).loc
                q_means.append(iql.qf(obs, actions).mean().item())
                v_means.append(iql.vf(obs).mean().item())
                policy_q_means.append(iql.qf(obs, policy_actions).mean().item())
        log(f'Batch Q(s,a) mean: {np.mean(q_means)}')
        log(f'Batch V(s) mean: {np.mean(v_means)}')
        log(f'Batch Q(s,pi(s)) mean: {np.mean(policy_q_means)}')

    for step in trange(args.n_steps):
        iql.update(**sample_batch(dataset, args.batch_size))
        if (step+1) % args.eval_period == 0:
            debug_info()
            eval_policy()

    torch.save(iql.state_dict(), log.dir/'final.pt')


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--env-name', required=True)
    parser.add_argument('--log-dir', required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--discount', type=float, default=0.99)
    parser.add_argument('--hidden-dim', type=int, default=256)
    parser.add_argument('--n-hidden', type=int, default=2)
    parser.add_argument('--n-steps', type=int, default=10**6)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--learning-rate', type=float, default=3e-4)
    parser.add_argument('--alpha', type=float, default=0.005)
    parser.add_argument('--beta', type=float, default=3.0)
    parser.add_argument('--tau', type=float, default=0.7)
    parser.add_argument('--eval-period', type=int, default=5000)
    parser.add_argument('--n-eval-episodes', type=int, default=10)
    parser.add_argument('--max-episode-steps', type=int, default=1000)
    main(parser.parse_args())