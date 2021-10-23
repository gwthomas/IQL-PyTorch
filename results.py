from pathlib import Path

import numpy as np
import pandas as pd


LOCOMOTION_ENVS = {
    'halfcheetah-medium-v2': 47.4,
    'hopper-medium-v2': 66.3,
    'walker2d-medium-v2': 78.3,
    'halfcheetah-medium-replay-v2': 44.2,
    'hopper-medium-replay-v2': 94.7,
    'walker2d-medium-replay-v2': 73.9,
    'halfcheetah-medium-expert-v2': 86.7,
    'hopper-medium-expert-v2': 91.5,
    'walker2d-medium-expert-v2': 109.6
}

ANTMAZE_ENVS = {
    'antmaze-umaze-v0': 87.5,
    'antmaze-umaze-diverse-v0': 62.2,
    'antmaze-medium-play-v0': 71.2,
    'antmaze-medium-diverse-v0': 70.0,
    'antmaze-large-play-v0': 39.6,
    'antmaze-large-diverse-v0': 47.5
}

KITCHEN_ENVS = {
    'kitchen-complete-v0': 62.5,
    'kitchen-partial-v0': 46.3,
    'kitchen-mixed-v0': 51.0
}

ADROIT_ENVS = {
    'pen-human-v0': 71.5,
    'hammer-human-v0': 1.4,
    'door-human-v0': 4.3,
    'relocate-human-v0': 0.1,
    'pen-cloned-v0': 37.3,
    'hammer-cloned-v0': 2.1,
    'door-cloned-v0': 1.6,
    'relocate-cloned-v0': -0.2
}

ENV_COLLECTIONS = {
    'locomotion-all': LOCOMOTION_ENVS,
    'antmaze-all': ANTMAZE_ENVS,
    'kitchen-all': KITCHEN_ENVS,
    'adroit-all': ADROIT_ENVS
}


def main(args):
    dir = Path(args.dir)
    assert dir.is_dir(), f'{dir} is not a directory'
    print('| Environment | This implementation | Official implementation |\n'
          '| ----------- | ------------------- | ----------------------- |')
    envs = ENV_COLLECTIONS[args.envs]
    for env, ref_score in envs.items():
        env_dir = dir/env
        assert env_dir.is_dir(), f'{env_dir} is not a directory'
        run_dirs = [d for d in env_dir.iterdir() if d.is_dir()]
        final_perfs = []
        for run_dir in run_dirs:
            data = pd.read_csv(run_dir/'progress.csv')
            normalized_returns = data['normalized return mean'].to_numpy()
            final_perfs.append(normalized_returns[-args.last_k:])
        print(f'| {env} | {np.mean(final_perfs):.1f} +/- {np.std(final_perfs):.1f} | {ref_score:.1f} |')


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('-d', '--dir', required=True)
    parser.add_argument('-e', '--envs', required=True)
    parser.add_argument('-k', '--last-k', type=int, default=10)   # average over last k evals
    main(parser.parse_args())