""" Exemplu de cod pentru tema 1.
"""

import matplotlib.pyplot as plt
import numpy as np
import gym
from optparse import OptionParser
import gym_minigrid  # pylint: disable=unused-import
from constants import Constants
from state import State
import random
import time
import math


def eps(state, N, constants):
    if N.get(state, 0) == 0:
        return 0.0
    return 1.0 * constants.const / N[state]


def epsilon_greedy(q, state, actions, N, constants):
    max_actions = []
    max_q = float('-inf')
    for action in actions:
        candidate = q.get((state, action), 0)
        if candidate > max_q:
            max_q = candidate
            max_actions.clear()
            max_actions.append(action)
        elif candidate == max_q:
            max_actions.append(action)
    p = []
    for action in actions:
        if action in max_actions:
            p.append(eps(state, N, constants) / len(actions) + (1 - eps(state, N, constants)) / len(max_actions), )
        else:
            p.append(eps(state, N, constants) / len(actions))
    return p


def get_best_action(q, state, actions, N, constants):
    p = epsilon_greedy(q, state, actions, N, constants)
    return random.choices(actions, weights=p)[0]


def sarsa():
    """ Exemplu de evaluare a unei politici pe parcursul antrenării (online).
    """
    parser = OptionParser()
    parser.add_option(
        "-e",
        "--env-name",
        dest="env_name",
        help="gym environment to load",
        default='MiniGrid-Empty-8x8-v0'
    )
    parser.add_option(
        "-a",
        "--alpha",
        dest="alpha",
    )
    parser.add_option(
        "-g",
        "--gamma",
        dest="gamma",
    )
    parser.add_option(
        "-c",
        "--constant",
        dest="constant",
    )

    (options, args) = parser.parse_args()

    # Load the gym environment
    env = gym.make(options.env_name)
    alpha = float(options.alpha)
    gamma = float(options.gamma)
    const = float(options.constant)
    constants = Constants(alpha, gamma, const)

    nsteps = 50000  # Lăsați milioane de pași pentru hărțile mari (16x16)
    report_freq = 2000

    steps, avg_returns, avg_lengths = [], [], []
    recent_returns, recent_lengths = [], []
    crt_return, crt_length = 0, 0

    _obs, done = env.reset(), False
    q = {}
    N = {}
    actions = list(range(0, 6))
    state = State(env.agent_pos, env.agent_dir)
    action = get_best_action(q, state, actions, N, constants)
    for step in range(1, nsteps + 1):
        N[state] = N.get(state, 0) + 1
        new_obs, reward, done, _ = env.step(action)
        new_state = State(env.agent_pos, env.agent_dir)
        new_action = get_best_action(q, new_state, actions, N, constants)
        q[(state, action)] = q.get((state, action), 0) + constants.alpha * (
                    reward + constants.gamma * q.get((new_state, new_action), 0) - q.get((state, action), 0))

        crt_return += reward
        crt_length += 1

        if done:
            _obs, done = env.reset(), False
            recent_returns.append(crt_return)  # câștigul episodului încheiat
            recent_lengths.append(crt_length)
            crt_return, crt_length = 0, 0
        else:
            _obs = new_obs
            action = new_action
            state = new_state

        if step % report_freq == 0:
            avg_return = np.mean(recent_returns)  # media câștigurilor recente
            avg_length = np.mean(recent_lengths)  # media lungimilor episoadelor

            steps.append(step)  # pasul la care am reținut valorile
            avg_returns.append(avg_return)
            avg_lengths.append(avg_length)

            print(  # pylint: disable=bad-continuation
                f"Step {step:4d}"
                f" | Avg. return = {avg_return:.2f}"
                f" | Avg. ep. length: {avg_length:.2f}"
            )
            recent_returns.clear()
            recent_lengths.clear()

    return steps, avg_lengths, avg_returns
    # La finalul antrenării afișăm evoluția câștigului mediu
    # În temă vreau să faceți media mai multor astfel de traiectorii pentru
    # a nu trage concluzii fără a lua în calcul varianța algoritmilor




if __name__ == "__main__":

    avg_lengths = {}
    avg_returns = {}
    AVG_SAMPLE = 10
    for i in range(0, AVG_SAMPLE):
        steps, lengths, returns = sarsa()
        for j in range(0, len(steps)):
            step = steps[j]
            avg_lengths[step] = avg_lengths.get(step, 0) + lengths[j] / AVG_SAMPLE
            avg_returns[step] = avg_returns.get(step, 0) + returns[j] / AVG_SAMPLE

    _fig, (ax1, ax2) = plt.subplots(ncols=2)
    ax1.plot(steps, avg_lengths.values(), label="random")
    ax1.set_title("Average episode length")
    ax1.legend()
    ax2.plot(steps, avg_returns.values(), label="random")
    ax2.set_title("Average episode return")
    ax2.legend()
    plt.show()
