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
from openpyxl import load_workbook
import pandas as pd


def eps(state, N, constants):
    if N.get(state, 0) == 0:
        return 1.0
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


def beta(q, state, N, actions):
    if N.get(state, 0) <= 1:
        return 1
    max_dif = float('-inf')
    for action1 in actions:
        for action2 in actions:
            diff = abs(q.get((state, action1), 0) - q.get((state, action2), 0))
            if diff >= max_dif:
                max_dif = diff
    if max_dif == 0:
        return 1
    return math.log(N[state]) / max_dif


def boltzman(q, state, actions, N, constants):
    p = {}
    total = 0.0
    beta_calculated = beta(q, state, N, actions)
    for action in actions:
        p[action] = math.exp(beta_calculated * q.get((state, action), 0))
        total += p[action]

    for action in actions:
        p[action] = p[action] / total

    return p.values()


def get_best_action(q, state, actions, N, constants):
    p = boltzman(q, state, actions, N, constants)
    return random.choices(actions, weights=p)[0]


def sarsa(env, constants):
    """ Exemplu de evaluare a unei politici pe parcursul antrenării (online).
    """

    report_freq = 2000

    steps, avg_returns, avg_lengths = [], [], []
    recent_returns, recent_lengths = [], []
    crt_return, crt_length = 0, 0

    _obs, done = env.reset(), False
    q = {}
    N = {}
    actions = list(range(0, 7))
    state = State(env.agent_pos, env.agent_dir)
    action = get_best_action(q, state, actions, N, constants)
    for step in range(1, constants.no_steps + 1):
        N[state] = N.get(state, 0) + 1
        new_obs, reward, done, _ = env.step(action)
        new_state = State(env.agent_pos, env.agent_dir)
        new_action = get_best_action(q, new_state, actions, N, constants)
        q[(state, action)] = q.get((state, action), 0) + constants.alpha * (
                    reward + constants.gamma * q.get((new_state, new_action), 0) - q.get((state, action), 0))

        crt_return += reward
        crt_length += 1
        # env.render('human')
        # time.sleep(0.01)

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
            #
            # print(  # pylint: disable=bad-continuation
            #     f"Step {step:4d}"
            #     f" | Avg. return = {avg_return:.2f}"
            #     f" | Avg. ep. length: {avg_length:.2f}"
            # )
            recent_returns.clear()
            recent_lengths.clear()

    return steps, avg_lengths, avg_returns
    # La finalul antrenării afișăm evoluția câștigului mediu
    # În temă vreau să faceți media mai multor astfel de traiectorii pentru
    # a nu trage concluzii fără a lua în calcul varianța algoritmilor


def init_envs():
    envs = {}
    envs['MiniGrid-Empty-6x6-v0'] = 24000
    envs['MiniGrid-Empty-8x8-v0'] = 40000
    envs['MiniGrid-Empty-16x16-v0'] = 1000000
    envs['MiniGrid-DoorKey-6x6-v0'] = 500000
    envs['MiniGrid-DoorKey-8x8-v0'] = 1000000
    envs['MiniGrid-DoorKey-16x16-v0'] = 10000000
    return envs


if __name__ == "__main__":

    # parser = OptionParser()
    # parser.add_option(
    #     "-e",
    #     "--env-name",
    #     dest="env_name",
    #     help="gym environment to load",
    #     default='MiniGrid-Empty-8x8-v0'
    # )
    # parser.add_option(
    #     "-a",
    #     "--alpha",
    #     dest="alpha",
    # )
    # parser.add_option(
    #     "-g",
    #     "--gamma",
    #     dest="gamma",
    # )
    # parser.add_option(
    #     "-c",
    #     "--constant",
    #     dest="constant",
    # )
    #
    # (options, args) = parser.parse_args()
    # Load the gym environment

    # alpha = float(options.alpha)
    # gamma = float(options.gamma)
    # const = float(options.constant)

    envs = init_envs()
    for filename in envs.keys():
        env = gym.make(filename)
        no_steps = envs[filename]
        writer_length = pd.ExcelWriter(filename + '_lenghts.xlsx', engine='openpyxl')
        wb_legnth = writer_length.book
        writer_return = pd.ExcelWriter(filename + '_returns.xlsx', engine='openpyxl')
        wb_return = writer_length.book
        length_map = {}
        return_map = {}
        for alpha in np.arange(0.0, 1.0, 0.2):
            print (alpha)
            for gamma in np.arange(0.0, 1.0, 0.2):
                for const in range(0, 1, 20):
                    constants = Constants(alpha, gamma, const, no_steps)

                    avg_lengths = {}
                    avg_returns = {}
                    AVG_SAMPLE = 5
                    for i in range(0, AVG_SAMPLE):
                        steps, lengths, returns = sarsa(env, constants)
                        for j in range(0, len(steps)):
                            step = steps[j]
                            avg_lengths[step] = avg_lengths.get(step, 0) + lengths[j] / AVG_SAMPLE
                            avg_returns[step] = avg_returns.get(step, 0) + returns[j] / AVG_SAMPLE
                    length_map[str(constants)] = avg_lengths
                    return_map[str(constants)] = avg_returns

        df = pd.DataFrame(length_map)
        df.to_excel(writer_length, index=False)
        wb_legnth.save(filename + '_lenghts.xlsx')

        df = pd.DataFrame(return_map)
        df.to_excel(writer_length, index=False)
        wb_legnth.save(filename + '_returns.xlsx')

    # _fig, (ax1, ax2) = plt.subplots(ncols=2)
    # ax1.plot(steps, avg_lengths.values(), label="random")
    # ax1.set_title("Average episode length")
    # ax1.legend()
    # ax2.plot(steps, avg_returns.values(), label="random")
    # ax2.set_title("Average episode return")
    # ax2.legend()
    # plt.show()
