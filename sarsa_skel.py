""" Exemplu de cod pentru tema 1.
"""

import matplotlib.pyplot as plt
import numpy as np
import gym
import time

from constants import Constants
from state import State
import random
import math
import gym_minigrid


def ucb(q, state, actions, N, constants):
    max_action = None
    max_value = float('-inf')
    logN_state = math.log(N.get(state, 1))

    for action in actions:
        candidate = q.get((state, action), constants.q0) + constants.const * \
            math.sqrt(logN_state / N.get((state, action), 1))

        if candidate > max_value:
            max_action = action
            max_value = candidate

    return max_action


def eps(state, N, constants):
    if N.get(state, 0) == 0:
        return 1.0
    return 1.0 * constants.const / N[state]


def epsilon_greedy(q, state, actions, N, constants):
    max_actions = []
    max_q = float('-inf')
    for action in actions:
        candidate = q.get((state, action), constants.q0)
        if candidate > max_q:
            max_q = candidate
            max_actions.clear()
            max_actions.append(action)
        elif candidate == max_q:
            max_actions.append(action)
    p = []
    for action in actions:
        if action in max_actions:
            p.append(eps(state, N, constants) / len(actions) + (1 - eps(state, N, constants)) / len(max_actions))
        else:
            p.append(eps(state, N, constants) / len(actions))
    return p


def beta(q, state, N, actions, constants):
    if N.get(state, 0) == 0:
        return 0
    max_dif = float('-inf')
    for action1 in actions:
        for action2 in actions:
            diff = abs(q.get((state, action1), constants.q0) - q.get((state, action2), constants.q0))
            if diff >= max_dif:
                max_dif = diff
    if max_dif == 0:
        return 1
    return math.log(N[state]) / max_dif


def boltzman(q, state, actions, N, constants):
    p = {}
    total = 0.0
    beta_calculated = beta(q, state, N, actions, constants)
    for action in actions:
        p[action] = np.exp(beta_calculated * q.get((state, action), constants.q0))
        total += p[action]

    for action in actions:
        p[action] = p[action] / total

    return list(p.values())


def get_best_action(q, state, actions, N, constants, expl_func):
    if expl_func == ucb:
        return expl_func(q, state, actions, N, constants)
    p = expl_func(q, state, actions, N, constants)
    return np.random.choice(actions, p=p)


def sarsa(env, constants, expl_func):
    """ Exemplu de evaluare a unei politici pe parcursul antrenării (online).
    """

    report_freq = constants.no_steps / 100

    steps, avg_returns, avg_lengths = [], [], []
    recent_returns, recent_lengths = [], []
    crt_return, crt_length = 0, 0

    _obs, done = env.reset(), False
    q = {}
    N = {}
    actions = list(range(0, 6))
    state = State(env.agent_pos, env.agent_dir, _obs['image'])
    action = get_best_action(q, state, actions, N, constants, expl_func)
    for step in range(1, constants.no_steps + 1):

        N[state] = N.get(state, 0) + 1
        N[(state, action)] = N.get((state, action), 0) + 1
        new_obs, reward, done, _ = env.step(action)
        new_state = State(env.agent_pos, env.agent_dir, new_obs['image'])

        new_action = get_best_action(q, new_state, actions, N, constants, expl_func)
        q[(state, action)] = q.get((state, action), constants.q0) + constants.alpha * (reward + constants.gamma * q.get((new_state, new_action), constants.q0) - q.get((state, action), constants.q0))

        crt_return += reward
        crt_length += 1
        if step > constants.no_steps * 10 /9:
            env.render('human')
            time.sleep(0.01)

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

            recent_returns.clear()
            recent_lengths.clear()

    return steps, avg_lengths, avg_returns
    # La finalul antrenării afișăm evoluția câștigului mediu
    # În temă vreau să faceți media mai multor astfel de traiectorii pentru
    # a nu trage concluzii fără a lua în calcul varianța algoritmilor


def get_no_steps(map_name):
    if map_name == 'MiniGrid-Empty-6x6-v0':
        return 10000
    if map_name == 'MiniGrid-Empty-8x8-v0':
        return 60000
    if map_name == 'MiniGrid-Empty-16x16-v0':
        return 1000000
    if map_name == 'MiniGrid-DoorKey-6x6-v0':
        return 5000000
    if map_name == 'MiniGrid-DoorKey-8x8-v0':
        return 50000000
    if map_name == 'MiniGrid-DoorKey-16x16-v0':
        return 100000000


def get_exploration_func(func_name):
    if func_name == 'boltzman':
        return boltzman
    if func_name == 'epsilon':
        return epsilon_greedy
    if func_name == 'ucb':
        return ucb


def start_sarsa(options):

    # Load the gym environment

    alpha = float(options['alpha'])
    gamma = float(options['gamma'])
    const = float(options['constant'])
    exploration_func = get_exploration_func(options['exploration'])
    map_name = options['env_name']
    env = gym.make(map_name)
    no_steps = get_no_steps(map_name)
    q0 = float(options['q0'])
    constants = Constants(alpha, gamma, const, no_steps, q0)

    avg_lengths = {}
    avg_returns = {}
    AVG_SAMPLE = 1
    steps = []
    for i in range(0, AVG_SAMPLE):
        # try:
        steps, lengths, returns = sarsa(env, constants, exploration_func)
        for j in range(0, len(steps)):
            step = steps[j]
            avg_lengths[step] = avg_lengths.get(step, 0) + lengths[j] / AVG_SAMPLE
            avg_returns[step] = avg_returns.get(step, 0) + returns[j] / AVG_SAMPLE
        # except Exception as e:
        #     print('FAILED -> ' + str(options['exploration']) + " - " + str(map_name) + " - " + str(constants))
        #     print(e)

    f = open("results/test/" + str(options['exploration']) + "/" + str(map_name) + "/" + str(constants), "w+")
    f.write(str(constants) + "\r\n")
    for j in range(0, len(steps)):
        step = steps[j]
        f.write("%d %.2f %.2f\r\n" % (step, avg_lengths[step], avg_returns[step]))
    f.close()
