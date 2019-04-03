#!/usr/bin/env python3

from __future__ import division, print_function

import sys
import numpy as np
import gym
import time
from optparse import OptionParser
import random
import matplotlib.pyplot as plt
import math

import gym_minigrid

class State:
    def __init__(self, pos, direction):
        self.pos = pos
        self.direction = direction

    def __eq__(self, other):
        """Overrides the default implementation"""
        if isinstance(other, State):
            return self.pos[0] == other.pos[0] and self.pos[1] == other.pos[1] and self.direction == other.direction
        return False

    def __hash__(self) -> int:
        return hash((self.pos[0], self.pos[1], self.direction))

    def __str__(self) -> str:
        return str(self.pos) + " " + str(self.direction)


class Constants:
    def __init__(self, alpha, gamma, const):
        self.alpha = alpha
        self.gamma = gamma
        self.const = const

def init_states(env):
    states = []
    for i in range(0, env.grid.width):
        for j in range(0, env.grid.height):
            for direction in range(0, 4):
                state = State((i, j), direction)
                states.append(state)
    return states


def init_actions(env):
    # TODO
    return list(range(0, 7))


def resetEnv(env):
    env.reset()
    if not hasattr(env, 'mission'):
        print('!!!!!!!!!!!!!COULD NOT CREATE NEW MISSION!!!!!!!!!')

def main():
    parser = OptionParser()
    parser.add_option(
        "-e",
        "--env-name",
        dest="env_name",
        help="gym environment to load",
        default='MiniGrid-Empty-6x6-v0'
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

    resetEnv(env)
    states = init_states(env)
    actions = init_actions(env)

    # Create a window to render into
    renderer = env.render('human')


    # renderer.window.setKeyDownCb(keyDownCb)
    sarsa(env, states, actions, constants)

    # while True:
    #     env.render('human')
    #     time.sleep(0.01)
    #
    #     # If the window was closed
    #     if renderer.window == None:
    #         break


Q0 = 0
NO_EPISODES = 100

def eps(state, N, constants):
    if N[state] == 0:
        return 0
    return 1.0 * constants.const / N[state]

def get_action_name(action_num):
    if action_num == 0:
        return "LEFT"
    elif action_num == 1:
        return "RIGHT"
    elif action_num == 2:
        return "FRONT"
    else:
        return str(action_num)

def epsilon_greedy(q, state, actions, N, constants):
    max_actions = []
    max_q = float('-inf')
    for action in actions:
        if q.get((state, action), 0) > max_q:
            max_q = q.get((state, action), 0)
            max_actions.clear()
            max_actions.append(action)
        elif q.get((state, action), 0) == max_q:
            max_actions.append(action)
    p = []
    for action in actions:
        if action in max_actions:
            p.append(eps(state, N, constants) / len(actions) + (1 - eps(state, N, constants)) / len(max_actions), )
        else:
            p.append(eps(state, N, constants) / len(actions))
    return p


def beta(q, state, N, actions):
    if N[state] <= 1:
        return 1
    max_dif = float('-inf')
    for action1 in actions:
        for action2 in actions:
            diff = abs(q.get((state, action1), 0) - q.get((state, action2), 0))
            if diff >= max_dif:
                max_dif = diff
    return max_dif / math.log(N[state])


def boltzman(q, state, actions, N):
    p = {}
    total = 0.0
    beta_calculated = beta(q, state, N, actions)
    for action in actions:
        p[action] = math.exp(beta_calculated * q.get((state, action), 0))
        total += p[action]

    for action in actions:
        p[action] = p[action] / total

    return p.values()

def get_action(state, q, actions, N, constants, debug=False):
    # TODO
    # unexplored = []
    # for legal_action in actions:
    #     if (state, legal_action) not in q:
    #         unexplored.append(legal_action)
    #
    # if len(unexplored) != 0:
    #     return random.choice(unexplored)

    p = epsilon_greedy(q, state, actions, N, constants)
    return random.choices(actions, weights=p)[0]

OUT_FILE="results.txt"
def write_results(constants, avg_length, avg_return):
    f = open(OUT_FILE, "a+")
    f.write(str(constants.const) + ' ' + str(constants.alpha) + ' ' + str(constants.gamma) + ' ' +  str(avg_length) + ' ' +  str(avg_return) + '\n')
    f.close()


def sarsa(env, states, actions, constants):
    q = {}
    N = {}
    for state in states:
        N[state] = 0
        # for action in actions:
        #     q[(state, action)] = Q0

    for episode in range(0, NO_EPISODES):
        # Set initial state
        resetEnv(env)
        state = State(env.agent_pos, env.agent_dir)
        # alege act,iunea a conform π (s, q)
        action = get_action(state, q, actions, N, constants)

        done = False
        while not done:
            N[state] = N[state] + 1
            obs, reward, done, info = env.step(action)
            new_state = State(env.agent_pos, env.agent_dir)
            new_action = get_action(state, q, actions, N, constants)
            q[(state, action)] = q.get((state, action), 0) + constants.alpha * (reward + constants.gamma * q.get((new_state, new_action), 0) - q.get((state, action), 0))

            state = new_state
            action = new_action
            # env.render('human')
            # time.sleep(0.01)

    recent_returns, recent_lengths = [], []
    for i in range(1, 100):
        resetEnv(env)
        crt_return = 0
        crt_length = 0
        state = State(env.agent_pos, env.agent_dir)
        # alege act,iunea a conform π (s, q)
        action = get_action(state, q, actions, N, constants)

        done = False
        while not done:
            N[state] = N[state] + 1
            obs, reward, done, info = env.step(action)
            crt_return += reward
            crt_length += 1
            new_state = State(env.agent_pos, env.agent_dir)
            new_action = get_action(state, q, actions, N, constants)
            q[(state, action)] = q.get((state, action), 0) + constants.alpha * (reward + constants.gamma * q.get((new_state, new_action), 0) - q.get((state, action), 0))

            state = new_state
            action = new_action
            # env.render('human')
            # time.sleep(0.01)
        recent_returns.append(crt_return)  # câștigul episodului încheiat
        recent_lengths.append(crt_length)

    avg_return = np.mean(recent_returns)  # media câștigurilor recente
    avg_length = np.mean(recent_lengths)  # media lungimilor episoadelor
    write_results(constants, avg_return, avg_length)


if __name__ == "__main__":
    main()


