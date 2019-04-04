from concurrent.futures import ThreadPoolExecutor

import sarsa_skel


executor = ThreadPoolExecutor(max_workers=4)


# epsilon
alpha_values = [0.1, 0.1, 0.3, 0.3, 0.5, 0.5, 0.7, 0.7, 0.9, 0.9]
gamma_values = [0.9, 0.9, 0.7, 0.7, 0.5, 0.5, 0.3, 0.3, 0.1, 0.1]
constants    = [10,  100, 50,  200, 40,  500, 20,  300, 60,  250]
maps = ['MiniGrid-Empty-6x6-v0', 'MiniGrid-Empty-8x8-v0', 'MiniGrid-Empty-16x16-v0',
        'MiniGrid-DoorKey-6x6-v0', 'MiniGrid-DoorKey-8x8-v0', 'MiniGrid-DoorKey-16x16-v0']
for map_name in maps:
    options = {}
    for i in range(0, len(alpha_values)):
        alpha = alpha_values[i]
        gamma = gamma_values[i]
        c = constants[i]
        options['alpha'] = alpha
        options['gamma'] = gamma
        options['constant'] = c
        options['exploration'] = 'epsilon'
        options['env_name'] = map_name

        executor.submit(sarsa_skel.start_sarsa(options))


# boltzman
alpha_values = [0.1, 0.3, 0.5, 0.7, 0.9]
gamma_values = [0.9, 0.7, 0.5, 0.3, 0.1]
maps = ['MiniGrid-Empty-6x6-v0', 'MiniGrid-Empty-8x8-v0', 'MiniGrid-Empty-16x16-v0',
        'MiniGrid-DoorKey-6x6-v0', 'MiniGrid-DoorKey-8x8-v0', 'MiniGrid-DoorKey-16x16-v0']
for map_name in maps:
    options = {}
    for i in range(0, len(alpha_values)):
        alpha = alpha_values[i]
        gamma = gamma_values[i]
        c = 0
        options['alpha'] = alpha
        options['gamma'] = gamma
        options['constant'] = c
        options['exploration'] = 'boltzman'
        options['env_name'] = map_name

        executor.submit(sarsa_skel.start_sarsa(options))



