import os
import pickle
import pandas as pd
import random

import time

import gym
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

#from local_fel_simulated_env import FelLocalEnv
from laser_trajectory_control_env import LaserTrajectoryControlEnv
from naf2 import NAF
from pernaf.pernaf.utils.statistic import Statistic

# from simple_environment import simpleEnv

# from pendulum import PendulumEnv as simpleEnv
# set random seed
random_seed = 111
# set random seed

np.random.seed(random_seed)
random.seed(random_seed)

# from simulated_tango import SimTangoConnection
from tango_connection import TangoConnection

random_seed = 123
# set random seed
# tf.set_random_seed(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)

# tango = SimTangoConnection()
# env = FelLocalEnv(tango=tango)
conf_file = '/home/niky/FERMI/2020_07_20/configuration/conf_eos.json'
tango = TangoConnection(conf_file=conf_file)
env = LaserTrajectoryControlEnv(tango=tango)

directory = "checkpoints/test_implementation/"

label = 'New NAF'

directory = "checkpoints/new_test/"


# TODO: Test the loading

def plot_results(env, label):
    # plotting
    print('now plotting')
    rewards = env.rewards
    initial_states = env.initial_conditions

    iterations = []
    final_rews = []
    # starts = []
    sum_rews = []
    mean_rews = []
    # init_states = pd.read_pickle('/Users/shirlaen/PycharmProjects/DeepLearning/spinningup/Environments/initData')

    for i in range(len(rewards)):
        if (len(rewards[i]) > 0):
            final_rews.append(rewards[i][len(rewards[i]) - 1])
            # starts.append(-np.sqrt(np.mean(np.square(initial_states[i]))))
            iterations.append(len(rewards[i]))
            sum_rews.append(np.sum(rewards[i]))
            mean_rews.append(np.mean(rewards[i]))
    plot_suffix = ""  # f', number of iterations: {env.TOTAL_COUNTER}, Linac4 time: {env.TOTAL_COUNTER / 600:.1f} h'

    fig, axs = plt.subplots(2, 1)

    ax = axs[0]
    color = 'blue'
    ax.plot(iterations, c=color)
    ax.set_ylabel('steps', color=color)
    ax.tick_params(axis='y', labelcolor=color)
    ax1 = plt.twinx(ax)
    color = 'k'
    ax1.plot(np.cumsum(iterations), c=color)
    ax1.set_ylabel('cumulative steps', color=color)
    ax.set_title('Iterations' + plot_suffix)
    # fig.suptitle(label, fontsize=12)

    ax = axs[1]
    color = 'red'
    # ax.plot(starts, c=color)
    ax.set_ylabel('starts', color=color)
    ax.tick_params(axis='y', labelcolor=color)
    ax.set_title('Final reward per episode')  # + plot_suffix)
    ax.set_xlabel('Episodes (1)')

    ax1 = plt.twinx(ax)
    color = 'lime'
    ax1.set_ylabel('finals', color=color)  # we already handled the x-label with ax1
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.plot(final_rews, color=color)

    fig.tight_layout()
    plt.savefig(label + '.pdf')
    plt.show()

    fig, ax = plt.subplots(1, 1)
    color = 'blue'
    ax.plot(sum_rews, color)
    ax.set_ylabel('cum. reward', color=color)
    ax.tick_params(axis='y', labelcolor=color)
    ax1 = plt.twinx(ax)
    color = 'lime'
    ax1.plot(mean_rews, c=color)
    ax1.set_ylabel('mean', color=color)  # we already handled the x-label with ax1
    ax1.tick_params(axis='y', labelcolor=color)
    plt.show()


def plot_convergence(agent, label):
    losses, vs = agent.losses, agent.vs
    losses2, vs2 = agent.losses2, agent.vs2

    fig, ax = plt.subplots()
    ax.set_title(label)
    ax.set_xlabel('# steps')

    color = 'tab:blue'
    ax.semilogy(losses, color=color)
    ax.semilogy(losses2, color=color)
    ax.tick_params(axis='y', labelcolor=color)
    ax.set_ylabel('td_loss', color=color)
    # ax.set_ylim(0, 1)

    ax1 = plt.twinx(ax)
    # ax1.set_ylim(-2, 1)
    color = 'lime'
    ax1.set_ylabel('V', color=color)  # we already handled the x-label with ax1
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.plot(vs, color=color)
    ax1.plot(vs2, color=color)
    plt.savefig(label + 'convergence' + '.pdf')
    plt.show()


if __name__ == '__main__':
    # discount = 0.999
    # batch_size = 1000000
    # learning_rate = 1e-3
    max_steps = 500
    # update_repeat = 1
    max_episodes = 200  # 20  # for debugging
    # polyak = 0.999
    is_train = True
    is_continued = False if is_train else True

    nafnet_kwargs = dict(hidden_sizes=[100, 100], activation=tf.nn.tanh
                         , weight_init=tf.random_uniform_initializer(-0.05, 0.05, seed=random_seed))

    noise_info = dict(noise_function=lambda nr: max(0., 1 * (1 - (nr / 100))))

    # prio_info = dict(alpha=.25, beta=.8, decay_function=lambda nr: max(1e-6, (1 - (nr / 25))),
    #                  beta_decay=lambda nr: max(1e-6, (1 - (nr / 25))))
    prio_info = dict()
    # the target network is updated at the end of each episode
    # the number of episodes is executed each step in the environment
    #
    #training_info = dict(polyak=0.9995, epoches=5, steps_per_epoch=10, batch_size=100,
    #                     learning_rate=1e-3, discount=0.999)
    training_info = dict(polyak=0.9995, epoches=10, steps_per_epoch=10, batch_size=64,
                         learning_rate=1e-3, discount=0.999)
    # filename = 'Scan_data.obj'
    # filehandler = open(filename, 'rb')
    # scan_data = pickle.load(filehandler)

    # init the agent
    agent = NAF(env=env, directory=directory, max_steps=max_steps, max_episodes=max_episodes, prio_info=prio_info,
                noise_info=noise_info, is_continued=is_continued, q_smoothing=0.01, clipped_double_q=True,
                warm_up_steps=25, training_stop=100, save_frequency=500, **nafnet_kwargs)
    # run the agent
    agent.run(is_train)
    # agent.verification(steps=100)

    # plot the results
    plot_convergence(agent=agent, label=label)
    plot_results(env, label)
