import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
# import PyQt5
from spinup import td3_tf1 as td3
from spinup import sac_tf1 as sac
from spinup.algos.tf1.ppo.ppo import ppo
from spinup.algos.tf1.trpo.trpo import trpo

from local_fel_simulated_env import FelLocalEnv
# from pendulum import PendulumEnv as simpleEnv
# set random seed
from simulated_tango import SimTangoConnection

random_seed = 111
# set random seed
tf.set_random_seed(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)

tango = SimTangoConnection()
env = FelLocalEnv(tango=tango)
env_fn = lambda: env

label = 'Sim_Tango'
directory = "checkpoints/test_implementation/"


def plot_results(env, label='def'):
    # plotting
    print('now plotting')
    rewards = env.rewards
    initial_states = env.initial_conditions

    iterations = []
    final_rews = []
    # starts = []
    sum_rews=[]
    mean_rews = []
    # init_states = pd.read_pickle('/Users/shirlaen/PycharmProjects/DeepLearning/spinningup/Environments/initData')

    for i in range(len(rewards)):
        if (len(rewards[i]) > 0):
            final_rews.append(rewards[i][len(rewards[i]) - 1])
            # starts.append(-np.sqrt(np.mean(np.square(initial_states[i]))))
            iterations.append(len(rewards[i]))
            sum_rews.append(np.sum(rewards[i]))
            mean_rews.append(np.mean(rewards[i]))
    plot_suffix = ""#f', number of iterations: {env.TOTAL_COUNTER}, Linac4 time: {env.TOTAL_COUNTER / 600:.1f} h'

    fig, axs = plt.subplots(2, 1, constrained_layout=True)

    ax=axs[0]
    ax.plot(iterations)
    ax.set_title('Iterations' + plot_suffix)
    fig.suptitle(label, fontsize=12)

    ax = axs[1]
    ax.plot(final_rews, 'r--')

    ax.set_title('Final reward per episode')  # + plot_suffix)
    ax.set_xlabel('Episodes (1)')

    # ax1 = plt.twinx(ax)
    # color = 'lime'
    # ax1.set_ylabel('starts', color=color)  # we already handled the x-label with ax1
    # ax1.tick_params(axis='y', labelcolor=color)
    # ax1.plot(starts, color=color)
    plt.savefig(label+'.pdf')
    # fig.tight_layout()
    plt.show()


    fig, axs = plt.subplots(1, 1)
    axs.plot(sum_rews)
    ax1 = plt.twinx(axs)
    ax1.plot(mean_rews,c='lime')
    plt.show()

output_dir = 'logging/debug/'

logger_kwargs = dict(output_dir=output_dir, exp_name='niky')
#

nafnet_kwargs = dict(hidden_sizes=[100, 100])
agent = sac(env_fn=env_fn, epochs=50, steps_per_epoch=2000,
            logger_kwargs=logger_kwargs, start_steps=2000, seed=random_seed)

# agent = ppo(env_fn=env_fn, epochs=50, steps_per_epoch=5000, ac_kwargs=nafnet_kwargs,
#             logger_kwargs=logger_kwargs, gamma=0.9999, save_freq=10000)


plot_name = 'Stats'
name = plot_name
data = pd.read_csv(output_dir + '/progress.txt', sep="\t")

data.index = data['TotalEnvInteracts']
data_plot = data[['EpLen', 'MinEpRet', 'AverageEpRet']]
data_plot.plot(secondary_y=['MinEpRet', 'AverageEpRet'])

plt.title(name)
# plt.savefig(name + '.pdf')
plt.show()

plot_results(env)