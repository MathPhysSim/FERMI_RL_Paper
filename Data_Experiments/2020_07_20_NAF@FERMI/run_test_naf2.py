import os
import pickle
import random
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# from local_fel_simulated_env import FelLocalEnv
from laser_trajectory_control_env import LaserTrajectoryControlEnv
# from simulated_tango import SimTangoConnection
from tango_connection import TangoConnection
from naf2 import NAF

# set random seed
random_seed = 123
np.random.seed(random_seed)
random.seed(random_seed)

# tango = SimTangoConnection()
# env = FelLocalEnv(tango=tango)
conf_file = '/home/niky/FERMI/2020_07_20/configuration/conf_fel.json'
tango = TangoConnection(conf_file=conf_file)
env = LaserTrajectoryControlEnv(tango=tango)

def plot_results(env, file_name):
    # plotting
    print('Now plotting')
    rewards = env.rewards
    initial_rewards = env.init_rewards
    # print('initial_rewards :', initial_rewards)

    iterations = []
    final_rews = []
    starts = []
    sum_rews = []
    mean_rews = []

    for i in range(len(rewards)):
        if (len(rewards[i]) > 0):
            final_rews.append(rewards[i][len(rewards[i]) - 1])
            starts.append(initial_rewards[i])
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
    ax.plot(starts, c=color)
    ax.set_ylabel('starts', color=color)
    ax.axhline(-0.05, ls=':', color='r')
    ax.tick_params(axis='y', labelcolor=color)
    ax.set_title('Final reward per episode')  # + plot_suffix)
    ax.set_xlabel('# episode')

    ax1 = plt.twinx(ax)
    color = 'lime'
    ax1.set_ylabel('finals', color=color)  # we already handled the x-label with ax1
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.plot(final_rews, color=color)

    fig.tight_layout()
    plt.savefig(file_name + '_episodes.pdf')
    plt.show()

    fig, ax = plt.subplots(1, 1)
    color = 'blue'
    ax.plot(sum_rews, color)
    ax.set_ylabel('cum. reward', color=color)
    ax.set_xlabel('# episode')
    ax.tick_params(axis='y', labelcolor=color)
    ax1 = plt.twinx(ax)
    color = 'lime'
    ax1.plot(mean_rews, c=color)
    ax1.set_ylabel('mean reward', color=color)  # we already handled the x-label with ax1
    ax1.tick_params(axis='y', labelcolor=color)
    plt.savefig(file_name + '_rewards.pdf')
    plt.show()

def plot_convergence(agent, file_name):
    losses, vs = agent.losses, agent.vs
    # losses2, vs2 = agent.losses2, agent.vs2

    fig, ax = plt.subplots()
    # ax.set_title(label)
    ax.set_xlabel('# steps')

    color = 'tab:blue'

    # ax.semilogy(losses2, color=color)
    ax.tick_params(axis='y', labelcolor=color)
    ax.set_ylabel('td_loss', color=color)
    ax.semilogy(losses, color=color)
    # ax.set_ylim(0, 1)

    ax1 = plt.twinx(ax)
    # ax1.set_ylim(-2, 1)
    color = 'lime'
    ax1.set_ylabel('V', color=color)  # we already handled the x-label with ax1
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.plot(vs, color=color)
    # ax1.plot(vs2, color=color)
    plt.savefig(file_name + '_convergence' + '.pdf')
    plt.show()

if __name__ == '__main__':

    directory = "checkpoints/fel_run_0/"

    is_continued = True  # False  # False if is_train else True

    nafnet_kwargs = dict(hidden_sizes=[100, 100], activation=tf.nn.tanh
                         , weight_init=tf.random_uniform_initializer(-0.05, 0.05, seed=random_seed))

    noise_info = dict(noise_function=lambda nr: max(0., 1 * (1 - (nr / 50))))

    # the target network is updated at the end of each episode
    # the number of episodes is executed each step in the environment
    training_info = dict(polyak=0.9995, epochs=2, steps_per_epoch=5, batch_size=100,
                         learning_rate=1e-3, discount=0.999)

    # init the agent
    agent = NAF(env=env, directory=directory, noise_info=noise_info,
                is_continued=is_continued, q_smoothing=0.05, clipped_double_q=True,
                training_info=training_info, save_frequency=25,
                **nafnet_kwargs)

    # run the agent training
    # agent.training(warm_up_steps=50, initial_episode_length=10, max_episodes=50, max_steps=500)
    agent.training(warm_up_steps=0, initial_episode_length=10, max_episodes=25, max_steps=500)
    # run the agent verification
    agent.verification(max_episodes=30, max_steps=20)

    # plot the results
    files = []
    for f in os.listdir(directory):
        if 'plot_data' in f and 'pkl' in f:
            files.append(f)
    print(files)
    if len(files) > 0:
        file_name = directory + f'plot_data_{len(files)}'
    else:
        file_name = directory + 'plot_data_0'

    plot_convergence(agent=agent, file_name=file_name)
    plot_results(env, file_name=file_name)

    out_put_writer = open(file_name + '.pkl', 'wb')
    out_rewards = env.rewards
    out_inits = env.initial_conditions
    out_losses, out_vs = agent.losses, agent.vs

    pickle.dump(out_rewards, out_put_writer, -1)
    pickle.dump(out_inits, out_put_writer, -1)

    pickle.dump(out_losses, out_put_writer, -1)
    pickle.dump(out_vs, out_put_writer, -1)
    out_put_writer.close()
