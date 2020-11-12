import pickle
import random
import random
import sys

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from local_fel_simulated_env import FelLocalEnv
from pernaf.pernaf.naf import NAF
from pernaf.pernaf.utils.statistic import Statistic

# from pendulum import PendulumEnv as simpleEnv
# set random seed
from simulated_tango import SimTangoConnection

random_seed = 123
# set random seed
tf.set_random_seed(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)

tango = SimTangoConnection()
env = FelLocalEnv(tango=tango)

directory = "checkpoints/test_implementation/"


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

    fig, axs = plt.subplots(2, 1, constrained_layout=True)

    ax = axs[0]
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
    plt.savefig(label + '.pdf')
    # fig.tight_layout()
    plt.show()

    fig, axs = plt.subplots(1, 1)
    axs.plot(sum_rews)
    ax1 = plt.twinx(axs)
    ax1.plot(mean_rews, c='lime')
    plt.title(label)
    plt.show()


def plot_convergence(agent, label):
    losses, vs = agent.losses, agent.vs
    fig, ax = plt.subplots()
    ax.set_title(label)
    ax.set_xlabel('# steps')

    color = 'tab:blue'
    ax.semilogy(losses, color=color)
    ax.tick_params(axis='y', labelcolor=color)
    ax.set_ylabel('td_loss', color=color)
    # ax.set_ylim(0, 1)

    ax1 = plt.twinx(ax)
    # ax1.set_ylim(-2, 1)
    color = 'lime'

    ax1.set_ylabel('V', color=color)  # we already handled the x-label with ax1
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.plot(vs, color=color)
    plt.savefig(label + 'convergence' + '.pdf')
    plt.show()


if __name__ == '__main__':
    try:
        random_seed = int(sys.argv[2])
    except:
        random_seed = 25
    try:
        file_name = sys.argv[1] + '_' + str(random_seed)
    except:
        file_name = 'Data/17062020_' + str(random_seed) + '_'

    try:
        root_dir = sys.argv[3]
    except:
        root_dir = "PAPER/tests/"

    try:
        index = int(sys.argv[4])
        parameter_list = [
            dict(noise_function=lambda nr: max(0, 1 * (1 / (nr / 10 + 1))) if nr < 40 else 0),
            dict(noise_function=lambda nr: max(0, 0.5 * (1 / (nr / 10 + 1))) if nr < 40 else 0),
            dict(noise_function=lambda nr: max(0,  (1 / (nr / 10 + 1))) ),
            dict(noise_function=lambda nr: max(0, 0.5 * (1 / (nr / 10 + 1)))),

        ]
        noise_info = parameter_list[index]
        print(noise_info)
    except:
        noise_info = dict(noise_function=lambda nr: max(0, .1 * (1 / (nr / 10 + 1))) if nr < 5 else 0)

    batch_info = lambda nr: 15
    update_repeat = 20
    prio_info = dict(alpha=.1, beta_start=.9, beta_decay=lambda nr: max(1e-12, 1-nr/10))
    decay_info = lambda nr: max(0, 1-nr/10)


    tf.set_random_seed(random_seed)
    np.random.seed(random_seed)
    directory = root_dir + file_name + '/'

    discount = 0.999
    batch_size = 10
    learning_rate = 1e-3
    max_steps = 10000

    max_episodes = 10
    tau = 1 - 0.999

    is_train = True

    is_continued = not (is_train)

    nafnet_kwargs = dict(hidden_sizes=[32, 32], activation=tf.nn.tanh
                         , weight_init=tf.random_uniform_initializer(-0.05, 0.05),
                         batch_info=batch_info, decay_info=decay_info)
    # filename = 'Scan_data.obj'

    # filename = 'Scan_data.obj'
    # filehandler = open(filename, 'rb')
    # scan_data = pickle.load(filehandler)

    with tf.Session() as sess:
        # statistics and running the agent
        stat = Statistic(sess=sess, env_name=env.__name__, model_dir=directory,
                         max_update_per_step=update_repeat, is_continued=is_continued, save_frequency=100)
        # init the agent
        agent = NAF(sess=sess, env=env, stat=stat, discount=discount, batch_size=batch_size,
                    learning_rate=learning_rate, max_steps=max_steps, update_repeat=update_repeat,
                    max_episodes=max_episodes, tau=tau, pretune=None, prio_info=prio_info,
                    noise_info=noise_info, **nafnet_kwargs)
        # run the agent
        agent.run(is_train)

    # plot the results
    label = file_name
    plot_convergence(agent=agent, label=label)
    plot_results(env, label)

    if is_train:
        file_name += '_training'

        out_put_writer = open(root_dir + file_name + '.pkl', 'wb')
        out_rewards = env.rewards
        out_inits = env.initial_conditions
        out_losses, out_vs = agent.losses, agent.vs
        out_buffer = agent.replay_buffer

        pickle.dump(out_rewards, out_put_writer, -1)
        pickle.dump(out_inits, out_put_writer, -1)

        pickle.dump(out_losses, out_put_writer, -1)
        pickle.dump(out_vs, out_put_writer, -1)
        pickle.dump(out_buffer, out_put_writer, -1)
        out_put_writer.close()
