import os
import pickle
import numpy as np
import matplotlib.pyplot as plt


def load_pickle_logging(file_name):
    directory = 'checkpoints/' + file_name + '/'
    files = []
    directory = directory + 'data/'
    for f in os.listdir(directory):
        if 'trajectory_data' in f and 'pkl' in f:
            files.append(f)
    files.sort()
    print(files[-1])

    with open(directory + files[-1], 'rb') as f:
        states = pickle.load(f)
        actions = pickle.load(f)
        rewards = pickle.load(f)
        dones = pickle.load(f)
    return states, actions, rewards, dones


def load_pickle_final(file_name):
    directory = 'checkpoints/' + file_name + '/'
    file = 'plot_data_0.pkl'

    with open(directory + file, 'rb') as f:
        rews = pickle.load(f)
        inits = pickle.load(f)
        losses = pickle.load(f)
        v_s = pickle.load(f)
    return rews, inits, losses, v_s


file_name = 'FEL_training_100_double_q_Tango_11'
states_0, actions_0, rewards_0, dones_0 = load_pickle_logging(file_name)
file_name = 'FEL_training_100_double_q_Tango_11_bis'
states_1, actions_1, rewards_1, dones_1 = load_pickle_logging(file_name)
rewards = [rewards_0, rewards_1]

file_name_s = 'FEL_training_100_single_q_Tango_11'
states_s0, actions_s0, rewards_s0, dones_s0 = load_pickle_logging(file_name_s)
file_name_s = 'FEL_training_100_single_q_Tango_11_bis'
states_s, actions_s, rewards_s, dones_s = load_pickle_logging(file_name_s)
rewards_s = [rewards_s, rewards_s0]


def read_rewards(rewards0):
    iterations_all = []
    final_rews_all = []
    mean_rews_all = []
    for k in range(len(rewards0)):
        rewards = rewards0[k]

        iterations = []
        final_rews = []
        mean_rews = []
        for i in range(len(rewards)):
            if len(rewards[i]) > 0:
                final_rews.append(rewards[i][len(rewards[i]) - 1])
                iterations.append(len(rewards[i]))
                try:
                    mean_rews.append(np.sum(rewards[i][1:]))
                except:
                    mean_rews.append([])
        iterations_all.append(iterations)
        final_rews_all.append(final_rews)
        mean_rews_all.append(mean_rews)

    iterations = np.mean(np.array(iterations_all), axis=0)
    final_rews = np.mean(np.array(final_rews_all), axis=0)
    mean_rews = np.mean(np.array(mean_rews_all), axis=0)
    return iterations, final_rews, mean_rews


def plot_results(rewards, rewards_s, plot_name):
    # plotting
    print('Now plotting')
    # rewards = env.rewards
    # initial_rewards = env.init_rewards
    # print('initial_rewards :', initial_rewards)

    iterations, final_rews, mean_rews = read_rewards(rewards)
    iterations_s, final_rews_s, mean_rews_s = read_rewards(rewards_s)

    plot_suffix = ""  # f', number of iterations: {env.TOTAL_COUNTER}, Linac4 time: {env.TOTAL_COUNTER / 600:.1f} h'
    fig, axs = plt.subplots(2, 1, sharex=True)

    ax = axs[0]
    ax.axvspan(0,100, alpha=0.2, color='coral')
    color = 'blue'
    ax.plot(iterations, c=color)
    ax.plot(iterations_s, c=color, ls=':')
    ax.set_ylabel('steps', color=color)
    ax.tick_params(axis='y', labelcolor=color)
    ax1 = plt.twinx(ax)
    color = 'k'
    ax1.plot(np.cumsum(iterations), c=color)
    ax1.plot(np.cumsum(iterations_s), c=color, ls=':')
    ax1.set_ylabel('cumulative steps', color=color)
    ax.set_title('Iterations' + plot_suffix)
    # fig.suptitle(label, fontsize=12)

    ax = axs[1]
    ax.axvspan(0, 100, alpha=0.2, color='coral')
    color = 'blue'
    # ax.plot(starts, c=color)
    ax.plot(mean_rews, c=color)
    ax.plot(mean_rews_s, c=color, ls=':')
    ax.set_ylabel('cum. return', color=color)
    # ax.axhline(-0.05, ls=':', color='r')
    ax.tick_params(axis='y', labelcolor=color)
    ax.set_title('Reward per episode')  # + plot_suffix)
    ax.set_xlabel('episodes')

    ax1 = plt.twinx(ax)
    color = 'lime'
    ax1.plot(final_rews[:-1], color=color)
    ax1.plot(final_rews_s[:-1], color=color, ls=':')

    ax1.set_ylabel('final return', color=color)
    ax1.axhline(-0.05, ls=':', color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    # fig.tight_layout()
    # plt.savefig(file_name + '_episodes.pdf')

    fig.align_labels()
    fig.tight_layout()
    # fig.suptitle('NonUniformImage class', fontsize='large')
    plt.savefig(plot_name + '_episodes.pdf')
    plt.show()

    # fig, ax = plt.subplots(1)
    # color = 'blue'
    # ax.plot(sum_rews, color)
    # ax.set_ylabel('cum. reward', color=color)
    # ax.set_xlabel('# episode')
    # ax.tick_params(axis='y', labelcolor=color)
    # ax1 = plt.twinx(ax)
    # color = 'lime'
    # ax1.plot(mean_rews, c=color)
    # ax1.set_ylabel('mean reward', color=color)  # we already handled the x-label with ax1
    # ax1.tick_params(axis='y', labelcolor=color)
    # # plt.savefig(file_name + '_rewards.pdf')
    # plt.show()


label = 'FERMI_all_experiments_NAF'
#
# plot_results(rewards, rewards_s, label)


def read_losses_v_s(losses0, v_s0, max_length):
    losses_all = []
    v_s_all = []
    for k in range(len(losses0)):

        losses = losses0[k]
        print(len(losses))
        v_s = v_s0[k]
        losses_all.append(losses[:max_length])
        v_s_all.append(v_s[:max_length])
    losses = np.mean(losses_all, axis=0)
    v_s = np.mean(v_s_all, axis=0)
    return losses, v_s


file_name = 'FEL_training_100_double_q_Tango_11'
rews0, inits0, losses0, v_s0 = load_pickle_final(file_name)
file_name = 'FEL_training_100_double_q_Tango_11_bis'
rews1, inits1, losses1, v_s1 = load_pickle_final(file_name)
losses, v_s = read_losses_v_s([losses0, losses1], [v_s0, v_s1], 691)
rewards = [rews0, rews1]

file_name = 'FEL_training_100_single_q_Tango_11'
rews0, inits0, losses0, v_s0 = load_pickle_final(file_name)
file_name = 'FEL_training_100_single_q_Tango_11_bis'
rews1, inits1, losses1, v_s1 = load_pickle_final(file_name)
losses_s, v_s_s = read_losses_v_s([losses0, losses1], [v_s0, v_s1], 691)

rewards_s = [rews0, rews1]


def plot_convergence(losses, v_s, losses_s, v_s_s, label):
    fig, ax = plt.subplots()
    ax.set_title(label)
    ax.set_xlabel('steps')

    color = 'tab:blue'
    ax.semilogy(losses, color=color)
    ax.semilogy(losses_s, color=color, ls=':')
    ax.tick_params(axis='y', labelcolor=color)
    ax.set_ylabel('td_loss', color=color)
    # ax.set_ylim(0, 1)

    ax1 = plt.twinx(ax)
    # ax1.set_ylim(-2, 1)
    color = 'lime'

    ax1.set_ylabel('V', color=color)  # we already handled the x-label with ax1
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.plot(v_s, color=color)
    ax1.plot(v_s_s, color=color, ls=':')
    plt.savefig(label + 'convergence' + '.pdf')
    plt.show()

plot_convergence(losses, v_s, losses_s, v_s_s, label)

label = 'FERMI_all_experiments_NAF'

plot_results(rewards, rewards_s, label)