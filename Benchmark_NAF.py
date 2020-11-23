import matplotlib.pyplot as plt
import numpy as np
import pickle

main_dir = 'checkpoints'


def read_rews(rewards):
    iterations = []
    final_rews = []
    sum_rews = []
    mean_rews = []

    for i in range(len(rewards)):
        if (len(rewards[i]) > 0):
            final_rews.append(rewards[i][len(rewards[i]) - 1])
            iterations.append(len(rewards[i]))
            sum_rews.append(np.sum(rewards[i]))
            mean_rews.append(np.mean(rewards[i]))
    return iterations, final_rews, sum_rews, mean_rews


def plot_results(rewards_list, file_name=None):
    fig, ax = plt.subplots(1, 1)
    # ax1 = plt.twinx(ax)

    for rewards in rewards_list:
        iterations, final_rews, sum_rews, mean_rews = read_rews(rewards)
        ax.plot(sum_rews)
        # ax1.plot(final_rews)
    # color = 'lime'
    # ax1.set_ylabel('finals', color=color)
    # ax1.tick_params(axis='y', labelcolor=color)

    ax.set_ylabel('cum. reward')
    ax.set_xlabel('# episode')
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

def load_pickle_final(test_name):
    directory = 'checkpoints/' + test_name + '/'
    file = 'plot_data_0.pkl'

    with open(directory + file, 'rb') as f:
        rews = pickle.load(f)
        losses = pickle.load(f)
        v_s = pickle.load(f)
    return rews, losses, v_s

reward_list = []
rews, losses, v_s = load_pickle_final('pendulum_video')
reward_list.append(rews)
rews, losses, v_s = load_pickle_final('pendulum_video2')
# rews = rews[25:]
reward_list.append(rews)

plot_results(rewards_list=reward_list)
