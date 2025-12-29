import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
project_directory = 'Data_logging/run_test/-nr_steps_25-cr_lr0.001-crit_it_80-d_0.05-conj_iters_10-n_ep_1-mini_bs_500' \
                    '-m_bs_100-mb_lr_0.0005-sim_steps_2000-m_iter_15-ensnr_5-init_100/'
project_directory = 'Data_logging/ME_TRPO_stable/2020_10_06_ME_TRPO_stable@FERMI/run2/'
# for file in os.listdir(project_directory):
#     filename = file
# print(filename)
# # filename = '18_17_49_35_default'
# filehandler = open(project_directory + filename, 'rb')
# object = pickle.load(filehandler)

filenames = []
for file in os.listdir(project_directory):
    if 'final' in file:
        filenames.append(file)

filenames.sort()

# filename = '09_25_19_18_04_training_observables'
filename = filenames[-1]
print(filename)

filehandler = open(project_directory + filename, 'rb')
object = pickle.load(filehandler)

def plot_results(data, label='Verification', **kwargs):
    # plotting
    print('now plotting')
    rewards = data
    print(rewards)

    # initial_states = env.initial_conditions

    iterations = []
    finals = []
    means = []

    # init_states = pd.read_pickle('/Users/shirlaen/PycharmProjects/DeepLearning/spinningup/Environments/initData')

    for i in range(len(rewards)):
        if (len(rewards[i]) > 1):
            # finals.append(rewards[i][len(rewards[i]) - 1])
            finals.append(rewards[i][-1])
            means.append(np.sum(rewards[i][1:]))
            iterations.append(len(rewards[i]))
    print(finals)
    plot_suffix = label  # , Fermi time: {env.TOTAL_COUNTER / 600:.1f} h'

    fig, axs = plt.subplots(2, 1, sharex=True)

    ax = axs[0]
    ax.plot(iterations)
    ax.set_ylabel('steps')
    ax.set_title(plot_suffix)
    # fig.suptitle(label, fontsize=12)

    ax = axs[1]
    color = 'blue'
    ax.set_ylabel('Final reward', color=color)  # we already handled the x-label with ax1
    ax.tick_params(axis='y', labelcolor=color)
    ax.plot(finals, color=color)

    ax.set_title('Final reward per episode')  # + plot_suffix)
    ax.axhline(y=-0.05, c='blue', ls=':')
    ax.set_xlabel('Episodes (1)')

    ax1 = plt.twinx(ax)
    color = 'lime'
    ax1.set_ylabel('Cumulative reward', color=color)  # we already handled the x-label with ax1
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.plot(means, color=color)

    # ax.set_ylim(ax1.get_ylim())
    if 'save_name' in kwargs:
        plt.savefig(kwargs.get('save_name') + '.pdf')
    # fig.tight_layout()
    fig.align_labels()
    fig.tight_layout()
    plt.show()

plot_results(data=object['rews'])
filenames = []
for file in os.listdir(project_directory):
    if 'training_observables' in file:
        filenames.append(file)

filenames.sort()

# filename = '09_25_19_18_04_training_observables'
filename = filenames[-1]
print(filename)

filehandler = open(project_directory + filename, 'rb')
object = pickle.load(filehandler)

sim_rewards_all = object['sim_rewards_all'][0]
entropy_all=object['entropy_all']
step_counts_all=object['step_counts_all']
batch_rews_all=object['batch_rews_all'][0]
tests_all = object['tests_all'][0]

fig, axs = plt.subplots(2, 1, sharex=True)
x = np.arange(len(batch_rews_all))
ax = axs[0]
ax.step(x, batch_rews_all)
ax.set_ylabel('rews per batch')
# plt.tw
ax2 = ax.twinx()

color = 'lime'
ax2.set_ylabel('data points', color=color)  # we already handled the x-label with ax1
ax2.tick_params(axis='y', labelcolor=color)
ax2.step(x, step_counts_all, color=color)

ax = axs[1]
ax.plot(sim_rewards_all, ls=':')
ax.plot(tests_all)
ax.set_ylabel('rewards model')
# plt.tw
ax2 = ax.twinx()

color = 'lime'
ax2.set_ylabel(r'- log(std($p_\pi$))', color=color)  # we already handled the x-label with ax1
ax2.tick_params(axis='y', labelcolor=color)
ax2.plot(entropy_all, color=color)

fig.align_labels()
fig.tight_layout()
fig.show()
