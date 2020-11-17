'''This script generates the plots for the AE-DYNA tests at FERMI FEL'''

import os
import pickle
import matplotlib.pyplot as plt
import numpy as np

label = "ME-TRPO"
# label = "AE-DYNA"
if label == "ME-TRPO":
    # ME-TRPO results
    project_directory = 'Data_Experiments/2020_10_06_ME_TRPO_stable@FERMI/run2/'
else:
    # AE-Dyna results
    project_directory = 'Data_Experiments/2020_11_05_AE_Dyna@FERMI/-nr_steps_25-cr_lr-n_ep_13-m_bs_100-sim_steps_3000-m_iter_35-ensnr_3-init_200/'

def read_rewards(rewards):
    iterations_all = []
    final_rews_all = []
    mean_rews_all = []
    stds = []



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
            stds.append(np.std(rewards[i][1:]))

    # iterations = np.mean(np.array(iterations_all), axis=0)
    # final_rews = np.mean(np.array(final_rews_all), axis=0)
    # mean_rews = np.mean(np.array(mean_rews_all), axis=0)

    return np.array(iterations), np.array(final_rews), np.array(mean_rews), np.array(stds)



def plot_results(data, label='Verification', **kwargs):
        '''plotting'''
        rewards = data['rews']
        # iterations = []
        # finals = []
        # means = []
        # stds = []
        #
        # for i in range(len(rewards)):
        #     if (len(rewards[i]) > 1):
        #         finals.append(rewards[i][-1])
        #         means.append(np.mean(rewards[i][1:]))
        #         stds.append(np.std(rewards[i][1:]))
        #         iterations.append(len(rewards[i]))
        #
        # x = range(len(iterations))
        # iterations = np.array(iterations)
        # finals = np.array(finals)
        # means = np.array(means)
        # stds = np.array(stds)

        iterations, finals, means, stds = read_rewards(rewards)
        plot_suffix = label  # , Fermi time: {env.TOTAL_COUNTER / 600:.1f} h'

        fig, axs = plt.subplots(2, 1, sharex=True)

        ax = axs[0]
        x = range(len(iterations))
        ax.plot(x, iterations)
        ax.set_ylabel('Iterations (1)')
        ax.set_title(plot_suffix)
        # fig.suptitle(label, fontsize=12)
        if 'data_number' in kwargs:
            ax1 = plt.twinx(ax)
            color = 'lime'
            ax1.set_ylabel('Mean reward', color=color)  # we already handled the x-label with ax1
            ax1.tick_params(axis='y', labelcolor=color)
            ax1.plot(x, kwargs.get('data_number'), color=color)

        ax = axs[1]
        color = 'blue'
        ax.set_ylabel('Final reward', color=color)  # we already handled the x-label with ax1
        ax.tick_params(axis='y', labelcolor=color)
        ax.plot(x, finals, color=color)

        ax.set_title('Final reward per episode')  # + plot_suffix)
        ax.set_xlabel('Episodes (1)')

        ax1 = plt.twinx(ax)
        color = 'lime'
        ax1.set_ylabel('Mean reward', color=color)  # we already handled the x-label with ax1
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.fill_between(x, means - stds, means + stds,
                         alpha=0.5, edgecolor=color, facecolor='#FF9848')
        ax1.plot(x, means, color=color)
        fig.align_labels()
        # ax.set_ylim(ax1.get_ylim())
        if 'save_name' in kwargs:
            plt.savefig(kwargs.get('save_name') + '.pdf')
            plt.savefig(kwargs.get('save_name') + '.png')
        plt.show()

def plot_observables(data, label='Experiment', **kwargs):
    """plot observables during the test"""

    sim_rewards_all = np.array(data.get('sim_rewards_all'))
    step_counts_all = np.array(data.get('step_counts_all'))
    batch_rews_all = np.array(data.get('batch_rews_all'))
    tests_all = np.array(data.get('tests_all'))
    length_all = object['entropy_all']

    fig, axs = plt.subplots(2, 1, sharex=True)
    x = np.arange(len(batch_rews_all[0]))
    ax = axs[0]
    ax.step(x, batch_rews_all[0])
    ax.fill_between(x, batch_rews_all[0] - batch_rews_all[1], batch_rews_all[0] + batch_rews_all[1],
                    alpha=0.5)
    ax.set_ylabel('rews per batch')

    ax.set_title(label)

    ax2 = ax.twinx()

    color = 'lime'
    ax2.set_ylabel('data points', color=color)  # we already handled the x-label with ax1
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.step(x, step_counts_all, color=color)

    ax = axs[1]
    ax.plot(sim_rewards_all[0], ls=':')
    ax.fill_between(x, sim_rewards_all[0] - sim_rewards_all[1], sim_rewards_all[0] + sim_rewards_all[1],
                    alpha=0.5)
    try:
        ax.plot(tests_all[0])
        ax.fill_between(x, tests_all[0] - tests_all[1], tests_all[0] + tests_all[1],
                        alpha=0.5)
        ax.axhline(y=np.max(tests_all[0]), c='orange')
    except:
        pass
    ax.set_ylabel('rewards tests')
    # plt.tw
    ax.grid(True)
    if length_all:
        ax2 = ax.twinx()
        color = 'lime'
        ax2.set_ylabel(r'- log(std($p_\pi$))', color=color)   # we already handled the x-label with ax1
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.plot(length_all, color=color)
    fig.align_labels()

    if 'save_name' in kwargs:
        plt.savefig(kwargs.get('save_name') + '.pdf')
        plt.savefig(kwargs.get('save_name') + '.png')
    plt.show()

# plot verification

filenames = []
for file in os.listdir(project_directory):
    if 'final' in file:
        filenames.append(file)

filenames.sort()

filename = filenames[-1]
print(filename)

filehandler = open(project_directory + filename, 'rb')
object = pickle.load(filehandler)
save_name = 'Figures/' + label+'_verification'
plot_results(object,label=label, save_name=save_name)

# plot observables

filenames = []
for file in os.listdir(project_directory):
    if 'training_observables' in file:
        filenames.append(file)

filenames.sort()

filename = filenames[-1]
print(filename)

filehandler = open(project_directory + filename, 'rb')
object = pickle.load(filehandler)
save_name = 'Figures/' + label+'_observables'
plot_observables(object, label=label, save_name=save_name)

