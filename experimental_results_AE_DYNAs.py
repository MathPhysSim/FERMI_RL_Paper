'''This script generates the plots for the AE-DYNA tests at FERMI FEL'''

import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

label = ["ME-TRPO", "AE-DYNA", 'Simulation'][0]

if label == "ME-TRPO":
    # ME-TRPO results
    project_directory = 'Data_Experiments/2020_10_06_ME_TRPO_stable@FERMI/run2/'
elif label == "AE-DYNA":
    # AE-Dyna results
    project_directory = 'Data_Experiments/2020_11_05_AE_Dyna@FERMI/-nr_steps_25-cr_lr-n_ep_13-m_bs_100-sim_steps_3000-m_iter_35-ensnr_3-init_200/'
else:
    project_directory = 'Data_logging/temp/-nr_steps_25-cr_lr-n_ep_7-m_bs_100-sim_steps_2500-m_iter_30-ensnr_5-init_100/'


def read_rewards(rewards):
    iterations_all = []
    final_rews_all = []
    mean_rews_all = []
    stds = []

    iterations = []
    final_rews = []
    mean_rews = []
    for i in range(len(rewards)):
        current_rewards = rewards[i][1:]
        if len(current_rewards) > 0:
            # print(current_rewards)
            final_rews.append(rewards[i][-1])
            iterations.append(len(current_rewards))
            try:
                mean_rews.append(np.sum(current_rewards))
            except:
                mean_rews.append([])
            stds.append(np.std(current_rewards))

    # iterations = np.mean(np.array(iterations_all), axis=0)
    # final_rews = np.mean(np.array(final_rews_all), axis=0)
    # mean_rews = np.mean(np.array(mean_rews_all), axis=0)

    return np.array(iterations), np.array(final_rews), np.array(mean_rews), np.array(stds)


# def plot_results(data, label='Verification', **kwargs):
#         '''plotting'''
#         rewards = data['rews']
#         # iterations = []
#         # finals = []
#         # means = []
#         # stds = []
#         #
#         # for i in range(len(rewards)):
#         #     if (len(rewards[i]) > 1):
#         #         finals.append(rewards[i][-1])
#         #         means.append(np.mean(rewards[i][1:]))
#         #         stds.append(np.std(rewards[i][1:]))
#         #         iterations.append(len(rewards[i]))
#         #
#         # x = range(len(iterations))
#         # iterations = np.array(iterations)
#         # finals = np.array(finals)
#         # means = np.array(means)
#         # stds = np.array(stds)
#
#         iterations, finals, means, stds = read_rewards(rewards)
#         plot_suffix = label  # , Fermi time: {env.TOTAL_COUNTER / 600:.1f} h'
#
#         fig, axs = plt.subplots(2, 1, sharex=True)
#
#         ax = axs[0]
#         x = range(len(iterations))
#         ax.plot(x, iterations)
#         ax.set_ylabel('no. iterations')
#         ax.set_title(plot_suffix)
#         # fig.suptitle(label, fontsize=12)
#         if 'data_number' in kwargs:
#             ax1 = plt.twinx(ax)
#             color = 'lime'
#             ax1.set_ylabel('Mean reward', color=color)  # we already handled the x-label with ax1
#             ax1.tick_params(axis='y', labelcolor=color)
#             ax1.plot(x, kwargs.get('data_number'), color=color)
#
#         ax = axs[1]
#         color = 'blue'
#         ax.set_ylabel('Final reward', color=color)  # we already handled the x-label with ax1
#         ax.tick_params(axis='y', labelcolor=color)
#         ax.plot(x, finals, color=color)
#
#         # ax.set_title('final reward per episode (arb. units)')  # + plot_suffix)
#         ax.set_xlabel('no. episodes')
#
#         ax1 = plt.twinx(ax)
#         color = 'lime'
#         ax1.set_ylabel('cum. reward (arb. units)', color=color)  # we already handled the x-label with ax1
#         ax1.tick_params(axis='y', labelcolor=color)
#         ax1.fill_between(x, means - stds, means + stds,
#                          alpha=0.5, edgecolor=color, facecolor='#FF9848')
#         ax1.plot(x, means, color=color)
#         fig.align_labels()
#         # ax.set_ylim(ax1.get_ylim())
#         if 'save_name' in kwargs:
#             plt.savefig(kwargs.get('save_name') + '.pdf')
#             plt.savefig(kwargs.get('save_name') + '.png')
#         plt.show()
def plot_results(data, label=None, **kwargs):
    rewards = data['rews']
    iterations, final_rews, mean_rews, _ = read_rewards(rewards)
    # iterations_s, final_rews_s, mean_rews_s = read_rewards(rewards_single, data_range_in=data_range_in)

    plot_suffix = ""  # f', number of iterations: {env.TOTAL_COUNTER}, Linac4 time: {env.TOTAL_COUNTER / 600:.1f} h'
    fig, axs = plt.subplots(2, 1, sharex=True)

    ax = axs[0]
    # ax.axvspan(0, 100, alpha=0.2, color='coral')
    color = 'blue'
    ax.plot(iterations, c=color)
    # ax.plot(iterations_s, c=color, ls=':')
    ax.set_ylabel('no. iterations', color=color)
    ax.tick_params(axis='y', labelcolor=color)
    ax.set_ylim(1,10)
    ax1 = plt.twinx(ax)
    color = 'k'
    ax1.plot(np.cumsum(iterations), c=color)
    # ax1.plot(np.cumsum(iterations_s), c=color, ls=':')
    # print(np.cumsum(iterations)[-1])
    ax1.set_ylabel('no. cumulative steps', color=color)
    ax.set_title(label)
    # fig.suptitle(label, fontsize=12)
    ax1.set_ylim(0, 275)
    ax = axs[1]
    # ax.axvspan(0, 100, alpha=0.2, color='coral')
    color = 'blue'
    # ax.plot(starts, c=color)
    ax.plot(mean_rews, c=color)
    # ax.plot(mean_rews_s, c=color, ls=':')
    ax.set_ylabel('cum. return (arb. units)', color=color)
    # ax.axhline(-0.05, ls=':', color='r')
    ax.tick_params(axis='y', labelcolor=color)
    # ax.set_title('reward per episode (arb. units)')  # + plot_suffix)
    ax.set_xlabel('no. episodes')

    ax1 = plt.twinx(ax)
    color = 'lime'
    ax1.plot(final_rews[:-1], color=color)
    # ax1.plot(final_rews_s[:-1], color=color, ls=':')

    ax1.set_ylabel('final return (arb. units)', color=color)
    ax1.axhline(-0.05, ls=':', color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    fig.align_labels()
    fig.tight_layout()
    # fig.suptitle('NonUniformImage class', fontsize='large')
    if 'save_name' in kwargs:
        save_name = kwargs.get('save_name')
        plt.savefig(save_name + '_verification.pdf')
        plt.savefig(save_name + '_verification.png')
    plt.show()

def plot_results_verification(data_array, label=None, **kwargs):
    if 'data_range_in' in kwargs:
        data_range_in = kwargs.get('data_range_in')
    else:
        data_range_in = None

    rewards = data_array[0]['rews']
    iterations, final_rews, mean_rews, _ = read_rewards(rewards)

    rewards = data_array[1]['rews']
    iterations_s, final_rews_s, mean_rews_s, iter_all_s = read_rewards(rewards)
    plot_suffix = ""  # f', number of iterations: {env.TOTAL_COUNTER}, Linac4 time: {env.TOTAL_COUNTER / 600:.1f} h'
    fig, axs = plt.subplots(1, 1, sharex=True)
    fig.set_size_inches(fig.get_size_inches()[0],fig.get_size_inches()[1]*0.6)
    ax = axs
    # ax.axvspan(0, 100, alpha=0.2, color='coral')
    color = 'blue'
    x = np.arange(0, len(iterations))
    ax.plot(iterations, c=color)
    # ax.fill_between(x, iterations[0] - iterations[1], iterations[0] + iterations[1], color=color, alpha=0.1)
    # color = 'orange'
    x = np.arange(0, len(iterations_s))
    ax.plot(iterations_s, c=color, ls=':')
    # ax.fill_between(x, iterations_s[0] - iterations_s[1], iterations_s[0] + iterations_s[1], color=color, alpha=0.1)

    ax.set_ylabel('no. iterations', color=color)
    ax.tick_params(axis='y', labelcolor=color)
    ax.set_ylim(0, 10)
    ax1 = plt.twinx(ax)
    if 'verification' in kwargs:
        verification = kwargs.get('verification')
        if verification:
            ax1.set_ylim(0, 275)
            ax.set_ylim(0, 10)
    color = 'lime'
    iterations_mean = np.cumsum(iterations)
    ax1.plot(iterations_mean, c=color)
    x = np.arange(0, len(iterations_mean))
    # ax1.fill_between(x, iterations_mean-iterations_std, iterations_mean
    #                  + iterations_std, color=color, alpha=0.1)

    iterations_mean = np.cumsum(iterations_s)
    ax1.plot(iterations_mean, c=color, ls=':')
    x = np.arange(0, len(iterations_mean))
    # ax1.fill_between(x, iterations_mean-iterations_std, iterations_mean
    #                  + iterations_std, color=color, alpha=0.1)

    ax1.set_ylabel('no. cumulative steps', color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax.set_title(label)
    ax.set_xlabel('no. episodes')
    # fig.suptitle(label, fontsize=12)
    import matplotlib.lines as mlines

    single_line = mlines.Line2D([], [], color='k', ls=':', label='AE-DYNA')
    double_line = mlines.Line2D([], [], color='k', ls='-', label='ME-TRPO')
    plt.legend(handles=[single_line, double_line])

    # import pandas as pd
    # df = pd.DataFrame([iterations[:10], iterations[-10:]], index=['initial','final']).T
    # df_d =df.describe().T[['mean']]
    # df = pd.DataFrame([iterations_s[:10], iterations_s[-10:]], index=['initial','final']).T
    # df_s = df.describe().T[['mean']]
    # df = pd.concat([df_s,df_d], axis=1)
    # df.columns =['single network', 'double network']
    # pd.plotting.table(ax, np.round(df, 2), loc='center right', colWidths=[0.15, 0.15],
    #                 colLabels=[single_line,double_line], label='av')
    # New subplot
    # plt.legend(['double network', 'single network'])

    # ax = axs[1]
    # # ax.axvspan(0, 100, alpha=0.2, color='coral')
    # color = 'blue'
    # # ax.plot(starts, c=color)
    # ax.plot(mean_rews[0], c=color)
    # x = np.arange(0, len(mean_rews[0]))
    # ax.fill_between(x, mean_rews[0] - mean_rews[1], mean_rews[0] + mean_rews[1], color=color, alpha=0.1)
    # color = 'orange'
    # ax.plot(mean_rews_s[0], c=color, ls=':')
    # x = np.arange(0, len(mean_rews_s[0]))
    # ax.fill_between(x, mean_rews_s[0] - mean_rews_s[1], mean_rews_s[0] + mean_rews_s[1], color=color, alpha=0.1)
    # ax.set_ylabel('cum. return (arb. units)', color=color)
    # # ax.axhline(-0.05, ls=':', color='r')
    # ax.tick_params(axis='y', labelcolor=color)
    # # ax.set_title('reward per episode (arb. units)')  # + plot_suffix)
    # ax.set_xlabel('no. episodes')
    #
    # ax1 = plt.twinx(ax)
    # color = 'lime'
    # ax1.plot(final_rews[0][:-1], color=color)
    # ax1.plot(final_rews_s[0][:-1], color=color, ls=':')
    #
    # ax1.set_ylabel('final return (arb. units)', color=color)
    # ax1.axhline(-0.05, ls=':', color=color)
    # ax1.tick_params(axis='y', labelcolor=color)

    fig.align_labels()
    fig.tight_layout()
    # fig.suptitle('NonUniformImage class', fontsize='large')
    if 'save_name' in kwargs:
        save_name = kwargs.get('save_name')
        plt.savefig(save_name + '_episodes.pdf')
        plt.savefig(save_name + '_episodes.png')
        plt.savefig(save_name + '_episodes.pgf')
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
    ax.set_ylabel('rews per batch (arb. units)')

    ax.set_title(label)

    ax2 = ax.twinx()

    color = 'lime'
    ax2.set_ylabel('no. data points', color=color)  # we already handled the x-label with ax1
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
    ax.set_ylabel('rewards tests (arb. units)')
    # plt.tw
    ax.grid(True)
    ax.set_xlabel('no. epochs')
    if length_all:
        ax2 = ax.twinx()
        color = 'lime'
        if label == 'ME-TRPO':
            ax2.set_ylabel(r'log(std($p_\pi$)) (arb. units)', color=color)  # we already handled the x-label with ax1
        else:
            ax2.set_ylabel(r'success (1)', color=color)  # we already handled the x-label with ax1
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.plot(length_all, color=color)
    fig.align_labels()

    if 'save_name' in kwargs:
        plt.savefig(kwargs.get('save_name') + '.pdf')
        plt.savefig(kwargs.get('save_name') + '.png')
    plt.show()

def save_data(data, name):
    '''logging function to save results to pickle'''
    out_put_writer = open(name + '.pkl', 'wb')
    pickle.dump(data, out_put_writer, -1)
    out_put_writer.close()

# plot verification
data_array=[]
for label in ["ME-TRPO", "AE-DYNA"]:
    if label == "ME-TRPO":
        # ME-TRPO results
        project_directory = 'Data_Experiments/2020_10_06_ME_TRPO_stable@FERMI/run2/'
    elif label == "AE-DYNA":
        # AE-Dyna results
        project_directory = 'Data_Experiments/2020_11_05_AE_Dyna@FERMI/-nr_steps_25-cr_lr-n_ep_13-m_bs_100-sim_steps_3000-m_iter_35-ensnr_3-init_200/'
    filenames = []
    for file in os.listdir(project_directory):
        if 'final' in file:
            filenames.append(file)

    filenames.sort()

    filename = filenames[-1]
    print(filename)

    filehandler = open(project_directory + filename, 'rb')
    object = pickle.load(filehandler)

    if label == "ME-TRPO":
        object_me = object.copy()
    else:
        object_ae = object.copy()
    data_array.append(object)
    save_name = 'Figures/' + label

    name = 'Rewards_'+label
    print(name)
    data_out = object['rews']

    save_data(data_out, name=name)

save_name = 'Figures/Verification_DYNA_all'
plot_results_verification(data_array, label='DYNA-verification', save_name=save_name)
# plot observables

filenames = []
for file in os.listdir(project_directory):
    if 'training_observables' in file:
        filenames.append(file)

filenames.sort()

filename = filenames[-1]
print(filename)

# filehandler = open(project_directory + filename, 'rb')
# object = pickle.load(filehandler)
save_name = 'Figures/' + label + '_observables'
# plot_observables(object, label=label, save_name=save_name)



object = object_me
states_1 = object['obss']
print('object', object['obss'])
dones_1 = object['dones']
rewards_1 = object['rews']
actions_1 = object['acts']
selected_item = np.argmax([len(states_1[i]) for i
                           in range( len(states_1))])

# for max_lengths in range(len(states_1)):
data_s = pd.DataFrame(states_1[selected_item])
data_s.columns = ['tt1 tilt', 'tt1 incline', 'tt2 tilt', 'tt2 incline']
data_r = pd.DataFrame(rewards_1[selected_item])
data_r = data_r + 1
max = data_r[1:][dones_1[selected_item][1:]].values

font = {'family': 'serif',
        'color': 'darkslategray',
        'weight': 'normal',
        'size': 12,
        }
import matplotlib.gridspec as gspec

figure_name = 'Worst_episode_MBRL'
fig, (ax_1, ax_2) = plt.subplots(2)

ax = ax_1
ax1 = ax.twinx()
ax1.set_title(r'Worst verification episode ME-TRPO', fontdict=font, zorder=10)
color = 'lime'

ax1.axhline(y=0.95, c=color, ls='--')
ax1.plot(data_r.index.values, data_r.values, c=color, drawstyle="steps-post")

ax1.set_ylabel('FEL intensity (% of max)', color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax.plot(data_r.index.values, data_s, ls=':', drawstyle="steps-post")

ax.set_xlabel('no. step')
ax.set_ylabel('norm. state (arb. units)')
ax.legend(labels=data_s.columns, bbox_to_anchor=(0., 1.22, 1., .102), loc='lower left',
          ncol=2, mode="expand", borderaxespad=0.)
ax.set_ylim(0, 1)

object = object_ae
states_1 = object['obss']
print('object', object['obss'])
dones_1 = object['dones']
rewards_1 = object['rews']
actions_1 = object['acts']
# shift = 52
selected_item = np.argmax([len(states_1[i]) for i
                           in range( len(states_1))])
# print(selected_item)
# selected_item = 4
# for max_lengths in range(len(states_1)):
data_s = pd.DataFrame(states_1[selected_item])
data_a = pd.DataFrame(actions_1[selected_item][1:], columns=['1','2','3','4'])
data_s.columns = data_a.columns
data_correct = pd.concat([pd.DataFrame(data_s.iloc[0,:]).T, data_a], ignore_index=True)
print(data_s.head())
print(data_correct.cumsum().head())
# print(data_a.iloc[:,:])
data_s.columns = ['tt1 tilt', 'tt1 incline', 'tt2 tilt', 'tt2 incline']
data_r = pd.DataFrame(rewards_1[selected_item])
data_r = data_r + 1
max = data_r[1:][dones_1[selected_item][1:]].values
# data = pd.concat([data_s, data_r], axis=1)

ax_0 = ax_2
# ax_0.text(0, 1.02, r'Worst training episode among the last ten.', fontdict=font)
ax_0.set_title(r'Worst verification episode AE-DYNA', fontdict=font)
ax_0.plot(data_s.index.values, data_s.values, ls=':', drawstyle="steps-post")
# plt.legend(loc='upper right')
ax_0.set_xlabel('no. step')
ax_0.set_ylabel('norm. state (arb. units)')
ax_0.set_ylim(0, 1)
color = 'lime'
ax_1 = ax_0.twinx()
ax_1.axhline(y=0.95, c=color, ls='--')
ax_1.set_ylabel('FEL intensity (% of max)', color=color)
ax_1.tick_params(axis='y', labelcolor=color)
ax_1.plot(data_r.index.values, data_r.values, c=color, drawstyle="steps-post")

# plt.title(f'len: {len(states_1[selected_item][1:])} nr: {selected_item}')
plt.tight_layout(h_pad=0.2)
fig.align_labels()
plt.savefig('Figures/' + figure_name + '.pdf')
plt.show()