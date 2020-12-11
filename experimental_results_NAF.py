'''This script generates the plots for the NAF2 tests at FERMI FEL'''

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt


def load_pickle_logging(file_name):
    directory = file_name + '/'
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
    directory = file_name + '/'
    file = 'plot_data_0.pkl'

    with open(directory + file, 'rb') as f:
        rews = pickle.load(f)
        inits = pickle.load(f)
        losses = pickle.load(f)
        v_s = pickle.load(f)
    return rews, inits, losses, v_s


def load_buffer_final(file_name):
    directory = file_name + '/'
    file = 'buffer_data_verification.pkl'

    with open(directory + file, 'rb') as f:
        buffer = pickle.load(f)
    return buffer


def load_buffer(file_name):
    directory = file_name + '/'
    files = []
    directory = directory + 'data/'
    for f in os.listdir(directory):
        if 'buffer_data' in f and 'pkl' in f:
            files.append(f)
    files.sort()
    print(files[-1])

    with open(directory + files[-1], 'rb') as f:
        buffer = pickle.load(f)
    return buffer


# file_name = 'Data_Experiments/2020_07_20_NAF@FERMI/FEL_training_100_double_q_Tango_11'
# states_0, actions_0, rewards_0, dones_0 = load_pickle_logging(file_name)
# file_name = 'Data_Experiments/2020_07_20_NAF@FERMI/FEL_training_100_double_q_Tango_11_bis'
# states_1, actions_1, rewards_1, dones_1 = load_pickle_logging(file_name)
# rewards = [rewards_0, rewards_1]
#
# file_name_s = 'Data_Experiments/2020_07_20_NAF@FERMI/FEL_training_100_single_q_Tango_11'
# states_s0, actions_s0, rewards_s0, dones_s0 = load_pickle_logging(file_name_s)
# file_name_s = 'Data_Experiments/2020_07_20_NAF@FERMI/FEL_training_100_single_q_Tango_11_bis'
# states_s, actions_s, rewards_s, dones_s = load_pickle_logging(file_name_s)
# rewards_s = [rewards_s, rewards_s0]

project_directory = ''


def save_data(data, name):
    '''logging function to save results to pickle'''
    clock_time = 'Paper'
    out_put_writer = open(project_directory + name + '.pkl', 'wb')
    pickle.dump(data, out_put_writer, -1)
    out_put_writer.close()


def read_rewards(rewards_in, data_range_in=None):
    iterations_all = []
    final_rews_all = []
    mean_rews_all = []
    for k in range(len(rewards_in)):
        rewards = rewards_in[k]

        iterations = []
        final_rews = []
        mean_rews = []
        if data_range_in is None:
            data_range = range(len(rewards))
        else:
            data_range = range(data_range_in[0], min(len(rewards), data_range_in[1]))
        for i in data_range:
            # print(i)
            if len(rewards[i]) > 0:

                final_rews.append(rewards[i][-1])
                iterations.append(len(rewards[i]))
                try:
                    mean_rews.append(np.sum(rewards[i][1:]))
                except:
                    mean_rews.append([])
        iterations_all.append(iterations)
        final_rews_all.append(final_rews)
        mean_rews_all.append(mean_rews)

    iterations_mean = np.mean(np.array(iterations_all), axis=0)
    iterations_std = np.std(np.array(iterations_all), axis=0)
    final_rews_mean = np.mean(np.array(final_rews_all), axis=0)
    final_rews_std = np.mean(np.array(final_rews_all), axis=0)
    mean_rews_mean = np.mean(np.array(mean_rews_all), axis=0)
    mean_rews_std = np.mean(np.array(mean_rews_all), axis=0)

    return [iterations_mean, iterations_std], [final_rews_mean, final_rews_std], \
           [mean_rews_mean, mean_rews_std], np.array(iterations_all)


def plot_results(rewards, rewards_single, label=None, **kwargs):
    if 'data_range_in' in kwargs:
        data_range_in = kwargs.get('data_range_in')
    else:
        data_range_in = None

    iterations, final_rews, mean_rews, iter_all = read_rewards(rewards, data_range_in=data_range_in)
    iterations_s, final_rews_s, mean_rews_s, iter_all_s = read_rewards(rewards_single, data_range_in=data_range_in)
    plot_suffix = ""  # f', number of iterations: {env.TOTAL_COUNTER}, Linac4 time: {env.TOTAL_COUNTER / 600:.1f} h'
    fig, axs = plt.subplots(1, 1, sharex=True)
    fig.set_size_inches(fig.get_size_inches()[0], fig.get_size_inches()[1] * 0.6)
    ax = axs
    # ax.axvspan(0, 100, alpha=0.2, color='coral')
    color = 'blue'
    x = np.arange(0, len(iterations[0]))
    ax.plot(iterations[0], c=color)
    ax.fill_between(x, iterations[0] - iterations[1], iterations[0] + iterations[1], color=color, alpha=0.1)
    # color = 'orange'
    x = np.arange(0, len(iterations_s[0]))
    ax.plot(iterations_s[0], c=color, ls=':')
    ax.fill_between(x, iterations_s[0] - iterations_s[1], iterations_s[0] + iterations_s[1], color=color, alpha=0.1)

    ax.set_ylabel('no. iterations', color=color)
    ax.tick_params(axis='y', labelcolor=color)

    ax1 = plt.twinx(ax)
    if 'verification' in kwargs:
        verification = kwargs.get('verification')
        if verification:
            ax1.set_ylim(0, 275)
            ax.set_ylim(1, 10)
    color = 'lime'
    iterations_mean = np.mean(np.cumsum(np.array(iter_all), axis=1), axis=0)
    iterations_std = np.std(np.cumsum(np.array(iter_all), axis=1), axis=0)
    ax1.plot(iterations_mean, c=color)
    x = np.arange(0, len(iterations_mean))
    ax1.fill_between(x, iterations_mean - iterations_std, iterations_mean
                     + iterations_std, color=color, alpha=0.1)

    iterations_mean = np.mean(np.cumsum(np.array(iter_all_s), axis=1), axis=0)
    iterations_std = np.std(np.cumsum(np.array(iter_all_s), axis=1), axis=0)
    ax1.plot(iterations_mean, c=color, ls=':')
    x = np.arange(0, len(iterations_mean))
    ax1.fill_between(x, iterations_mean - iterations_std, iterations_mean
                     + iterations_std, color=color, alpha=0.1)

    ax1.set_ylabel('no. cumulative steps', color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax.set_title(label)
    # fig.suptitle(label, fontsize=12)
    import matplotlib.lines as mlines

    single_line = mlines.Line2D([], [], color='k', ls=':', label='single network')
    double_line = mlines.Line2D([], [], color='k', ls='-', label='double network')
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


def plot_results_verification(rewards, rewards_single, label=None, **kwargs):
    if 'data_range_in' in kwargs:
        data_range_in = kwargs.get('data_range_in')
    else:
        data_range_in = None

    iterations, final_rews, mean_rews, iter_all = read_rewards(rewards, data_range_in=data_range_in)
    iterations_s, final_rews_s, mean_rews_s, iter_all_s = read_rewards(rewards_single, data_range_in=data_range_in)
    plot_suffix = ""  # f', number of iterations: {env.TOTAL_COUNTER}, Linac4 time: {env.TOTAL_COUNTER / 600:.1f} h'
    fig, axs = plt.subplots(1, 1, sharex=True)
    fig.set_size_inches(fig.get_size_inches()[0], fig.get_size_inches()[1] * 0.6)
    ax = axs
    # ax.axvspan(0, 100, alpha=0.2, color='coral')
    color = 'blue'
    x = np.arange(0, len(iterations[0]))
    ax.plot(iterations[0], c=color)
    # ax.fill_between(x, iterations[0] - iterations[1], iterations[0] + iterations[1], color=color, alpha=0.1)
    # color = 'orange'
    x = np.arange(0, len(iterations_s[0]))
    ax.plot(iterations_s[0], c=color, ls=':')
    # ax.fill_between(x, iterations_s[0] - iterations_s[1], iterations_s[0] + iterations_s[1], color=color, alpha=0.1)

    ax.set_ylabel('no. iterations', color=color)
    ax.tick_params(axis='y', labelcolor=color)

    ax1 = plt.twinx(ax)
    if 'verification' in kwargs:
        verification = kwargs.get('verification')
        if verification:
            ax1.set_ylim(0, 275)
            ax.set_ylim(0, 10)
    color = 'lime'
    iterations_mean = np.mean(np.cumsum(np.array(iter_all), axis=1), axis=0)
    iterations_std = np.std(np.cumsum(np.array(iter_all), axis=1), axis=0)
    ax1.plot(iterations_mean, c=color)
    x = np.arange(0, len(iterations_mean))
    # ax1.fill_between(x, iterations_mean-iterations_std, iterations_mean
    #                  + iterations_std, color=color, alpha=0.1)

    iterations_mean = np.mean(np.cumsum(np.array(iter_all_s), axis=1), axis=0)
    iterations_std = np.std(np.cumsum(np.array(iter_all_s), axis=1), axis=0)
    ax1.plot(iterations_mean, c=color, ls=':')
    x = np.arange(0, len(iterations_mean))
    # ax1.fill_between(x, iterations_mean-iterations_std, iterations_mean
    #                  + iterations_std, color=color, alpha=0.1)

    ax1.set_ylabel('no. cumulative steps', color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax.set_title(label)
    # fig.suptitle(label, fontsize=12)
    import matplotlib.lines as mlines

    single_line = mlines.Line2D([], [], color='k', ls=':', label='single network')
    double_line = mlines.Line2D([], [], color='k', ls='-', label='double network')
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


label = 'FERMI_all_experiments_NAF'


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


def plot_convergence(losses, v_s, losses_s, v_s_s, label, **kwargs):
    fig, ax = plt.subplots()
    fig.set_size_inches(fig.get_size_inches()[0], fig.get_size_inches()[1] * 0.6)
    ax.set_title(label)
    ax.set_xlabel('no. steps')

    color = 'tab:blue'
    ax.semilogy(losses, color=color)
    ax.semilogy(losses_s, color=color, ls=':')
    ax.tick_params(axis='y', labelcolor=color)
    ax.set_ylabel('Bellman error (arb. units)', color=color)
    # ax.set_ylim(0, 1)

    ax1 = plt.twinx(ax)
    # ax1.set_ylim(-2, 1)
    color = 'lime'

    ax1.set_ylabel('V (arb. units)', color=color)  # we already handled the x-label with ax1
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.plot(v_s, color=color)
    ax1.plot(v_s_s, color=color, ls=':')
    plt.tight_layout()
    if 'save_name' in kwargs:
        save_name = kwargs.get('save_name')
        plt.savefig(save_name + '_convergence' + '.pdf')
        plt.savefig(save_name + '_convergence' + '.png')
    plt.show()


file_name = 'Data_Experiments/2020_07_20_NAF@FERMI/FEL_training_100_double_q_Tango_11'
rews0, inits0, losses0, v_s0 = load_pickle_final(file_name)
file_name = 'Data_Experiments/2020_07_20_NAF@FERMI/FEL_training_100_double_q_Tango_11_bis'
rews1, inits1, losses1, v_s1 = load_pickle_final(file_name)
losses, v_s = read_losses_v_s([losses0, losses1], [v_s0, v_s1], 691)
# rewards = [rews0, rews1]
rewards = [rews1[:99], rews0[:99]]
# rewards = [rews1[99:]+rews0[99:]]
#
file_name = 'Data_Experiments/2020_07_20_NAF@FERMI/FEL_training_100_single_q_Tango_11'
rews_0, inits0, losses0, v_s0 = load_pickle_final(file_name)
file_name = 'Data_Experiments/2020_07_20_NAF@FERMI/FEL_training_100_single_q_Tango_11_bis'
rews_1, inits1, losses1, v_s1 = load_pickle_final(file_name)
losses_s, v_s_s = read_losses_v_s([losses0, losses1], [v_s0, v_s1], 691)

# rewards_s = [rews0, rews1]
rewards_s = [rews_1[:99], rews_0[:99]]

label = 'FERMI_all_experiments_NAF_training'
save_name = 'Figures/' + label
# plot_results(rewards, rewards_s, 'NAF trainings', save_name=save_name, data_range_in=[0, 100])
plot_results(rewards, rewards_s, 'NAF trainings', save_name=save_name)
#
label = 'FERMI_all_experiments_NAF_verification'
save_name = 'Figures/' + label
# plot_results(rewards, rewards_s, 'NAF verifications', save_name=save_name, data_range_in=[99, 150], verification=True)
rewards_s = [rews_1[99:] + rews_0[99:]]
rewards = [rews1[99:] + rews0[99:]]
name = 'Rewards_NAF_double'
save_data(rewards, name=name)
name = 'Rewards_NAF_single'
save_data(rewards_s, name=name)

plot_results_verification(rewards, rewards_s, 'NAF verifications', save_name=save_name, verification=True)

label = 'FERMI_all_experiments_NAF'
save_name = 'Figures/' + label
plot_convergence(losses, v_s, losses_s, v_s_s, label='NAF trainings - metrics', save_name=save_name)

figure_name = 'Evolution_double'
file_name = 'Data_Experiments/2020_07_20_NAF@FERMI/FEL_training_100_double_q_Tango_11'
# states_0, actions_0, rewards_0, dones_0 = load_pickle_logging(file_name)
# file_name = 'Data_Experiments/2020_07_20_NAF@FERMI/FEL_training_100_double_q_Tango_11_bis'
# states_1, actions_1, rewards_1, dones_1 = load_pickle_logging(file_name)
#
# figure_name = 'Evolution_single'
# file_name = 'Data_Experiments/2020_07_20_NAF@FERMI/FEL_training_100_single_q_Tango_11'
# states_0, actions_0, rewards_0, dones_0 = load_pickle_logging(file_name)
# file_name = 'Data_Experiments/2020_07_20_NAF@FERMI/FEL_training_100_single_q_Tango_11_bis'
states_1, actions_1, rewards_1, dones_1 = load_pickle_logging(file_name)
print(len(states_1[-1]), len(actions_1[-1]), len(rewards_1[-1]), len(dones_1[-1]))
import pandas as pd

print(len(states_1))
selected_item = np.argmax([len(states_1[i]) for i in range(len(states_1))])

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

fig, (ax_1, ax_2) = plt.subplots(2)

ax = ax_1
ax1 = ax.twinx()
ax1.set_title(r'Worst training episode among the first ten.', fontdict=font, zorder=10)
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

shift = 2
selected_item = np.argmax([len(states_1[i]) for i
                           in range(len(states_1) - shift, len(states_1))]) + len(states_1) - shift
print('selected_item', selected_item)
# selected_item = 55
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
ax_0.set_title(r'Worst training episode among the last ten.', fontdict=font)
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
# plt.pause(0.05)

buffer = load_buffer_final(file_name=file_name)
# buffer = load_buffer(file_name=file_name)
# print(len(list(map(len, load_pickle_final(file_name=file_name)[0]))))

# print((buffer_0['obs1'][-20:]))
# print((buffer_1['obs1'][-20:]))
# print(len(list(map(len, load_pickle_logging(file_name=file_name)[2]))))

#
traj = []
traj_nr = 0
traj.append([])

for pos_ind in range(len(buffer['done'])):
    # print(traj_nr, buffer['done'][pos_ind])
    if not buffer['done'][pos_ind]:

        traj[traj_nr].append(buffer['rews'][pos_ind])
    else:
        traj[traj_nr].append(buffer['rews'][pos_ind])
        traj_nr += 1
        traj.append([])

trajs = pd.DataFrame(traj)
trajs.T.plot()
# print(trajs)
# plt.show()
