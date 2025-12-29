import os
import pickle
import sys

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def read_experiment_observables(root_dir, key='tests_all'):
    data_all = pd.DataFrame()
    n_average = 5
    for root, dirs, files in os.walk(root_dir):
        file_names = []
        for file in files:
            if 'observables' in file and not 'Noise-off' in root:
                file_names.append(file)
        if file_names:
            file_names.sort()
            file_name = file_names[-1]
            file_name = root + os.sep + file_name
            openfile = open(file_name, 'rb')
            if not key in ['step_counts_all']:
                object = pickle.load(openfile)
                data = pd.DataFrame(object[key]).sum()
                # index = object['step_counts_all']
                # data.index = index

                data_frame = pd.DataFrame(data, columns=[root.split('_')[1]])
                data_frame = data_frame.rolling(n_average).mean().shift(int(n_average / 2))
            else:
                data = pickle.load(openfile)[key]
                print(root.split('_')[1])
                data_frame = pd.DataFrame(data, columns=[root.split('_')[1]])

            data_all = pd.concat([data_all, data_frame.T], sort=False)
            openfile.close()

    return data_all


def read_experiment_observables(root_dir, key='tests_all'):
    data_all = pd.DataFrame()
    n_average = 5
    for root, dirs, files in os.walk(root_dir):
        file_names = []
        for file in files:
            if 'observables' in file and not 'Noise-off' in root:
                file_names.append(file)
        if file_names:
            file_names.sort()
            file_name = file_names[-1]
            file_name = root + os.sep + file_name
            openfile = open(file_name, 'rb')
            if not key in ['step_counts_all'] and not root.split('_')[1] == 'Five':

                object = pickle.load(openfile)
                data = pd.DataFrame(object[key][0], columns=[root.split('_')[1]])
                index = object['step_counts_all']
                data.index = index
                data = data.groupby(data.index).max()
                # print(data)
            # else:
            #     data = pickle.load(openfile)[key]
            #     print(root.split('_')[1])
            #     data_frame = pd.DataFrame(data, columns=[root.split('_')[1]])

                data_all = pd.concat([data_all, data], axis=1, sort=False)

            openfile.close()

    return data_all


def read_rewards(rewards):
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
    return np.array(iterations), np.array(final_rews), np.array(mean_rews), np.array(stds)


def read_experiment_final(root_dir):
    data_all = pd.DataFrame()
    n_average = 5
    for root, dirs, files in os.walk(root_dir):
        file_names = []
        for file in files:
            if 'final' in file and not 'Noise-off' in root:
                file_names.append(file)
        if file_names:
            file_names.sort()
            file_name = file_names[-1]
            file_name = root + os.sep + file_name
            print(file_name)
            openfile = open(file_name, 'rb')
            # data = np.sum(pickle.load(openfile)['rews'], axis=1)
            data = pd.DataFrame(pickle.load(openfile)['rews']).sum()
            # data_frame = pd.DataFrame(data, columns=[root.split('/')[0]])
            data_frame = pd.DataFrame(data, columns=[root.split('_')[4]])
            data_frame = data_frame.rolling(n_average).mean().shift(int(n_average / 2))
            data_all = pd.concat([data_all, data_frame.T], sort=False)
            openfile.close()

    return data_all


# root_dir = 'Data_logging/Experiment_noise_pendulum/'
#
# figure_name = '../../Figures/Comparison_noise_ae_dyna.pdf'
#
# fig, axs = plt.subplots(3, 1, sharex=True)
# ax=axs[0]
# data_all_observables = read_experiment_observables(root_dir=root_dir, key='tests_all')
# data_all_observables = data_all_observables.loc[:, :60]
# sns.set_palette(sns.color_palette("bright", 3))
# sns.lineplot(ax=ax, data=data_all_observables.T, legend=False)
# # ax.set_title('Verification on real environment')
# ax.axhline(y=-200, ls=':', color = 'lime')
# ax.set_ylabel('mean cum. reward\n (arb. units.)')
#
# ax = axs[1]
# data_all_observables = read_experiment_observables(root_dir=root_dir, key='step_counts_all')
# data_all_observables = data_all_observables.loc[:, :60]
# sns.set_palette(sns.color_palette("bright", 8))
# sns.lineplot(ax=ax, data=data_all_observables.T)
# ax.set_ylabel('no. data points')
#
# ax=axs[2]
# data_all_observables = read_experiment_observables(root_dir=root_dir, key='batch_rews_all')
# data_all_observables = data_all_observables.loc[:, :60]
# sns.set_palette(sns.color_palette("bright", 8))
# sns.lineplot(ax=ax, data=data_all_observables.T, legend=False)
# ax.set_ylabel('batch reward\n (arb. units.)')
# ax.set_xlabel('no. epochs')
# # ax.set_title('Batch rewards during training')
#
# plt.tight_layout()
#
# plt.savefig(root_dir + figure_name)
# plt.show()

root_dir = 'Data/Simulation/ModelSizeLong/'
n_average = 30
figure_name = 'Figures/Compare_models_sizes.pdf'
data_all = read_experiment_observables(root_dir=root_dir)
data_all = data_all.reindex(np.linspace(201, 2010, num=151)).interpolate('values')

data_all = data_all.rolling(n_average).mean().shift(int(n_average / 2)).T
print(data_all.T)
fig, axs = plt.subplots(1, 1)

sns.set_palette(sns.color_palette("bright", 8))
sns.lineplot(ax=axs, data=data_all.T)
plt.ylabel('mean max cum. reward (arb. units.)')
plt.xlabel('no. data points')
plt.axhline(y=-200, ls=':', color='lime')
# plt.xlim(200,1600)
plt.savefig(figure_name)
plt.show()
data_all_rew = read_experiment_observables(root_dir=root_dir)
# print(data_all_rew)
# data_all_nr = read_experiment_observables(root_dir=root_dir, key='step_counts_all')
# print(data_all_nr)

# data_all = pd.concat([data_all_rew, data_all_nr], keys=['rew', 'nr'])
# print(data_all.T)
