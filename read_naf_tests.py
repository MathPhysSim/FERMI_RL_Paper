import os
import pickle
import sys

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def read_experiment(root_dir, label = ''):
    data_all = pd.DataFrame()
    n_average = 25
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file == 'plot_data_0.pkl':
                file_name = root + os.sep + file
                print(file_name)
                openfile = open(file_name, 'rb')
                data = np.sum(pickle.load(openfile), axis=1)
                # data_frame = pd.DataFrame(data, columns=[root.split('/')[0]])
                column_names = [root.split('_')[1] + label]
                data_frame = pd.DataFrame(data, columns=column_names)
                data_frame = data_frame.rolling(n_average).mean().shift(int(n_average / 2))
                data_all = pd.concat([data_all, data_frame.T], sort=False)
                openfile.close()

    return data_all

try:
    root_dir = sys.argv[1]
except:
    # root_dir = "SINGLE/"
    # single = read_experiment(root_dir=root_dir)
    # root_dir = "DOUBLE/"
    # double = read_experiment(root_dir=root_dir)

    # data_all = pd.concat([single, double], sort=False)
    root_dir = "Experiments-150-200-50-long/"
    data_all = read_experiment(root_dir=root_dir)


# root_dir = "Experiments-150-200-50-long-0.05/"
# data_all_1 = read_experiment(root_dir=root_dir)
root_dir = "Experiments-150-200-50-noisy/"
data_all = read_experiment(root_dir=root_dir)

# data_all = data_all_1
# data_all = pd.concat([data_all, data_all_2])
figure_name = '../Figures/Comparison_noise'
current_test_name = ''

fig, axs = plt.subplots(1, 1, sharex=True)
ax = axs

print(data_all)

sns.set_palette(sns.color_palette("bright", 8))
sns.lineplot(ax=ax, data=data_all.T)
ax.axhline(y=-200, ls=':', c='lime')
plt.ylabel('cum. reward (arb. units.)')
ax.set_ylim(-1250,-100)
print(data_all)

ax.axhline(y=-200, ls=':', c='lime')
ax.legend(['Clipping','No-clipping-no-smoothing','No-clipping-smoothing'])
plt.ylabel('cum. reward (arb. units.)')
plt.xlabel('no. episode')
ax.set_ylim(-1250,-100)
plt.savefig(root_dir + figure_name+'.pdf')
plt.savefig(root_dir + figure_name+'.png')
plt.show()

# fig, axs = plt.subplots(2, 1, sharex=True)
# ax = axs[0]
#
# print(data_all)
#
# sns.set_palette(sns.color_palette("bright", 8))
# sns.lineplot(ax=ax, data=data_all.T)
# ax.axhline(y=-200, ls=':', c='lime')
# plt.ylabel('cum. reward (arb. units.)')
# ax.set_ylim(-1250,-100)
# print(data_all)
#
# root_dir = "Experiments-300-20-large-smoothing/"
# data_all = read_experiment(root_dir=root_dir)
# # sns.set_palette(sns.color_palette("bright", 8))
# ax = axs[1]
# sns.lineplot(ax=ax, data=data_all.T)
# ax.axhline(y=-200, ls=':', c='lime')
#
# plt.ylabel('cum. reward (arb. units.)')
# plt.xlabel('no. episode')
# ax.set_ylim(-1250,-100)
# # plt.savefig(root_dir + figure_name)
# plt.show()