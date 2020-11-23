import os
import pickle
import sys

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def read_experiment(root_dir):
    data_all = pd.DataFrame()
    n_average = 15
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file == 'plot_data_0.pkl':
                file_name = root + os.sep + file
                print(file_name)
                openfile = open(file_name, 'rb')
                data = np.sum(pickle.load(openfile), axis=1)
                # data_frame = pd.DataFrame(data, columns=[root.split('/')[0]])
                data_frame = pd.DataFrame(data, columns=[root.split('_')[1]])
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
    root_dir = "Experiments-150-200-50/"
    data_all = read_experiment(root_dir=root_dir)

figure_name = 'Smoothing_comparison.pdf'
current_test_name = ''

fig, axs = plt.subplots(1, 1)

# data_all = data_all.loc[:, :length]
print(data_all)

sns.lineplot(ax=axs, data=data_all.T)
plt.ylabel(' cum reward (arb. units.)')
plt.xlabel('no. episode')
plt.savefig(root_dir + figure_name)
plt.show()
