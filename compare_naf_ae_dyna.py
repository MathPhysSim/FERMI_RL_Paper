import os
import sys
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Read NAF data
# def read_experiment(root_dir):
#     data_all = pd.DataFrame()
#     n_average = 15
#     for root, dirs, files in os.walk(root_dir):
#         for file in files:
#             if file == 'plot_data_0.pkl':
#                 file_name = root + os.sep + file
#                 print(file_name)
#                 openfile = open(file_name, 'rb')
#                 data = np.sum(pickle.load(openfile), axis=1)
#                 # data_frame = pd.DataFrame(data, columns=[root.split('/')[0]])
#                 data_frame = pd.DataFrame(data, columns=[root.split('_')[1]])
#                 # data_frame = data_frame.rolling(n_average).mean().shift(int(n_average / 2))
#                 data_all = pd.concat([data_all, data_frame.T], sort=False)
#                 openfile.close()
#
#     return data_all
#
# root_dir = "Experiments-150-200-50-noisy/"
# data_all_naf = read_experiment(root_dir=root_dir)
#
# data_all_naf.to_csv('NAF.csv')

# def read_experiment_observables(root_dir, key = 'tests_all'):
#     data_all = pd.DataFrame()
#     n_average = 5
#     for root, dirs, files in os.walk(root_dir):
#         file_names = []
#         for file in files:
#             if 'observables' in file and not 'Noise-off' in root:
#                 file_names.append(file)
#         if file_names:
#             file_names.sort()
#             file_name = file_names[-1]
#             file_name = root + os.sep + file_name
#             openfile = open(file_name, 'rb')
#             if not key in ['step_counts_all']:
#                 data = pd.DataFrame(pickle.load(openfile)[key]).sum()
#                 print(data)
#                 data_frame = pd.DataFrame(data, columns=[root.split('_')[4]])
#                 print(data_frame)
#             else:
#                 data = pickle.load(openfile)[key]
#                 data_frame = pd.DataFrame(data, columns=[root.split('_')[4]])
#
#             # data_frame = data_frame.rolling(n_average).mean().shift(int(n_average / 2))
#             data_all = pd.concat([data_all, data_frame.T], sort=False)
#             openfile.close()
#
#     return data_all
# root_dir = 'Data_logging/Experiment_noise_pendulum/'
# data_all = read_experiment_observables(root_dir=root_dir, key='tests_all')
# data_all.to_csv('AEDYNA_rews.csv')
# data_all = read_experiment_observables(root_dir=root_dir, key='step_counts_all')
# data_all.to_csv('AEDYNA_steps.csv')


data_naf = pd.read_csv('NAF.csv', index_col=0)
data_naf.columns = np.array(data_naf.columns, dtype='int') + 1
data_naf.columns = (200 * (data_naf.columns))
data_naf = data_naf.loc['Clipping-noisy', :10000]
data_ae_rew = pd.read_csv('AEDYNA_rews.csv', index_col=0)
data_ae_rew = data_ae_rew.loc['Noise-on-aleatoric', :]
data_ae_steps = pd.read_csv('AEDYNA_steps.csv', index_col=0)
data_ae_steps = data_ae_steps.groupby(by=data_ae_steps.index).mean().iloc[0, :]
data_ae_rew.columns = data_ae_steps
data_ae_rew.columns = np.array(data_ae_rew.columns, dtype='int')
data_ae_rew = data_ae_rew.T.groupby(by=data_ae_rew.T.index).max()
# n_average = 50
# data_ae_rew = data_ae_rew.reindex(np.linspace(201, 2010, num=351)).interpolate('values')
# data_ae_rew = data_ae_rew.rolling(n_average).mean().shift(int(n_average / 2)).T
data_ae_rew =data_ae_rew.T
fig, axs = plt.subplots(1, 1, sharex=True)
sns.set_palette("rocket")
sns.lineplot(ax=axs, data=data_ae_rew.T)

sns.set_palette(sns.color_palette("bright", 8))
sns.lineplot(ax=axs, data=data_naf.T)

axs.axhline(y=-200, c='lime', ls=':')
plt.ylabel('cum. reward (arb. units.)')
plt.xlabel('no. data points')
plt.savefig('Figures/Comparison_NAF_ae_dyna.pdf')
plt.show()
