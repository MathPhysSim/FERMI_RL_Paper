import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


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
    iterations = pd.DataFrame(iterations_all)
    final_rews = pd.DataFrame(final_rews_all)
    mean_rews = pd.DataFrame(mean_rews_all)
    return pd.concat([iterations, mean_rews, final_rews], keys=['nr', 'cum', 'final'])


project_directory = ''
name = 'Rewards_NAF_double.pkl'

filehandler = open(project_directory + name, 'rb')
object = pickle.load(filehandler)

naf_double = read_rewards(object, data_range_in=[99, 149]).mean(level=0)

name = 'Rewards_NAF_single.pkl'

filehandler = open(project_directory + name, 'rb')
object = pickle.load(filehandler)
naf_single = read_rewards(object, data_range_in=[99, 149]).mean(level=0)

name = 'Rewards_AE-DYNA.pkl'

filehandler = open(project_directory + name, 'rb')
object = pickle.load(filehandler)
ae_dyna = read_rewards([object]).mean(level=0)
name = 'Rewards_ME-TRPO.pkl'

filehandler = open(project_directory + name, 'rb')
object = pickle.load(filehandler)
me_trpo = read_rewards([object]).mean(level=0)

# print(naf_double)
# print(naf_single)
# print(ae_dyna)
# print(me_trpo)

data_all = pd.concat([naf_single, naf_double, ae_dyna, me_trpo],
                     keys=['naf_single', 'naf_double', 'ae_dyna', 'me_trpo'])

idx = pd.IndexSlice
print(data_all.loc[idx[:, 'nr'], :].T.describe().T)
# print(data_all.T.describe().T[['mean', 'std']].apply(lambda x: np.round(x,2)).to_latex())
# print(data_all.loc[idx[:, 'final'], :].mean(axis=1))
# print(data_all)
# plt.show()

# print(data_all.T.describe().T[['mean', 'std']].apply(lambda x: np.round(x,2)).swaplevel(0, 1, axis=0))
# print(data_all.T.describe().T[['mean', 'std']].apply(lambda x: np.round(x,2)).sort_index(level=1).to_latex())
print(data_all.T.describe().T)