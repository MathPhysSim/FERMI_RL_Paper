import os
import numpy as np
import matplotlib.pyplot as plt
# data = dict(data=data,
#             model=model_nr,
#             rews=rews,
#             X=X,
#             Y=Y)
import pickle

project_directory = 'Data_logging/Test_plots/'+'-nr_steps_25-cr_lr-n_ep_7-m_bs_100-sim_steps_2500-m_iter_30-ensnr_3-init_100/'
files = []
inner_list = []
for f in os.listdir(project_directory):
    if 'plot_model' in f:
        if f[-1] == '0':
            if len(inner_list)>0:
                files.append(inner_list)
            inner_list = []
            inner_list.append(f)
        else:
            inner_list.append(f)

# print(files)
number = 3
with open(project_directory+files[number][0], 'rb') as f:
    file_data = pickle.load(f)
# print(file_data)

maximum = 0
data = file_data['data']
if data is not None:
    action = [data[1]]
    state = data[0]
    maximum = (data[2] - 1) / 2
else:
    action = np.zeros(4)
    state = np.zeros(4)

delta = 0.05
x = np.arange(-1, 1, delta)
y = np.arange(-1, 1, delta)
X, Y = np.meshgrid(x, y)
Nr = 1
Nc = 1
fig, axs = plt.subplots(Nr, Nc)
fig.subplots_adjust(hspace=0.3)
images = []
for pos in range(len(files[0])):
    with open(project_directory+files[number][pos], 'rb') as f:
        file_data = pickle.load(f)
    rewards= file_data['rews']

    # print(self.number_models)
    for i1 in range(len(x)):
        for j1 in range(len(y)):
            state[0] = x[i1]
            state[1] = y[j1]

    axs.contour(X, Y, (rewards - 1) / 2, alpha=1)

# plt.colorbar()
fig.show()