import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import time

# import PyQt5
matplotlib.use("Qt5Agg")

from local_fel_simulated_env import FelLocalEnv
from simulated_tango import SimTangoConnection

tango = SimTangoConnection()


# real_env = FelLocalEnv(tango=tango)

class EnvironmentWrapper():
    def __init__(self, **kwargs):
        self.real_env = FelLocalEnv(tango=tango)
        self.real_env.reset()
        self.action_space_dimensions = self.real_env.action_space.shape[0]
        self.action_bounds_high, self.action_bounds_low = self.real_env.action_space.high, self.real_env.action_space.low
        print(self.action_bounds_high)
        # print(self.action_bounds_high, self.action_bounds_low)
        self.action_names = ['a1', 'a2', 'a3', 'a4']
        self.action_history = pd.DataFrame(columns=self.action_names)
        self.state_names = ['s1', 's2', 's3', 's4']
        self.state_history = pd.DataFrame(columns=self.state_names)

        # self.state_names = list(self.awake_env.state_names)
        # state_index_h, state_index_v = [], []
        #
        # for element in self.state_names:
        #     name, plane = element.split('_')
        #     if plane == 'horizontal':
        #        state_index_h.append(name)
        #     else:
        #         state_index_v.append(name)
        # self.state_h_history = pd.DataFrame(columns=state_index_h)
        # self.state_v_history = pd.DataFrame(columns=state_index_v)
        # print(self.state_h_history, self.state_v_history)

        self.fig = plt.figure()
        self.ax1 = self.fig.add_subplot(211)
        self.ax2 = self.fig.add_subplot(212, sharex=self.ax1)
        # plt.show(block=False)

        self.fig1, self.axs = plt.subplots(2, 1, sharex=True)
        # plt.show(block=False)

        self.counter = -1

        self.session_name = 'Niky_has_fun.h5'

        self.algorithm_name = None

        self.current_action = np.zeros(self.action_space_dimensions)

    def reset(self, state=None):
        # self.current_state, self.current_reward = self.real_env.reset() NIKY
        self.current_state = self.real_env.reset()
        self.current_reward = np.nan
        self.set_history(action=[np.nan for i in range(4)], r=self.current_reward, s=self.current_state)

    def set_history(self, action, r, s):
        self.counter += 1
        size = self.action_history.shape[0]
        self.action_history.loc[size, self.action_names] = action
        self.action_history.loc[size, 'objective'] = r
        self.state_history.loc[size, :] = s
        storage_key = 'scan_' + str(self.counter)
        history_all = pd.concat([self.action_history, self.state_history], axis=1)
        history_all.to_hdf(self.session_name, key=storage_key)

    def objective(self, action):
        inner_action = action-self.current_action
        state = self.real_env.state
        s, r, d, _ = self.real_env.step(action=inner_action)
        self.set_history(inner_action, r=r, s=s)
        print(state, action, s, r)
        self.current_action=inner_action.copy()
        return -r


algorithm_list = ['Powell', 'COBYLA']
algotihm_name = algorithm_list[1]

if __name__ == '__main__':
    print('starting the algorithm:', algotihm_name)

    environment_instance = EnvironmentWrapper()
    start_vector = np.zeros(environment_instance.action_space_dimensions)
    environment_instance.reset()

    if algotihm_name == 'COBYLA':
        def constr(action):
            if any(action > environment_instance.action_bounds_high):
                return -1
            elif any(action < environment_instance.action_bounds_low):
                return -1
            else:
                return 1
        rhobeg = 0.25 * environment_instance.action_bounds_high[0]
        res = opt.fmin_cobyla(environment_instance.objective, start_vector, [constr],
                              rhobeg=rhobeg, rhoend=.01, maxfun=10)

    elif algotihm_name == 'Powell':
        res = opt.fmin_powell(environment_instance.objective, start_vector, ftol=0.1, xtol=0.001,
                              direc=0.5 * environment_instance.action_bounds_high)

    print(res)
    data = pd.read_hdf('Niky_has_fun.h5', key='scan_' + str(10))
    print(data.T)