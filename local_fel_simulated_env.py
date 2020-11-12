import pickle

import numpy as np
import gym
# import pybobyqa
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd

from simulated_tango import SimTangoConnection


class FelLocalEnv(gym.Env):

    def __init__(self, tango, **kwargs):
        self.max_steps = 10
        print('init env ' * 20)
        self.init_rewards = []
        self.done = False
        self.current_length = 0
        self.__name__ = 'FelLocalEnv'

        self.curr_episode = -1
        self.TOTAL_COUNTER = -1

        self.rewards = []
        self.states = []
        self.actions = []
        self.dones = []

        self.initial_conditions = []

        # tango = SimTangoConnection() simulates the behaviour of the system we want to control
        self.tango = tango

        # some information from tango
        self.state_size = self.tango.state_size
        self.action_size = self.tango.action_size

        self.target_state = self.tango.target_state
        self.target_intensity = self.tango.target_intensity

        # current state
        self.init_state = self.tango.state

        # scaling factor definition
        if 'half_range' in kwargs:
            self.half_range = kwargs.get('half_range')
        else:
            self.half_range = 3000

        self.state_range = self.get_range()
        self.state_scale = 2 * self.half_range

        # state, intensity and reward first definition
        self.state = self.scale(self.init_state)
        self.intensity = self.get_intensity()
        self.reward = self.get_reward()

        # max action allowed
        if 'max_action' in kwargs:
            max_action = kwargs.get('max_action')
        else:
            max_action = 500
            # max_action = 6000
        self.max_action = max_action / self.state_scale

        print('max_action', max_action)

        # state space definition
        self.observation_space = gym.spaces.Box(low=0.0,
                                                high=1.0,
                                                shape=(self.state_size,),
                                                dtype=np.float64)

        # action space definition
        self.action_space = gym.spaces.Box(low=-self.max_action,
                                           high=self.max_action,
                                           shape=(self.state_size,),
                                           dtype=np.float64)
        self.test = False

        print('real env scale:', self.action_space.low, self.action_space.high, self.observation_space.low,
              self.observation_space.high)

    def get_range(self):
        # defines the available state space
        state_range = np.c_[self.init_state - self.half_range, self.init_state + self.half_range]
        return state_range

    def scale(self, state):
        # scales the state from state_range values to [0, 1]
        state_scaled = (state - self.state_range[:, 0]) / self.state_scale
        return state_scaled

    def descale(self, state):
        # descales the state from [0, 1] to state_range values
        state_descaled = state * self.state_scale + self.state_range[:, 0]
        return state_descaled

    def set_state(self, state):
        # writes descaled state
        state_descaled = self.descale(state)
        self.tango.set_state(state_descaled)

    def get_state(self):
        # read scaled state
        state = self.tango.get_state()
        state_scaled = self.scale(state)
        return state_scaled

    def set_state_ext(self, state):
        state_descaled = self.descale(state)
        self.tango.set_state(state_descaled)
        state = self.tango.get_state()
        self.state = self.scale(state)

    def norm_intensity(self, intensity):
        # normalize the intensity with respect to target_intensity
        intensity_norm = intensity / self.target_intensity
        return intensity_norm

    def get_intensity(self):
        # read normalized intensity
        intensity = self.tango.get_intensity()
        intensity_norm = self.norm_intensity(intensity)
        return intensity_norm

    def step(self, action):
        action = np.squeeze(action)
        # print('a', action)
        # step method
        self.current_length += 1
        # rescale action
        # action /= 6
        # action = np.clip(action, -1, 1)
        state, reward = self.take_action(action.copy())
        # state = state + 1e-4*np.random.randn(self.observation_space.shape[-1])
        # reward += 1e-4 * np.random.randn(1)[0]
        intensity = self.get_intensity()
        # print('intensity', intensity)
        # if any(self.states[self.curr_episode][-1] == state):
        #     self.boundary += 1
        #     print('boundary hit nr: ', self.boundary)
        # else:
        #     self.boundary = -1

        if intensity > .95:
            self.done = True
            # print('passed at', intensity)
        # elif self.boundary > 10:
        #     self.done = True
        elif self.current_length >= self.max_steps:
            # print('failed at', intensity)
            self.done = True

        # elif any(self.state + action)<0 or any(self.state + action)>1:
        #     self.done = True

        # print('step:')
        # print()
        # # print('s ', state)

        ########################################################################################################
        # print(self.curr_episode, self.current_length, 'state ', state, 'a ', action, 'r ', reward)
        ########################################################################################################
        if self.test:
            self.add_trajectory_data(state=state, action=action, reward=reward, done=self.done)
        # if self.done:
        #     print('done at ', reward)
        return state, reward, self.done, {}

    def take_action(self, action):
        # print('action inner: ', np.round(action*12,2))
        # action /= 12
        # take action method
        new_state = self.state + action  # + 0.05*np.random.randn(action.shape[-1])
        # state must remain in [0, 1]
        if any(new_state < 0.0) or any(new_state > 1.0):
            new_state = np.clip(new_state, 0.0, 1.0)
            # self.done = True
            # print('WARNING: state boundaries!')

        # set new state to the machine
        self.set_state(new_state)
        state = self.get_state()
        self.state = state

        # get new intensity from the machine
        intensity = self.get_intensity()
        self.intensity = intensity

        # reward calculation
        reward = self.get_reward()
        self.reward = reward

        return state, reward

    def get_reward(self):
        # You can change reward function, but it should depend on intensity
        # e.g. next line
        reward = -(1 - self.intensity / self.target_intensity)

        # reward = self.intensity
        return reward

    def reset(self, **kwargs):
        # print('reset true env')
        self.boundary = -1
        # reset method
        self.done = False
        self.current_length = 0
        # self.curr_step = 0

        bad_init = True
        while bad_init:
            if 'set_state' in kwargs:
                new_state = kwargs.get('set_state')
                print('set_state')
            else:
                new_state = self.observation_space.sample()

            self.set_state(new_state)
            state = self.get_state()
            self.state = state

            intensity = self.get_intensity()
            self.intensity = intensity
            # bad_init = False if -(1 - self.intensity / self.target_intensity) > -1 else True
            reward = -(1 - self.intensity / self.target_intensity)
            self.init_rewards.append(reward)
            bad_init = False

            done = self.intensity > .95
            action = np.zeros(self.action_space.shape)
        self.curr_episode += 1
        if self.test:
            # self.curr_episode += 1
            self.rewards.append([])
            self.actions.append([])
            self.states.append([])
            self.dones.append([])
            self.add_trajectory_data(state=state, action=action, reward=reward, done=done)
            # print('reset',self.dones)

        # print('\n init:', -(1 - self.intensity / self.target_intensity))
        # return 2 * (state - 0.5)
        return state

    def add_trajectory_data(self, state, action, reward, done):
        self.rewards[self.curr_episode].append(reward)
        self.actions[self.curr_episode].append(action)
        self.states[self.curr_episode].append(state)
        self.dones[self.curr_episode].append(done)

    def seed(self, seed=None):
        # seed method
        np.random.seed(seed)

    def render(self, mode='human'):
        # render method
        print('ERROR\nnot yet implemented!')
        pass

    def store_trajectories_to_pkl(self, name, directory):
        out_put_writer = open(directory + name, 'wb')
        pickle.dump(self.states, out_put_writer, -1)
        pickle.dump(self.actions, out_put_writer, -1)
        pickle.dump(self.rewards, out_put_writer, -1)
        pickle.dump(self.dones, out_put_writer, -1)
        out_put_writer.close()


if __name__ == '__main__':
    import scipy.optimize as opt

    tng = SimTangoConnection()
    env = FelLocalEnv(tng)
    low = env.action_space.low
    high = env.action_space.high


    def normalize(input, box):
        low = tf.convert_to_tensor(box.low, dtype=tf.float64)
        high = tf.convert_to_tensor(box.high, dtype=tf.float64)
        return tf.math.scalar_mul(tf.convert_to_tensor(2, dtype=tf.float64),
                                  tf.math.add(tf.convert_to_tensor(-0.5, dtype=tf.float64),
                                              tf.multiply(tf.math.add(input, -low), 1 / (high - low))))


    def de_normalize(input, box):
        low = tf.convert_to_tensor(box.low, dtype=tf.float64)
        high = tf.convert_to_tensor(box.high, dtype=tf.float64)
        return tf.math.add(
            tf.multiply(tf.math.add(tf.math.scalar_mul(tf.convert_to_tensor(1 / 2, dtype=tf.float64), input),
                                    tf.convert_to_tensor(0.5, dtype=tf.float64)),
                        (high - low)), low)


    # print((env.action_space.sample() - low)/(high-low))
    # print('')
    # for _ in range(1):
    #     s = env.reset()
    #     a = env.action_space.sample()
    #     box = env.action_space
    #     # ns, r = env.step(a)
    #     print(a)
    #     print(normalize(a, box=box))
    #     print(de_normalize(normalize(a, box=box), box=box))
    #     # print(env.action_space.low)
    #     # print('state:', env.descale(s))
    #     # # print(a)
    #     # print('new state:', env.descale(ns))
    #     # print('reward:', r)
    #     # print('')
    class WrappedEnv(gym.Wrapper):
        def __init__(self, env, **kwargs):
            gym.Wrapper.__init__(self, env)
            self.current_action = np.zeros(env.action_space.shape[0])

        def reset(self, **kwargs):
            self.current_obs = self.env.reset(**kwargs)
            return self.current_obs

        def step(self, action):
            self.env.state = self.current_obs
            ob, reward, done, info = self.env.step(action)
            return ob, reward, done, info


    environment_instance = WrappedEnv(env=env)

    rews = []
    actions = []
    states = []


    def objective(action):
        actions.append(action.copy())
        _, r, _, _ = environment_instance.step(action=action.copy())
        rews.append(abs(r))
        return abs(r)


    if True:

        def constr(action):
            if any(action > environment_instance.action_space.high[0]):
                return -1
            elif any(action < environment_instance.action_space.low[0]):
                return -1
            else:
                return 1

        init = environment_instance.reset()
        print('init: ', init)
        start_vector = np.zeros(environment_instance.action_space.shape[0])
        # rhobeg = 1 * environment_instance.action_space.high[0]
        # print('rhobeg: ', rhobeg)
        # res = opt.fmin_cobyla(objective, start_vector, [constr], rhobeg=rhobeg, rhoend=.001)
        # constr = {'type': 'ineq', 'fun': lambda x: any(abs(x) > 1/12)}
        # minimizer_kwargs = {"method": "COBYLA", "constraints": constr}
        # res = opt.basinhopping(objective, start_vector, minimizer_kwargs=minimizer_kwargs)
        # print(res)
        upper = environment_instance.action_space.high*12
        lower = environment_instance.action_space.low*12
        soln = pybobyqa.solve(objective, start_vector, maxfun=500, bounds=(lower, upper),
                              rhobeg=1, seek_global_minimum=True)
        print(soln)

        fig, axs = plt.subplots(2, sharex=True)
        axs[1].plot(rews)

        pd.DataFrame(actions).plot(ax=axs[0])
        plt.show()
        environment_instance.state = init
        print(environment_instance.step(soln.x))