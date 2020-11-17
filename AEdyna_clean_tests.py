import os
import pickle
from datetime import datetime
import itertools as it
import gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# from stable_baselines.common.noise import NormalActionNoise

# from stable_baselines.sac.policies import MlpPolicy
# from stable_baselines import SAC as Agent
# from stable_baselines.td3.policies import MlpPolicy
# from stable_baselines import TD3 as Agent
from stable_baselines.common.policies import MlpPolicy
from stable_baselines import PPO2 as Agent

import tensorflow as tf

from local_fel_simulated_env import FelLocalEnv
from simulated_tango import SimTangoConnection

# from naf2 import NAF

# set random seed
random_seed = 111
np.random.seed(random_seed)
config = tf.ConfigProto(
    device_count={'GPU': 0}
)
# config = None

tango = SimTangoConnection()
real_env = FelLocalEnv(tango=tango)

# Hyper papameters
steps_per_env = 25
init_random_steps = 100
total_steps = 250
num_epochs = int((total_steps - init_random_steps) / steps_per_env) + 1

print('Number of epochs: ', num_epochs)

hidden_sizes = [100, 100]

max_training_iterations = 50
delay_before_convergence_check = 1

algorithm = 'SAC'

# minibatch_size = 100
simulated_steps = 3000

model_batch_size = 5
num_ensemble_models = 5

early_stopping = True
model_iter = 30

# model_training_iterations = 10
network_size = 15
# Set the priors for the anchor method:
#  TODO: How to set these correctly?
init_params = dict(init_stddev_1_w=np.sqrt(1),
                   init_stddev_1_b=np.sqrt(1),
                   init_stddev_2_w=1 / np.sqrt(network_size))

data_noise = 0.00000  # estimated noise variance
lambda_anchor = data_noise / (np.array([init_params['init_stddev_1_w'],
                                        init_params['init_stddev_1_b'],
                                        init_params['init_stddev_2_w']]) ** 2)

# How often to check the progress of the network training
# e.g. lambda it, episode: (it + 1) % max(3, (ep+1)*2) == 0
dynamic_wait_time = lambda it, ep: (it + 1) % 1 == 0  #

# Learning rate as function of ep
lr_start = 1e-3
lr_end = 1e-3
lr = lambda ep: max(lr_start + ep / 30 * (lr_end - lr_start), lr_end)

# Create the logging directory:
project_directory = 'Data_logging/Simulation/'

hyp_str_all = '-nr_steps_' + str(steps_per_env) + '-cr_lr' + '-n_ep_' + str(num_epochs) + \
              '-m_bs_' + str(model_batch_size) + \
              '-sim_steps_' + str(simulated_steps) + \
              '-m_iter_' + str(model_iter) + '-ensnr_' + str(num_ensemble_models) + '-init_' + str(
    init_random_steps) + '/'
project_directory = project_directory + hyp_str_all

# To label the plots:
hyp_str_all = '-nr_steps_' + str(steps_per_env) + '-n_ep_' + str(num_epochs) + \
              '-m_bs_' + str(model_batch_size) + \
              '-sim_steps_' + str(simulated_steps) + \
              '-m_iter_' + str(model_iter) + \
              '\n-ensnr_' + str(num_ensemble_models)

if not os.path.isdir(project_directory):
    os.makedirs(project_directory)
    print("created folder : ", project_directory)


# Class for data storage during the tests
class TrajectoryBuffer():
    '''Class for data storage during the tests'''

    def __init__(self, name, directory):
        self.save_frequency = 100000
        self.directory = directory
        self.name = name
        self.rews = []
        self.obss = []
        self.acts = []
        self.dones = []
        self.info = ""
        self.idx = -1

    def new_trajectory(self, obs):
        self.idx += 1
        self.rews.append([])
        self.acts.append([])
        self.obss.append([])
        self.dones.append([])
        self.store_step(obs=obs)

    def store_step(self, obs=None, act=None, rew=None, done=None):
        self.rews[self.idx].append(rew)
        self.obss[self.idx].append(obs)
        self.acts[self.idx].append(act)
        self.dones[self.idx].append(done)

        if self.__len__() % self.save_frequency == 0:
            self.save_buffer()

    def __len__(self):
        assert (len(self.rews) == len(self.obss) == len(self.acts) == len(self.dones))
        return len(self.obss)

    def save_buffer(self, **kwargs):
        if 'info' in kwargs:
            self.info = kwargs.get('info')
        now = datetime.now()
        # clock_time = "{}_{}_{}_{}_".format(now.day, now.hour, now.minute, now.second)
        clock_time = f'{now.month:0>2}_{now.day:0>2}_{now.hour:0>2}_{now.minute:0>2}_{now.second:0>2}_'
        data = dict(obss=self.obss,
                    acts=self.acts,
                    rews=self.rews,
                    dones=self.dones,
                    info=self.info)
        # print('saving...', data)
        out_put_writer = open(self.directory + clock_time + self.name, 'wb')
        pickle.dump(data, out_put_writer, -1)
        # pickle.dump(self.actions, out_put_writer, -1)
        out_put_writer.close()

    def get_data(self):
        return dict(obss=self.obss,
                    acts=self.acts,
                    rews=self.rews,
                    dones=self.dones,
                    info=self.info)


class MonitoringEnv(gym.Wrapper):
    '''
    Gym Wrapper to store information for scaling to correct scpace and for post analysis.
    '''

    def __init__(self, env, **kwargs):
        gym.Wrapper.__init__(self, env)
        self.data_dict = dict()
        self.environment_usage = 'default'
        self.directory = project_directory
        self.data_dict[self.environment_usage] = TrajectoryBuffer(name=self.environment_usage,
                                                                  directory=self.directory)
        self.current_buffer = self.data_dict.get(self.environment_usage)

        self.test_env_flag = False

        self.obs_dim = self.env.observation_space.shape
        self.obs_high = self.env.observation_space.high
        self.obs_low = self.env.observation_space.high
        self.act_dim = self.env.action_space.shape
        self.act_high = self.env.action_space.high
        self.act_low = self.env.action_space.low

        # state space definition
        self.observation_space = gym.spaces.Box(low=-1.0,
                                                high=1.0,
                                                shape=self.obs_dim,
                                                dtype=np.float64)

        # action space definition
        self.action_space = gym.spaces.Box(low=-1.0,
                                           high=1.0,
                                           shape=self.act_dim,
                                           dtype=np.float64)

        # if 'test_env' in kwargs:
        #     self.test_env_flag = True
        self.verification = False
        if 'verification' in kwargs:
            self.verification = kwargs.get('verification')

    def reset(self, **kwargs):
        init_obs = self.env.reset(**kwargs)
        # print('Reset Env: ', (init_obs),10*'-- ')
        self.current_buffer.new_trajectory(init_obs)
        init_obs = self.scale_state_env(init_obs)
        # print('Reset Menv: ', (init_obs))
        return init_obs

    def step(self, action):
        # print('a', action)
        action = self.descale_action_env(action)
        # print('as', action)
        ob, reward, done, info = self.env.step(action)
        # print('Env: ', reward)
        # print('Env: ', ob, 'r:', reward, done)
        self.current_buffer.store_step(obs=ob, act=action, rew=reward, done=done)
        ob = self.scale_state_env(ob)
        reward = self.rew_scale(reward)
        # print('Menv: ',  ob, 'r:', reward, done)
        # print('Menv: ', reward)
        return ob, reward, done, info

    def set_usage(self, usage):
        self.environment_usage = usage
        if usage in self.data_dict:
            self.current_buffer = self.data_dict.get(usage)
        else:
            self.data_dict[self.environment_usage] = TrajectoryBuffer(name=self.environment_usage,
                                                                      directory=self.directory)
            self.current_buffer = self.data_dict.get(usage)

    def close_usage(self, usage):
        # Todo: Implement to save complete data
        self.current_buffer = self.data_dict.get(usage)
        self.current_buffer.save_buffer()

    def scale_state_env(self, ob):
        scale = (self.env.observation_space.high - self.env.observation_space.low)
        return (2 * ob - (self.env.observation_space.high + self.env.observation_space.low)) / scale
        # return ob

    def descale_action_env(self, act):
        scale = (self.env.action_space.high - self.env.action_space.low)
        return (scale * act + self.env.action_space.high + self.env.action_space.low) / 2
        # return act

    def rew_scale(self, rew):
        # we only scale for the network training:
        # if not self.test_env_flag:
        #     rew = rew * 2 + 1

        if not self.verification:
            '''Rescale reward from [-1,0] to [-1,1] for the training of the network in case of tests'''
            rew = rew * 2 + 1
            # pass
        #     if rew < -1:
        #         print('Hallo was geht: ', rew)
        #     else:
        #         print('Okay...', rew)
        return rew

    def save_current_buffer(self, info=''):
        self.current_buffer = self.data_dict.get(self.environment_usage)
        self.current_buffer.save_buffer(info=info)
        print('Saved current buffer', self.environment_usage)

    def set_directory(self, directory):
        self.directory = directory


def make_env(**kwargs):
    '''Create the environement'''
    return MonitoringEnv(env=real_env, **kwargs)


def flatten_list(tensor_list):
    '''
    Flatten a list of tensors
    '''
    return tf.concat([flatten(t) for t in tensor_list], axis=0)


def flatten(tensor):
    '''
    Flatten a tensor
    '''
    return tf.reshape(tensor, shape=(-1,))


def test_agent(env_test, agent_op, num_games=10):
    '''
    Test an agent 'agent_op', 'num_games' times
    Return mean and std
    '''
    games_r = []
    games_length = []
    games_dones = []
    for _ in range(num_games):
        d = False
        game_r = 0
        o = env_test.reset()
        game_length = 0
        while not d:
            try:
                a_s, _ = agent_op([o])
            except:
                a_s, _ = agent_op(o)
            o, r, d, _ = env_test.step(a_s)
            game_r += r
            game_length += 1
            # print(o, a_s, r)
        success = r > -0.05
        # print(r)
        games_r.append(game_r)
        games_length.append(success)
        games_dones.append(d)
    return np.mean(games_r), np.std(games_r), np.mean(games_length), np.mean(games_dones)


class FullBuffer():
    def __init__(self):
        self.rew = []
        self.obs = []
        self.act = []
        self.nxt_obs = []
        self.done = []

        self.train_idx = []
        self.valid_idx = []
        self.idx = 0

    def store(self, obs, act, rew, nxt_obs, done):
        self.rew.append(rew)
        self.obs.append(obs)
        self.act.append(act)
        self.nxt_obs.append(nxt_obs)
        self.done.append(done)

        self.idx += 1

    def generate_random_dataset(self, ratio=False):
        '''ratio: how much for valid taken'''
        rnd = np.arange(len(self.obs))
        np.random.shuffle(rnd)
        self.valid_idx = rnd[:]
        self.train_idx = rnd[:]  # change back
        if ratio:
            self.valid_idx = rnd[: int(len(self.obs) * ratio)]
            self.train_idx = rnd[int(len(self.obs) * ratio):]

        print('Train set:', len(self.train_idx), 'Valid set:', len(self.valid_idx))

    def get_training_batch(self):
        return np.array(self.obs)[self.train_idx], np.array(np.expand_dims(self.act, -1))[self.train_idx], \
               np.array(self.rew)[self.train_idx], np.array(self.nxt_obs)[self.train_idx], np.array(self.done)[
                   self.train_idx]

    def get_valid_batch(self):
        return np.array(self.obs)[self.valid_idx], np.array(np.expand_dims(self.act, -1))[self.valid_idx], \
               np.array(self.rew)[self.valid_idx], np.array(self.nxt_obs)[self.valid_idx], np.array(self.done)[
                   self.valid_idx]

    def get_maximum(self):
        idx = np.argmax(self.rew)
        print('rew', np.array(self.rew)[idx])
        return np.array(self.obs)[idx], np.array(np.expand_dims(self.act, -1))[idx], \
               np.array(self.rew)[idx], np.array(self.nxt_obs)[idx], np.array(self.done)[
                   idx]

    def __len__(self):
        assert (len(self.rew) == len(self.obs) == len(self.act) == len(self.nxt_obs) == len(self.done))
        return len(self.obs)


class NN:
    def __init__(self, x, y, y_dim, hidden_size, n, learning_rate, init_params):
        self.init_params = init_params

        # set up NN
        with tf.variable_scope('model_' + str(n) + '_nn'):
            self.inputs = x
            self.y_target = y
            if True:
                self.inputs = tf.scalar_mul(0.8, self.inputs)
                self.layer_1_w = tf.layers.Dense(hidden_size,
                                                 activation=tf.nn.tanh,
                                                 kernel_initializer=tf.random_normal_initializer(mean=0.,
                                                                                                 stddev=self.init_params.get(
                                                                                                     'init_stddev_1_w'),
                                                                                                 dtype=tf.float64),
                                                 bias_initializer=tf.random_normal_initializer(mean=0.,
                                                                                               stddev=self.init_params.get(
                                                                                                   'init_stddev_1_b'),
                                                                                               dtype=tf.float64))

                self.layer_1 = self.layer_1_w.apply(self.inputs)

                self.layer_2_w = tf.layers.Dense(hidden_size,
                                                 activation=tf.nn.tanh,
                                                 kernel_initializer=tf.random_normal_initializer(mean=0.,
                                                                                                 stddev=self.init_params.get(
                                                                                                     'init_stddev_1_w'),
                                                                                                 dtype=tf.float64),
                                                 bias_initializer=tf.random_normal_initializer(mean=0.,
                                                                                               stddev=self.init_params.get(
                                                                                                   'init_stddev_1_b'),
                                                                                               dtype=tf.float64))

                self.layer_2 = self.layer_2_w.apply(self.layer_1)
                #
                self.output_w = tf.layers.Dense(y_dim,
                                                activation=None,
                                                use_bias=False,
                                                kernel_initializer=tf.random_normal_initializer(mean=0.,
                                                                                                stddev=self.init_params.get(
                                                                                                    'init_stddev_2_w'),
                                                                                                dtype=tf.float64))
            else:
                self.layer_1_w = tf.layers.Dense(hidden_size,
                                                 activation=tf.nn.tanh
                                                 )

                self.layer_1 = self.layer_1_w.apply(self.inputs)

                self.layer_2_w = tf.layers.Dense(hidden_size,
                                                 activation=tf.nn.tanh)

                self.layer_2 = self.layer_2_w.apply(self.layer_1)

                self.output_w = tf.layers.Dense(y_dim,
                                                activation=None)
                # #

            self.output = self.output_w.apply(self.layer_2)

            # set up loss and optimiser - we'll modify this later with anchoring regularisation
            self.opt_method = tf.train.AdamOptimizer(learning_rate)
            # self.mse_ = 1 / tf.shape(self.inputs, out_type=tf.int64)[0] * \
            #             tf.reduce_sum(tf.square(self.y_target - self.output))
            self.mse_ = tf.reduce_mean(((self.y_target - self.output)) ** 2)
            self.loss_ = 1 / tf.shape(self.inputs, out_type=tf.int64)[0] * \
                         tf.reduce_sum(tf.square(self.y_target - self.output))
            self.optimizer = self.opt_method.minimize(self.loss_)
            self.optimizer_mse = self.opt_method.minimize(self.mse_)

    def get_weights(self, sess):
        '''method to return current params'''

        ops = [self.layer_1_w.kernel, self.layer_1_w.bias,
               self.layer_2_w.kernel, self.layer_2_w.bias,
               self.output_w.kernel]
        w1, b1, w2, b2, w = sess.run(ops)

        return w1, b1, w2, b2, w

    # def get_weights(self, sess):
    #     '''method to return current params'''
    #
    #     ops = [self.layer_1_w.kernel, self.layer_1_w.bias,
    #            self.output_w.kernel]
    #     w1, b1, w = sess.run(ops)
    #
    #     return w1, b1, w

    def anchor(self, lambda_anchor, sess):
        '''regularise around initialised parameters after session has started'''

        w1, b1, w2, b2, w = self.get_weights(sess=sess)

        # get initial params to hold for future trainings
        self.w1_init, self.b1_init, self.w2_init, self.b2_init, self.w_out_init = w1, b1, w2, b2, w

        loss_anchor = lambda_anchor[0] * tf.reduce_sum(tf.square(self.w1_init - self.layer_1_w.kernel))
        loss_anchor += lambda_anchor[1] * tf.reduce_sum(tf.square(self.b1_init - self.layer_1_w.bias))

        loss_anchor = lambda_anchor[0] * tf.reduce_sum(tf.square(self.w2_init - self.layer_2_w.kernel))
        loss_anchor += lambda_anchor[1] * tf.reduce_sum(tf.square(self.b2_init - self.layer_2_w.bias))

        loss_anchor += lambda_anchor[2] * tf.reduce_sum(tf.square(self.w_out_init - self.output_w.kernel))

        # combine with original loss
        self.loss_ = self.loss_ + tf.scalar_mul(1 / tf.shape(self.inputs)[0], loss_anchor)
        self.optimizer = self.opt_method.minimize(self.loss_)
        return self.optimizer, self.loss_


class NetworkEnv(gym.Wrapper):
    '''
    Wrapper to handle the network interaction
    '''

    def __init__(self, env, model_func=None, done_func=None, number_models=1, **kwargs):
        gym.Wrapper.__init__(self, env)

        self.model_func = model_func
        self.done_func = done_func
        self.number_models = number_models
        self.len_episode = 0
        # self.threshold = self.env.threshold
        # print('the threshold is: ', self.threshold)
        self.max_steps = env.max_steps
        self.verification = False
        if 'verification' in kwargs:
            self.verification = kwargs.get('verification')
        self.visualize()

    def reset(self, **kwargs):

        # self.threshold = -0.05 * 2 + 1  # rescaled [-1,1]
        self.len_episode = 0
        self.done = False
        # kwargs['simulation'] = True
        # action = self.env.reset(**kwargs)
        if self.model_func is not None:
            obs = np.random.uniform(-1, 1, self.env.observation_space.shape)
            # print('reset', obs)
            # Todo: remove
            # self.obs = self.env.reset()
            # obs = self.env.reset()
        else:
            # obs = self.env.reset(**kwargs)
            pass
        # Does this work?
        self.obs = np.clip(obs, -1.0, 1.0)
        # self.obs = obs.copy()
        # if self.test_phase:
        #     print('test reset', self.obs)
        # print('Reset : ',self.obs)
        self.current_model = np.random.randint(0, max(self.number_models, 1))
        return self.obs

    def step(self, action):
        # self.visualize([np.squeeze(action)])
        if self.model_func is not None:
            # predict the next state on a random model
            # obs, rew = self.model_func(self.obs, [np.squeeze(action)], np.random.randint(0, self.number_models))

            if self.verification:
                # obs_cov = np.diag(np.square(np.std(obss, axis=0, ddof=1)) + data_noise)
                # print(obs_std)
                # obs = np.squeeze(np.random.multivariate_normal(obs_m, (obs_cov), 1))
                # obs, rew, done, info = self.env.step(
                #     action)  #
                obs, rew = self.model_func(self.obs, [np.squeeze(action)], self.number_models)
            else:
                # obs, rew, done, info = self.env.step(
                #     action)
                obs, rew = self.model_func(self.obs, [np.squeeze(action)], self.current_model)
                # obss = []
                # rews = []
                # for i in range(num_ensemble_models):
                #
                #     obs, rew = self.model_func(self.obs, [np.squeeze(action)], i)
                #     obss.append((obs))
                #     rews.append(rew)
                # # idx = np.argmin(rews)
                # # obs = obss[idx]
                # obs_m = np.mean(obss, axis=0)
                # obs = obs_m
            # print(obs)
            # rew = rews[idx]
            # rew = np.mean(np.clip(rews, -1, 1))
            self.obs = np.clip(obs.copy(), -1, 1)
            # if (self.obs == -1).any() or (self.obs == 1).any():
            #     rew = -1
            rew = np.clip(rew, -1, 1)
            if not self.verification:
                rew = (rew - 1) / 2

            # obs_real, rew_real, _, _ = self.env.step(action)
            # obs, rew, self.done, _ = self.env.step(action)
            # print('Diff: ', np.linalg.norm(obs - obs_real), np.linalg.norm(rew - rew_real))
            # print('MEnv: ', np.linalg.norm(obs ), np.linalg.norm(rew ))
            # obs += np.random.randn(obs.shape[-1])
            # # Todo: remove
            # self.env.state = self.obs
            # done = rew > self.threshold

            self.len_episode += 1
            # print('threshold at:', self.threshold)
            # For niky hardcoded reward threshold in [-1,1] space from [0,1] -0.05 => 0.9------------------------------------------------------------
            if rew > -0.05:  # self.threshold: TODO: to be changed
                # ----------------------------------------------------------------------------------------------------------------------
                self.done = True
            #     if (self.obs == -1).any() or (self.obs == 1).any():
            #         print('boundary hit...', self.obs, rew)
            #     # print("Done", rew)
            if self.len_episode >= self.max_steps:
                self.done = True
            #
            return self.obs, rew, self.done, dict()
            # return obs, rew, done, info
        else:
            # self.obs, rew, done, _ = real_env.step(action)
            # return self.obs, rew, done, ""
            pass
        # return env.step(action)

    def visualize(self, data=None, label=None):

        action = [np.zeros(self.env.action_space.shape)]
        state = np.zeros(self.env.observation_space.shape)
        maximum = 0
        if data is not None:
            action = [data[1]]
            state = data[0]
            maximum = (data[2] - 1) / 2
        delta = 0.05
        x = np.arange(-1, 1, delta)
        y = np.arange(-1, 1, delta)
        X, Y = np.meshgrid(x, y)

        if self.number_models == num_ensemble_models:
            Nr = 1
            Nc = 1
            fig, axs = plt.subplots(Nr, Nc)
            fig.subplots_adjust(hspace=0.3)
            images = []
            for nr in range(self.number_models):
                rewards = np.zeros(X.shape)

                # print(self.number_models)
                for i1 in range(len(x)):
                    for j1 in range(len(y)):
                        state[0] = x[i1]
                        state[1] = y[j1]
                        rewards[i1, j1] = (self.model_func(state, [np.squeeze(action)],
                                                           nr))[1] / num_ensemble_models
                axs.contour(X, Y, (rewards - 1) / 2, alpha=1)
                self.save_buffer(nr, data, X, Y, rewards)
            # list_combinations = list(it.combinations([0, 1, 2, 3], 2))
            #
            # for i in range(Nr):
            #     for j in range(Nc):
            #
            #         for nr in range(self.number_models):
            #             rewards = np.zeros(X.shape)
            #
            #             # print(self.number_models)
            #             for i1 in range(len(x)):
            #                 for j1 in range(len(y)):
            #                     current_pair = list_combinations[i * Nc + j]
            #                     state[current_pair[0]] = x[i1]
            #                     state[current_pair[1]] = y[j1]
            #                     rewards[i1, j1] = (self.model_func(state, [np.squeeze(action)],
            #                                                        nr))[1] / num_ensemble_models
            #             axs[i, j].contour(X, Y, (rewards - 1) / 2, alpha=1)
            #             # plt.plot(np.array(states, dtype=object)[:, 1],)
            #         # images.append(axs[i, j].contour(X, Y, (rewards - 1) / 2, 25, alpha=1))
            #         # axs[i, j].label_outer()
            plt.title(maximum)
            # plt.title(label)
            # plt.colorbar()
            fig.show()
        else:
            pass
            # action = [np.random.uniform(-1, 1, 4)]
            # state_vec = np.linspace(-1, 1, 100)
            # states = []
            # # print(self.number_models)
            #
            # for i in state_vec:
            #     states.append(self.model_func(np.array([i, 0, 0, 0]), action,
            #                                   self.number_models))
            #
            # plt.plot(np.array(states, dtype=object)[:, 1])

            # states = np.zeros(X.shape)
            # # print(self.number_models)
            # for i in range(len(x)):
            #     for j in range(len(y)):
            #         states[i, j] = (self.model_func(np.array([x[i], y[j], 0, 0]), action,
            #                                         self.number_models)[1])
            # plt.contourf(states)

    def save_buffer(self, model_nr, data, X, Y, rews, **kwargs):
        if 'info' in kwargs:
            self.info = kwargs.get('info')
        now = datetime.now()
        # clock_time = "{}_{}_{}_{}_".format(now.day, now.hour, now.minute, now.second)
        clock_time = f'{now.month:0>2}_{now.day:0>2}_{now.hour:0>2}_{now.minute:0>2}_{now.second:0>2}_'
        data = dict(data=data,
                    model=model_nr,
                    rews=rews,
                    X=X,
                    Y=Y)
        # print('saving...', data)
        out_put_writer = open(project_directory + clock_time + 'plot_model_' + str(model_nr), 'wb')
        pickle.dump(data, out_put_writer, -1)
        # pickle.dump(self.actions, out_put_writer, -1)
        out_put_writer.close()


class StructEnv(gym.Wrapper):
    '''
    Gym Wrapper to store information like number of steps and total reward of the last espisode.
    '''

    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self.n_obs = self.env.reset()
        self.total_rew = 0
        self.len_episode = 0

    def reset(self, **kwargs):
        self.n_obs = self.env.reset(**kwargs)
        self.total_rew = 0
        self.len_episode = 0
        return self.n_obs.copy()

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        # print('reward in struct', reward)
        self.total_rew += reward
        self.len_episode += 1
        return ob, reward, done, info

    def get_episode_reward(self):
        return self.total_rew

    def get_episode_length(self):
        return self.len_episode


def restore_model(old_model_variables, m_variables):
    # variable used as index for restoring the actor's parameters
    it_v2 = tf.Variable(0, trainable=False)

    restore_m_params = []
    for m_v in m_variables:
        upd_m_rsh = tf.reshape(old_model_variables[it_v2: it_v2 + tf.reduce_prod(m_v.shape)], shape=m_v.shape)
        restore_m_params.append(m_v.assign(upd_m_rsh))
        it_v2 += tf.reduce_prod(m_v.shape)

    return tf.group(*restore_m_params)


def aedyna(env_name, hidden_sizes=[32, 32], cr_lr=5e-3, num_epochs=50,
           critic_iter=10, steps_per_env=100, delta=0.05, algorithm='TRPO', conj_iters=10,
           simulated_steps=1000, num_ensemble_models=2, model_iter=15, model_batch_size=512,
           init_random_steps=steps_per_env):
    '''
    Anchor ensemble dyna reinforcement learning

    The states and actions are provided by the gym environment with the correct boxes.
    The reward has to be between [-1,0].

    Parameters:
    -----------
    env_name: Name of the environment
    hidden_sizes: list of the number of hidden units for each layer
    num_epochs: number of training epochs
    number_envs: number of "parallel" synchronous environments
        # NB: it isn't distributed across multiple CPUs
    steps_per_env: number of steps per environment
            # NB: the total number of steps per epoch will be: steps_per_env*number_envs
    algorithm: type of algorithm. Either 'TRPO' or 'NPO'
    minibatch_size: Batch size used to train the critic
    mb_lr: learning rate of the environment model
    model_batch_size: batch size of the environment model
    simulated_steps: number of simulated steps for each policy update
    model_iter: number of iterations without improvement before stopping training the model
    '''
    model_batch_size = model_batch_size
    tf.reset_default_graph()

    # Create a few environments to collect the trajectories
    env = StructEnv(make_env())
    env_test = StructEnv(make_env(verification=True))

    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape[0]

    # Placeholders for model
    act_ph = tf.placeholder(shape=(None, act_dim), dtype=tf.float64, name='act')
    obs_ph = tf.placeholder(shape=(None, obs_dim[0]), dtype=tf.float64, name='obs')
    # NEW
    nobs_ph = tf.placeholder(shape=(None, obs_dim[0]), dtype=tf.float64, name='nobs')
    rew_ph = tf.placeholder(shape=(None, 1), dtype=tf.float64, name='rew')

    # Placeholder for learning rate
    mb_lr_ = tf.placeholder("float", None)

    old_model_variables = tf.placeholder(shape=(None,), dtype=tf.float64, name='old_model_variables')

    def variables_in_scope(scope):
        # get all trainable variables in 'scope'
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)

    #########################################################
    ######################### MODEL #########################
    #########################################################

    m_opts = []
    m_losses = []

    nobs_pred_m = []
    act_obs = tf.concat([obs_ph, act_ph], 1)
    target = tf.concat([nobs_ph, rew_ph], 1)

    # computational graph of N models and the correct losses for the anchor method
    m_classes = []

    for i in range(num_ensemble_models):
        m_class = NN(x=act_obs, y=target, y_dim=obs_dim[0] + 1,
                     learning_rate=mb_lr_, n=i,
                     hidden_size=network_size, init_params=init_params)

        nobs_pred = m_class.output

        nobs_pred_m.append(nobs_pred)

        m_classes.append(m_class)
        m_losses.append(m_class.mse_)
        m_opts.append(m_class.optimizer_mse)

    ##################### RESTORE MODEL ######################
    initialize_models = []
    models_variables = []
    for i in range(num_ensemble_models):
        m_variables = variables_in_scope('model_' + str(i) + '_nn')
        initialize_models.append(restore_model(old_model_variables, m_variables))
        #  List of weights
        models_variables.append(flatten_list(m_variables))

    #########################################################
    ##################### END MODEL #########################
    #########################################################
    # Time
    now = datetime.now()
    clock_time = "{}_{}_{}_{}".format(now.day, now.hour, now.minute, now.second)
    print('Time:', clock_time)

    hyp_str = '-spe_' + str(steps_per_env) + '-cr_lr' + str(cr_lr) + '-crit_it_' + str(
        critic_iter) + '-delta_' + str(delta) + '-conj_iters_' + str(conj_iters)

    file_writer = tf.summary.FileWriter('log_dir/' + env_name + '/' + algorithm + '_' + clock_time + '_' + hyp_str,
                                        tf.get_default_graph())

    #################################################################################################

    # Session start!!!!!!!!
    # create a session
    sess = tf.Session(config=config)
    # initialize the variables
    sess.run(tf.global_variables_initializer())

    def model_op(o, a, md_idx):
        mo = sess.run(nobs_pred_m[md_idx], feed_dict={obs_ph: [o], act_ph: [a[0]]})
        return np.squeeze(mo[:, :-1]), float(np.squeeze(mo[:, -1]))

    def run_model_loss(model_idx, r_obs, r_act, r_nxt_obs, r_rew):
        # print({'obs_ph': r_obs.shape, 'act_ph': r_act.shape, 'nobs_ph': r_nxt_obs.shape})
        r_act = np.squeeze(r_act, axis=2)
        # print(r_act.shape)
        r_rew = np.reshape(r_rew, (-1, 1))
        # print(r_rew.shape)
        return_val = sess.run(m_loss_anchor[model_idx],
                              feed_dict={obs_ph: r_obs, act_ph: r_act, nobs_ph: r_nxt_obs, rew_ph: r_rew})
        return return_val

    def run_model_opt_loss(model_idx, r_obs, r_act, r_nxt_obs, r_rew, mb_lr):
        r_act = np.squeeze(r_act, axis=2)
        r_rew = np.reshape(r_rew, (-1, 1))
        # return sess.run([m_opts[model_idx], m_losses[model_idx]],
        #                 feed_dict={obs_ph: r_obs, act_ph: r_act, nobs_ph: r_nxt_obs, rew_ph: r_rew, mb_lr_: mb_lr})
        return sess.run([m_opts_anchor[model_idx], m_loss_anchor[model_idx]],
                        feed_dict={obs_ph: r_obs, act_ph: r_act, nobs_ph: r_nxt_obs, rew_ph: r_rew, mb_lr_: mb_lr})

    def model_assign(i, model_variables_to_assign):
        '''
        Update the i-th model's parameters
        '''
        return sess.run(initialize_models[i], feed_dict={old_model_variables: model_variables_to_assign})

    def train_model(tr_obs, tr_act, tr_nxt_obs, tr_rew, v_obs, v_act, v_nxt_obs, v_rew, step_count, model_idx, mb_lr):

        # Get validation loss on the old model only used for monitoring
        mb_valid_loss1 = run_model_loss(model_idx, v_obs, v_act, v_nxt_obs, v_rew)

        # Restore the initial random weights to have a new, clean neural network
        # initial_variables_models - list stored before already in the code below -
        # important for the anchor method
        model_assign(model_idx, initial_variables_models[model_idx])

        # Get validation loss on the now initialized model
        mb_valid_loss = run_model_loss(model_idx, v_obs, v_act, v_nxt_obs, v_rew)

        acc_m_losses = []

        md_params = sess.run(models_variables[model_idx])
        best_mb = {'iter': 0, 'loss': mb_valid_loss, 'params': md_params}
        it = 0

        # Create mini-batch for training
        lb = len(tr_obs)

        shuffled_batch = np.arange(lb)
        np.random.shuffle(shuffled_batch)

        if not early_stopping:
            # model_batch_size = lb
            # Take a fixed accuracy
            not_converged = True
            while not_converged:

                # update the model on each mini-batch
                last_m_losses = []
                for idx in range(0, lb, lb):
                    minib = shuffled_batch

                    _, ml = run_model_opt_loss(model_idx, tr_obs[minib], tr_act[minib], tr_nxt_obs[minib],
                                               tr_rew[minib], mb_lr=mb_lr)
                    acc_m_losses.append(ml)
                    last_m_losses.append(ml)
                    mb_valid_loss = run_model_loss(model_idx, v_obs, v_act, v_nxt_obs, v_rew)
                    # mb_lr
                    if mb_valid_loss < max(mb_lr, 1e-4) or it > 1e5:
                        not_converged = False
                    it += 1

            best_mb['loss'] = mb_valid_loss
            best_mb['iter'] = it
            # store the parameters to the array
            best_mb['params'] = sess.run(models_variables[model_idx])
        else:
            # Run until the number of model_iter has passed from the best val loss at it on...
            # ml = 1
            # while not (best_mb['iter'] < it - model_iter and ml < 5e-3):
            while best_mb['iter'] > it - model_iter:
                # update the model on each mini-batch
                last_m_losses = []
                for idx in range(0, lb, model_batch_size):
                    minib = shuffled_batch[idx:min(idx + model_batch_size, lb)]
                    _, ml = run_model_opt_loss(model_idx, tr_obs[minib], tr_act[minib], tr_nxt_obs[minib],
                                               tr_rew[minib], mb_lr=mb_lr)
                    acc_m_losses.append(ml)
                    last_m_losses.append(ml)

                # Check if the loss on the validation set has improved
                mb_valid_loss = run_model_loss(model_idx, v_obs, v_act, v_nxt_obs, v_rew)

                if mb_valid_loss < best_mb['loss']:
                    best_mb['loss'] = mb_valid_loss
                    best_mb['iter'] = it
                    # store the parameters to the array
                    best_mb['params'] = sess.run(models_variables[model_idx])

                it += 1

        # Restore the model with the lower validation loss
        model_assign(model_idx, best_mb['params'])

        print('Model:{}, iter:{} -- Old Val loss:{:.6f}  New Val loss:{:.6f} -- '
              'New Train loss:{:.6f} -- Loss_data {:.6f}'.format(model_idx,
                                                                 it,
                                                                 mb_valid_loss1,
                                                                 best_mb[
                                                                     'loss'],
                                                                 np.mean(
                                                                     last_m_losses), ml))
        summary = tf.Summary()
        summary.value.add(tag='supplementary/m_loss', simple_value=np.mean(acc_m_losses))
        summary.value.add(tag='supplementary/iterations', simple_value=it)
        file_writer.add_summary(summary, step_count)
        file_writer.flush()

    def plot_results(env_wrapper, label, **kwargs):
        # plotting
        print('now plotting...')
        rewards = env_wrapper.env.current_buffer.get_data()['rews']

        # initial_states = env.initial_conditions

        iterations = []
        finals = []
        means = []
        stds = []

        # init_states = pd.read_pickle('/Users/shirlaen/PycharmProjects/DeepLearning/spinningup/Environments/initData')

        for i in range(len(rewards)):
            if (len(rewards[i]) > 1):
                # finals.append(rewards[i][len(rewards[i]) - 1])
                finals.append(rewards[i][-1])
                means.append(np.mean(rewards[i][1:]))
                stds.append(np.std(rewards[i][1:]))
                iterations.append(len(rewards[i]))
        # print(iterations)
        x = range(len(iterations))
        iterations = np.array(iterations)
        finals = np.array(finals)
        means = np.array(means)
        stds = np.array(stds)

        plot_suffix = label  # , Fermi time: {env.TOTAL_COUNTER / 600:.1f} h'

        fig, axs = plt.subplots(2, 1, sharex=True)

        ax = axs[0]
        ax.plot(x, iterations)
        ax.set_ylabel('Iterations (1)')
        ax.set_title(plot_suffix)
        # fig.suptitle(label, fontsize=12)
        if 'data_number' in kwargs:
            ax1 = plt.twinx(ax)
            color = 'lime'
            ax1.set_ylabel('Mean reward', color=color)  # we already handled the x-label with ax1
            ax1.tick_params(axis='y', labelcolor=color)
            ax1.plot(x, kwargs.get('data_number'), color=color)

        ax = axs[1]
        color = 'blue'
        ax.set_ylabel('Final reward', color=color)  # we already handled the x-label with ax1
        ax.tick_params(axis='y', labelcolor=color)
        ax.plot(x, finals, color=color)

        ax.set_title('Final reward per episode')  # + plot_suffix)
        ax.set_xlabel('Episodes (1)')

        ax1 = plt.twinx(ax)
        color = 'lime'
        ax1.set_ylabel('Mean reward', color=color)  # we already handled the x-label with ax1
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.fill_between(x, means - stds, means + stds,
                         alpha=0.5, edgecolor=color, facecolor='#FF9848')
        ax1.plot(x, means, color=color)

        # ax.set_ylim(ax1.get_ylim())
        if 'save_name' in kwargs:
            plt.savefig(kwargs.get('save_name') + '.pdf')
        # fig.tight_layout()
        plt.show()

    def plot_observables(data, label, **kwargs):
        """plot observables during the test"""

        sim_rewards_all = np.array(data.get('sim_rewards_all'))
        step_counts_all = np.array(data.get('step_counts_all'))
        batch_rews_all = np.array(data.get('batch_rews_all'))
        tests_all = np.array(data.get('tests_all'))

        fig, axs = plt.subplots(2, 1, sharex=True)
        x = np.arange(len(batch_rews_all[0]))
        ax = axs[0]
        ax.step(x, batch_rews_all[0])
        ax.fill_between(x, batch_rews_all[0] - batch_rews_all[1], batch_rews_all[0] + batch_rews_all[1],
                        alpha=0.5)
        ax.set_ylabel('rews per batch')

        ax.set_title(label)

        # plt.tw
        ax2 = ax.twinx()

        color = 'lime'
        ax2.set_ylabel('data points', color=color)  # we already handled the x-label with ax1
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
        ax.set_ylabel('rewards tests')
        # plt.tw
        ax.grid(True)
        ax2 = ax.twinx()

        color = 'lime'
        ax2.set_ylabel('success', color=color)  # we already handled the x-label with ax1
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.plot(length_all, color=color)
        fig.align_labels()
        plt.show()

    def save_data(data, **kwargs):
        '''logging function'''
        now = datetime.now()
        clock_time = f'{now.month:0>2}_{now.day:0>2}_{now.hour:0>2}_{now.minute:0>2}_{now.second:0>2}'
        out_put_writer = open(project_directory + clock_time + '_training_observables', 'wb')
        pickle.dump(data, out_put_writer, -1)
        out_put_writer.close()

    # variable to store the total number of steps
    step_count = 0
    model_buffer = FullBuffer()
    print('Env batch size:', steps_per_env, ' Batch size:', steps_per_env)

    # Create a simulated environment
    sim_env = NetworkEnv(make_env(), model_op, None, num_ensemble_models)

    # ------------------------------------------------------------------------------------------------------
    # -------------------------------------Try to set correct anchors---------------------------------------
    # Get the initial parameters of each model
    # These are used in later epochs when we aim to re-train the models anew with the new dataset
    initial_variables_models = []
    for model_var in models_variables:
        initial_variables_models.append(sess.run(model_var))

    # update the anchor model losses:
    m_opts_anchor = []
    m_loss_anchor = []
    for i in range(num_ensemble_models):
        opt, loss = m_classes[i].anchor(lambda_anchor=lambda_anchor, sess=sess)
        m_opts_anchor.append(opt)
        m_loss_anchor.append(loss)

    # ------------------------------------------------------------------------------------------------------
    # -------------------------------------Try to set correct anchors---------------------------------------

    total_iterations = 0

    sim_rewards_all = []
    sim_rewards_std_all = []
    length_all = []
    tests_all = []
    tests_std_all = []
    batch_rews_all = []
    batch_rews_std_all = []
    step_counts_all = []

    agent = Agent(MlpPolicy, sim_env, verbose=1)
    for ep in range(num_epochs):

        # lists to store rewards and length of the trajectories completed
        batch_rew = []
        batch_len = []
        print('============================', ep, '============================')
        # Execute in serial the environment, storing temporarily the trajectories.

        # Todo: Test randomization stronger if reward lower...we need a good scheme
        env.reset()

        # iterate over a fixed number of steps
        steps_train = init_random_steps if ep == 0 else steps_per_env

        for _ in range(steps_train):
            # run the policy

            if ep == 0:
                # Sample random action during the first epoch
                if step_count % 5 == 0:
                    env.reset()
                act = np.random.uniform(-1, 1, size=env.action_space.shape[-1])

            else:

                # act = sess.run(a_sampl, feed_dict={obs_ph: [env.n_obs], log_std: init_log_std})
                # act = np.clip(act + np.random.randn(act.shape[0], act.shape[1]) * 0.1, -1, 1)
                act, _ = agent.predict(env.n_obs)

            act = np.squeeze(act)
            # take a step in the environment
            obs2, rew, done, _ = env.step(np.array(act))

            # add the new transition to the temporary buffer
            model_buffer.store(env.n_obs.copy(), act, rew.copy(), obs2.copy(), done)

            env.n_obs = obs2.copy()
            step_count += 1

            if done:
                batch_rew.append(env.get_episode_reward())
                batch_len.append(env.get_episode_length())

                env.reset()

        # save the data for plotting the collected data for the model
        env.save_current_buffer()

        print('Ep:%d Rew:%.2f -- Step:%d' % (ep, np.mean(batch_rew), step_count))

        # env_test.env.set_usage('default')
        # plot_results(env_test, f'Total {total_iterations}, '
        #                        f'data points: {len(model_buffer.train_idx) + len(model_buffer.valid_idx)}, '
        #                        f'modelit: {ep}')

        ############################################################
        ###################### MODEL LEARNING ######################
        ############################################################

        target_threshold = max(model_buffer.rew)
        sim_env.threshold = target_threshold  # min(target_threshold, -0.05)
        print('maximum: ', sim_env.threshold)

        mb_lr = lr(ep)
        print('mb_lr: ', mb_lr)
        if early_stopping:
            model_buffer.generate_random_dataset(ratio=0.1)
        else:
            model_buffer.generate_random_dataset()

        for i in range(num_ensemble_models):
            # Initialize randomly a training and validation set

            # get both datasets
            train_obs, train_act, train_rew, train_nxt_obs, _ = model_buffer.get_training_batch()
            valid_obs, valid_act, valid_rew, valid_nxt_obs, _ = model_buffer.get_valid_batch()
            # train the dynamic model on the datasets just sampled
            train_model(train_obs, train_act, train_nxt_obs, train_rew, valid_obs, valid_act, valid_nxt_obs, valid_rew,
                        step_count, i, mb_lr=mb_lr)

        ############################################################
        ###################### POLICY LEARNING ######################
        ############################################################
        data = model_buffer.get_maximum()
        print(data)
        label = f'Total {total_iterations}, ' + \
                f'data points: {len(model_buffer)}, ' + \
                f'ep: {ep}, max: {data}\n' + hyp_str_all
        sim_env.visualize(data=data, label=label)
        # sim_env.visualize()
        best_sim_test = -1e16 * np.ones(num_ensemble_models)
        agent = Agent(MlpPolicy, sim_env, verbose=1)
        for it in range(max_training_iterations):
            total_iterations += 1
            print('\t Policy it', it, end='..')

            ################# Agent UPDATE ################
            agent.learn(total_timesteps=simulated_steps, log_interval=1000, reset_num_timesteps=False)
            # Testing the policy on a real environment
            # summary = tf.Summary()
            # summary.value.add(tag='test/performance', simple_value=mn_test)
            # file_writer.add_summary(summary, step_count)
            # file_writer.flush()

            # Test the policy on simulated environment.
            # dynamic_wait_time_count = dynamic_wait_time(ep)
            if dynamic_wait_time(it, ep):
                print('Iterations: ', total_iterations)

                # for niky perform test! -----------------------------
                # env_test.env.set_usage('test')
                #
                mn_test, mn_test_std, mn_length, mn_success = test_agent(env_test, agent.predict, num_games=50)
                # # perform test! -----------------------------
                label = f'Total {total_iterations}, ' + \
                        f'data points: {len(model_buffer)}, ' + \
                        f'ep: {ep}, it: {it}\n' + hyp_str_all
                #
                # # for niky plot results of test -----------------------------
                # # plot_results(env_test, label=label)
                #
                env_test.save_current_buffer(info=label)

                # print(' Test score: ', np.round(mn_test, 2), np.round(mn_test_std, 2),
                #       np.round(mn_length, 2), np.round(mn_success, 2))
                #
                # # save the data for plotting the tests
                tests_all.append(mn_test)
                tests_std_all.append(mn_test_std)
                length_all.append(mn_length)
                # perform test end! -----------------------------
                env_test.env.set_usage('default')

                print('Simulated test:', end=' ** ')

                sim_rewards = []
                for i in range(num_ensemble_models):
                    sim_m_env = NetworkEnv(make_env(), model_op, None, number_models=i, verification=True)
                    mn_sim_rew, _, _, _ = test_agent(sim_m_env, agent.predict, num_games=10)
                    sim_rewards.append(mn_sim_rew)
                    print(mn_sim_rew, end=' ** ')

                print("")

                step_counts_all.append(step_count)

                sim_rewards = np.array(sim_rewards)
                sim_rewards_all.append(np.mean(sim_rewards))
                sim_rewards_std_all.append(np.std(sim_rewards))

                batch_rews_all.append(np.mean(batch_rew))
                batch_rews_std_all.append(np.std(batch_rew))

                data = dict(sim_rewards_all=[sim_rewards_all, sim_rewards_std_all],
                            entropy_all=length_all,
                            step_counts_all=step_counts_all,
                            batch_rews_all=[batch_rews_all, batch_rews_std_all],
                            tests_all=[tests_all, tests_std_all],
                            info=label)

                # save the data for plotting the progress -------------------
                save_data(data=data)

                # plotting the progress -------------------
                # if it % 10 == 0:
                plot_observables(data=data, label=label)

                # stop training if the policy hasn't improved
                if (np.sum(best_sim_test >= sim_rewards) > int(num_ensemble_models * 0.7)):
                    if it > delay_before_convergence_check and ep < num_epochs - 1:
                        print('break')
                        break
                else:
                    best_sim_test = sim_rewards

    # Final verification:
    env_test.env.set_usage('final')
    mn_test, mn_test_std, mn_length, _ = test_agent(env_test, agent.predict, num_games=50)

    label = f'Verification : total {total_iterations}, ' + \
            f'data points: {len(model_buffer.train_idx) + len(model_buffer.valid_idx)}, ' + \
            f'ep: {ep}, it: {it}\n' + \
            f'rew: {mn_test}, std: {mn_test_std}'
    plot_results(env_test, label=label)

    env_test.save_current_buffer(info=label)

    env_test.env.set_usage('default')

    # closing environments..
    env.close()
    file_writer.close()


if __name__ == '__main__':
    aedyna('', hidden_sizes=hidden_sizes, num_epochs=num_epochs,
           steps_per_env=steps_per_env, algorithm='TRPO', model_batch_size=model_batch_size,
           simulated_steps=simulated_steps,
           num_ensemble_models=num_ensemble_models, model_iter=model_iter, init_random_steps=init_random_steps)
