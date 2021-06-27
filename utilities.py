import pickle
import random
import time
from datetime import datetime
import matplotlib.pyplot as plt

import gym
from keras import Sequential
from keras.layers import Dense, Lambda
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

import inverted_pendulum


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

    # def generate_random_dataset(self, ratio=0.8):
    #     """ratio: how much for valid taken"""
    #     rnd = np.arange(len(self.obs))
    #     np.random.shuffle(rnd)
    #
    #     valid_idx = rnd[: int(len(self.obs) * ratio)]
    #     train_idx = rnd[int(len(self.obs) * ratio):]
    #
    #     # print('Train set:', len(self.train_idx), 'Valid set:', len(self.valid_idx))
    #
    #     train_set_x = {'states': np.array(fb.obs)[train_idx], "actions": np.array(fb.act)[train_idx]}
    #     train_set_y = {"next_states": np.array(fb.nxt_obs)[train_idx], "rewards": np.array(fb.rew)[train_idx]}
    #
    #     valid_set_x = {'states': np.array(fb.obs)[valid_idx], "actions": np.array(fb.act)[valid_idx]}
    #     valid_set_y = {"next_states": np.array(fb.nxt_obs)[valid_idx], "rewards": np.array(fb.rew)[valid_idx]}
    #
    #     return train_set_x, train_set_y, valid_set_x, valid_set_y

    def generate_random_dataset(self):
        """ratio: how much for valid taken"""
        rnd = np.arange(len(self.obs))
        np.random.shuffle(rnd)

        train_set_x = {'states': np.array(self.obs)[rnd], "actions": np.array(self.act)[rnd]}
        train_set_y = {"next_states": np.array(self.nxt_obs)[rnd], "rewards": np.array(self.rew)[rnd]}

        return train_set_x, train_set_y, len(self)

    def get_maximum(self):
        idx = np.argmax(self.rew)
        print('rew', np.array(self.rew)[idx])
        return np.array(self.obs)[idx], np.array(np.expand_dims(self.act, -1))[idx], \
               np.array(self.rew)[idx], np.array(self.nxt_obs)[idx], np.array(self.done)[
                   idx]

    def __len__(self):
        assert (len(self.rew) == len(self.obs) == len(self.act) == len(self.nxt_obs) == len(self.done))
        return len(self.obs)


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
    Gym Wrapper to store information for scaling to correct space and for post analysis.
    '''

    def __init__(self, env, project_directory, **kwargs):
        gym.Wrapper.__init__(self, env)
        self.data_dict = dict()
        self.environment_usage = 'default'
        self.directory = project_directory
        self.data_dict[self.environment_usage] = TrajectoryBuffer(name=self.environment_usage,
                                                                  directory=self.directory)
        self.current_buffer = self.data_dict.get(self.environment_usage)

        self.test_env_flag = False

        if 'test_env' in kwargs:
            self.test_env_flag = True
        self.verification = False
        if 'verification' in kwargs:
            self.verification = kwargs.get('verification')
        self.max_steps = kwargs.get('max_steps')
        self.current_step = 0

    def reset(self, **kwargs):
        self.current_step = 0
        init_obs = self.env.reset(**kwargs)
        self.current_buffer.new_trajectory(init_obs)
        return init_obs

    def step(self, action):
        ob, reward, done, info = self.env.step(np.squeeze(action))
        self.current_buffer.store_step(obs=ob, act=action, rew=reward, done=done)
        self.current_step += 1
        if self.current_step >= self.max_steps:
            done = True
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

    def save_current_buffer(self, info=''):
        self.current_buffer = self.data_dict.get(self.environment_usage)
        self.current_buffer.save_buffer(info=info)
        print('Saved current buffer', self.environment_usage)

    def set_directory(self, directory):
        self.directory = directory


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
            a_s = agent_op(o.astype(np.float32))
            a_s = np.squeeze(a_s)
            o, r, d, _ = env_test.step(a_s)
            game_r += r
            game_length += 1
        success = r > -0.05
        games_r.append(game_r)
        games_length.append(success)
        games_dones.append(d)
    return np.mean(games_r), np.std(games_r), np.mean(games_length), np.mean(games_dones)

class NetworkEnv(gym.Wrapper):
    '''
    Wrapper to handle the network interaction
    Here you can change the treatment of the uncertainty
    '''

    def __init__(self, env, model_func=None, done_func=None, number_models=1, **kwargs):
        gym.Wrapper.__init__(self, env)
        self.number_models = number_models
        self.current_model = np.random.randint(0, max(self.number_models, 1))
        self.model_func = model_func
        self.done_func = done_func
        self.len_episode = 0
        self.max_steps = env.max_steps
        self.verification = False
        if 'verification' in kwargs:
            self.verification = kwargs.get('verification')
        # self.visualize()
        self.project_directory = kwargs.get('project_directory')

    def reset(self, **kwargs):
        self.current_model = np.random.randint(0, max(self.number_models, 1))
        self.len_episode = 0
        self.done = False
        # Here is a main difference to other dyna style methods:
        # obs = np.random.uniform(-1, 1, self.env.observation_space.shape)
        self.obs = self.env.reset()
        return self.obs

    def step(self, action):
        if self.verification:
            self.obs, rew = self.model_func(self.obs, [np.squeeze(action)])
        else:
            # Can be activated to randomize each step
            current_model = np.random.randint(0, max(self.number_models, 1))  # self.current_model
            self.obs, rew = self.model_func(self.obs, [np.squeeze(action)], current_model)
        # obs, rew, _, _ = self.env.step(action)
        self.len_episode += 1
        if self.len_episode >= self.max_steps:
            self.done = True
        return self.obs, rew, self.done, dict()

    def save_buffer(self, model_nr, data, X, Y, rews, **kwargs):
        if 'info' in kwargs:
            self.info = kwargs.get('info')
        now = datetime.now()
        clock_time = f'{now.month:0>2}_{now.day:0>2}_{now.hour:0>2}_{now.minute:0>2}_{now.second:0>2}_'
        data = dict(data=data,
                    model=model_nr,
                    rews=rews,
                    X=X,
                    Y=Y)
        out_put_writer = open(self.project_directory + clock_time + 'plot_model_' + str(model_nr), 'wb')
        pickle.dump(data, out_put_writer, -1)
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


class ReplayBuffer:
    """
    a ring buffer for storing transitions and sampling for training
    :state: (state_dim,)
    :action: (action_dim,)
    :reward: (,), scalar
    :next_state: (state_dim,)
    :done: (,), scalar (0 and 1) or bool (True and False)
    """

    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = int((self.position + 1) % self.capacity)  # as a ring buffer

    def sample(self, BATCH_SIZE):
        batch = random.sample(self.buffer, BATCH_SIZE)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))  # stack for each element
        """ 
        the * serves as unpack: sum(a,b) <=> batch=(a,b), sum(*batch) ;
        zip: a=[1,2], b=[2,3], zip(a,b) => [(1, 2), (2, 3)] ;
        the map serves as mapping the function on each list element: map(square, [2,3]) => [4,9] ;
        np.stack((1,2)) => array([1, 2])
        """
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)


class DynamicsModel:

    def __init__(self, state_dim, action_dim, reg='free', activation_in='relu', data_noise=0, n_data=1):
        # NN options
        self.activation_in = activation_in
        # self.n_ensemble = n_ensemble  # no. NNs in ensemble
        self.reg = reg  # type of regularisation to use - anc (anchoring) reg (regularised) free (unconstrained)
        self.n_hidden = 100  # no. hidden units in NN
        self.data_noise = data_noise  # estimated noise variance
        self.n_data = n_data

        # optimisation options
        self.epochs = 2000  # run reg for 15+ epochs seems to mess them up
        self.l_rate = 1e-4  # learning rate

        # variance of priors
        self.W1_var = 0.1  # 1st layer weights and biases
        self.W_mid_var = 1 / self.n_hidden  # 2nd layer weights and biases
        self.W_last_var = 1 / self.n_hidden  # 3rd layer weights
        self.state_dim = state_dim
        self.action_dim = action_dim

        # self.input_dim = self.state_dim + self.action_dim

        self.main_model = self.make_model()
        self.initial_weights = self.main_model.get_weights()
        self.reg = reg

        self.callbacks = [
            keras.callbacks.EarlyStopping(
                # Stop training when `val_loss` is no longer improving
                monitor="val_loss",
                # "no longer improving" being defined as "no better than 1e-2 less"
                min_delta=1e-3,
                # "no longer improving" being further defined as "for at least 2 epochs"
                patience=10,
                verbose=1, )
        ]

        self.n = 0

    def restore(self):
        self.main_model.set_weights(self.initial_weights)

    def initialize_network(self):
        # get initialisations, and regularisation values
        self.W1_lambda = self.data_noise / self.W1_var
        self.W1_anc = np.random.normal(loc=0, scale=np.sqrt(self.W1_var),
                                       size=[self.state_dim + self.action_dim, self.n_hidden])
        self.W1_init = np.random.normal(loc=0, scale=np.sqrt(self.W1_var),
                                        size=[self.state_dim + self.action_dim, self.n_hidden])

        self.b1_var = self.W1_var
        self.b1_lambda = self.data_noise / self.b1_var
        self.b1_anc = np.random.normal(loc=0, scale=np.sqrt(self.b1_var), size=[self.n_hidden])
        self.b1_init = np.random.normal(loc=0, scale=np.sqrt(self.b1_var), size=[self.n_hidden])

        self.W_mid_lambda = self.data_noise / self.W_mid_var
        self.W_mid_anc = np.random.normal(loc=0, scale=np.sqrt(self.W_mid_var), size=[self.n_hidden, self.n_hidden])
        self.W_mid_init = np.random.normal(loc=0, scale=np.sqrt(self.W_mid_var), size=[self.n_hidden, self.n_hidden])

        self.b_mid_var = self.W_mid_var
        self.b_mid_lambda = self.data_noise / self.b_mid_var
        self.b_mid_anc = np.random.normal(loc=0, scale=np.sqrt(self.b_mid_var), size=[self.n_hidden])
        self.b_mid_init = np.random.normal(loc=0, scale=np.sqrt(self.b_mid_var), size=[self.n_hidden])

        self.W_last_lambda = self.data_noise / self.W_last_var
        self.W_last_anc = np.random.normal(loc=0, scale=np.sqrt(self.W_last_var),
                                           size=[self.n_hidden, self.state_dim + 1])
        self.W_last_init = np.random.normal(loc=0, scale=np.sqrt(self.W_last_var),
                                            size=[self.n_hidden, self.state_dim + 1])

    # create custom regularised
    def custom_reg_W1(self, weight_matrix):
        if self.reg == 'reg':
            return keras.sum(keras.square(weight_matrix)) * self.W1_lambda / self.n_data
        elif self.reg == 'free':
            return 0.
        elif self.reg == 'anc':
            return keras.sum(keras.square(weight_matrix - self.W1_anc)) * self.W1_lambda / self.n_data

    def custom_reg_b1(self, weight_matrix):
        if self.reg == 'reg':
            return keras.sum(keras.square(weight_matrix)) * self.b1_lambda / self.n_data
        elif self.reg == 'free':
            return 0.
        elif self.reg == 'anc':
            return keras.sum(keras.square(weight_matrix - self.b1_anc)) * self.b1_lambda / self.n_data

    def custom_reg_W_mid(self, weight_matrix):
        if self.reg == 'reg':
            return keras.sum(keras.square(weight_matrix)) * self.W_mid_lambda / self.n_data
        elif self.reg == 'free':
            return 0.
        elif self.reg == 'anc':
            return keras.sum(keras.square(weight_matrix - self.W_mid_anc)) * self.W_mid_lambda / self.n_data

    def custom_reg_b_mid(self, weight_matrix):
        if self.reg == 'reg':
            return keras.sum(keras.square(weight_matrix)) * self.b_mid_lambda / self.n_data
        elif self.reg == 'free':
            return 0.
        elif self.reg == 'anc':
            return keras.sum(keras.square(weight_matrix - self.b_mid_anc)) * self.b_mid_lambda / self.n_data

    def custom_reg_W_last(self, weight_matrix):
        if self.reg == 'reg':
            return keras.sum(keras.square(weight_matrix)) * self.W_last_lambda / self.n_data
        elif self.reg == 'free':
            return 0.
        elif self.reg == 'anc':
            return keras.sum(keras.square(weight_matrix - self.W_last_anc)) * self.W_last_lambda / self.n_data

    def make_model(self):
        self.initialize_network()

        inputs_states = keras.Input(shape=(self.state_dim,), name='states')
        inputs_actions = keras.Input(shape=(self.action_dim,), name='actions')
        state_input = layers.concatenate([inputs_states, inputs_actions])

        # x = layers.Dense(self.n_hidden, activation=self.activation_in, name='hidden1')(state_input)
        # x = layers.Dense(self.n_hidden, activation=self.activation_in, name='hidden2')(x)
        # out = layers.Dense(self.state_dim + 1, activation='linear', name='out')(x)

        x = layers.Dense(self.n_hidden, activation=self.activation_in,
                         kernel_initializer=keras.initializers.Constant(value=self.W1_init),
                         bias_initializer=keras.initializers.Constant(value=self.b1_init),
                         kernel_regularizer=self.custom_reg_W1,
                         bias_regularizer=self.custom_reg_b1,
                         name='hidden1')(state_input)

        x = layers.Dense(self.n_hidden, activation=self.activation_in,
                         kernel_initializer=keras.initializers.Constant(value=self.W_mid_init),
                         bias_initializer=keras.initializers.Constant(value=self.b_mid_init),
                         kernel_regularizer=self.custom_reg_W_mid,
                         bias_regularizer=self.custom_reg_b_mid,
                         name='hidden2')(x)

        out = layers.Dense(self.state_dim + 1, activation='linear', use_bias=False,
                           kernel_initializer=keras.initializers.Constant(value=self.W_last_init),
                           kernel_regularizer=self.custom_reg_W_last,
                           name='out')(x)

        next_states = layers.Lambda(lambda x: x[:, :-1], name='next_states')(out)
        rewards = layers.Lambda(lambda x: x[:, -1], name='rewards')(out)

        model = keras.Model(
            inputs=[inputs_states, inputs_actions],
            outputs=[next_states, rewards])

        # keras.utils.plot_model(model, "multi_input_and_multi_output_model.png", show_shapes=True)

        model.compile(loss='mean_squared_error',
                      optimizer=keras.optimizers.Adam(learning_rate=self.l_rate))

        return model

    @tf.function
    def predict(self, x_pred):
        # return self.main_model.predict(x_pred)
        return self.main_model(x_pred, training=False)

    def evaluate(self, train_set_x, train_set_y):
        return self.main_model.evaluate(train_set_x, train_set_y, verbose=0, return_dict=True)['loss']

    def train_model(self, train_set_x, train_set_y, n_data, batch_size=512):
        self.n_data = n_data
        history = self.main_model.fit(train_set_x, train_set_y, validation_split=0.2, epochs=self.epochs, shuffle=True,
                                      callbacks=self.callbacks, batch_size=batch_size, verbose=0)
        return history


def plot_observables(data, label, length_all, **kwargs):
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

    ax2 = ax.twinx()

    color = 'lime'
    ax2.set_ylabel('data points', color=color)  # we already handled the x-label with ax1
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.step(x, step_counts_all, color=color)

    ax = axs[1]
    ax.plot(sim_rewards_all[0], ls=':', label='sim')
    ax.fill_between(x, sim_rewards_all[0] - sim_rewards_all[1], sim_rewards_all[0] + sim_rewards_all[1],
                    alpha=0.5)
    try:
        ax.plot(tests_all[0], label='real')
        ax.fill_between(x, tests_all[0] - tests_all[1], tests_all[0] + tests_all[1],
                        alpha=0.5)
        ax.axhline(y=np.max(tests_all[0]), c='orange')
    except:
        pass
    ax.set_ylabel('rewards tests')
    ax.legend(loc="lower left")
    # plt.tw
    ax.grid(True)
    ax2 = ax.twinx()

    color = 'lime'
    ax2.set_ylabel('success', color=color)  # we already handled the x-label with ax1
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.plot(length_all, color=color)

    fig.align_labels()
    plt.show()

def plot_results(env_wrapper, label=None, **kwargs):
    """ Plot the validation episodes"""
    rewards = env_wrapper.env.current_buffer.get_data()['rews']

    iterations = []
    finals = []
    means = []
    stds = []

    for i in range(len(rewards)):
        if len(rewards[i]) > 1:
            finals.append(rewards[i][-1])
            # means.append(np.mean(rewards[i][1:]))
            means.append(np.sum(rewards[i][1:]))
            stds.append(np.std(rewards[i][1:]))
            iterations.append(len(rewards[i]))
    x = range(len(iterations))
    iterations = np.array(iterations)
    finals = np.array(finals)
    means = np.array(means)
    stds = np.array(stds)

    plot_suffix = label

    # fig, axs = plt.subplots(2, 1, sharex=True)
    fig, ax = plt.subplots(1, 1)

    # ax = axs[0]
    # ax.plot(x, iterations)
    # ax.set_ylabel('Iterations (1)')
    # ax.set_title(plot_suffix)

    # if 'data_number' in kwargs:
    #     ax1 = plt.twinx(ax)
    #     color = 'lime'
    #     ax1.set_ylabel('Mean reward', color=color)  # we already handled the x-label with ax1
    #     ax1.tick_params(axis='y', labelcolor=color)
    #     ax1.plot(x, kwargs.get('data_number'), color=color)

    # ax = axs[1]
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

    if 'save_name' in kwargs:
        plt.savefig(kwargs.get('save_name') + '.pdf')
    plt.show()


def train_agent(env, agent, replay_buffer, TRAIN_EPISODES=10, MAX_STEPS=200, EXPLORE_STEPS=500, BATCH_SIZE=256,
                UPDATE_ITR=3, REWARD_SCALE=1., AUTO_ENTROPY=True, RENDER=False, frame_idx_in=0):
    t0 = time.time()
    frame_idx = frame_idx_in
    all_episode_reward = []
    act_dim = env.action_space.shape[0]

    # need an extra call here to make inside functions be able to use model.forward
    state = env.reset().astype(np.float32)
    agent.policy_net([state])

    for episode in range(TRAIN_EPISODES):
        state = env.reset().astype(np.float32)
        episode_reward = 0
        for step in range(MAX_STEPS):
            if RENDER:
                env.render()
            if frame_idx > EXPLORE_STEPS:
                action = agent.policy_net.get_action(state)
            else:
                action = agent.policy_net.sample_action()

            next_state, reward, done, _ = env.step(action)
            next_state = next_state.astype(np.float32)
            done = 1 if done is True else 0

            replay_buffer.push(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward
            frame_idx += 1

            if len(replay_buffer) > BATCH_SIZE:
                for i in range(UPDATE_ITR):
                    agent.update(
                        BATCH_SIZE, reward_scale=REWARD_SCALE, auto_entropy=AUTO_ENTROPY,
                        target_entropy=-1. * act_dim
                    )

            if done:
                break
        if episode == 0:
            all_episode_reward.append(episode_reward)
        else:
            all_episode_reward.append(all_episode_reward[-1] * 0.9 + episode_reward * 0.1)
        print(
            'Training  | Episode: {}/{}  | Episode Reward: {:.4f}  | Running Time: {:.4f}'.format(
                episode + 1, TRAIN_EPISODES, episode_reward,
                time.time() - t0
            )
        )

    return frame_idx


if __name__ == '__main__':
    REPLAY_BUFFER_SIZE = 1000000
    # initialization of buffer
    replay_buffer = ReplayBuffer(REPLAY_BUFFER_SIZE)

    env = inverted_pendulum.PendulumEnv()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    frame_idx = 0
    all_episode_reward = []

    # need an extra call here to make inside functions be able to use model.forward
    state = env.reset().astype(np.float32)
    TRAIN_EPISODES = 1
    MAX_STEPS = 200
    RENDER = False
    EXPLORE_STEPS = 1000

    fb = FullBuffer()

    for episode in range(TRAIN_EPISODES):
        state = env.reset().astype(np.float32)
        episode_reward = 0
        for step in range(MAX_STEPS):
            if RENDER:
                env.render()

            action = env.action_space.sample()

            next_state, reward, done, _ = env.step(action)
            next_state = next_state.astype(np.float32)
            done = 1 if done is True else 0

            # replay_buffer.push(state, action, reward, next_state, done)
            fb.store(state, action, reward, next_state, done)

            # print(state, action, reward, next_state, done)

            state = next_state
            episode_reward += reward
            frame_idx += 1

    train_set_x, train_set_y, n_data = fb.generate_random_dataset()
    model = DynamicsModel(state_dim, action_dim)

    train_dict = {'states': np.array([[0.37486365, -0.92708, 1.416558]]),
                  'actions': np.array([7.03400493e-01])}
    print(model.evaluate(train_set_x, train_set_y))

    model.train_model(train_set_x, train_set_y, n_data=n_data)
    print(model.evaluate(train_set_x, train_set_y))

    model.restore()
    print(model.evaluate(train_set_x, train_set_y))
