import os
import pickle

import numpy as np
import tensorflow as tf
import gym
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt

from laser_trajectory_control_env import LaserTrajectoryControlEnv
from tango_connection import TangoConnection

# set random seed
random_seed = 111
np.random.seed(random_seed)

conf_file = '/home/niky/FERMI/2020_10_06/configuration/conf_fel2.json'
tango = TangoConnection(conf_file=conf_file)
real_env = LaserTrajectoryControlEnv(tango=tango)

# Hyper papameters
steps_per_env = 20
init_random_steps = 100
num_epochs = int((500-init_random_steps)/(steps_per_env)) # increase 500 to 700
# num_epochs = int((125-init_random_steps)/(steps_per_env))
print('num of epoches', num_epochs)


hidden_sizes = [100, 100]
cr_lr = 1e-3
gamma = 0.9999
lam = 0.95

max_training_iterations = 100
# max_training_iterations = 10
delay_before_convergence_check = 2

number_envs = 1
critic_iter = 80
delta = 0.05
algorithm = 'TRPO'
conj_iters = 10
minibatch_size = 500
simulated_steps = 2000

#mb_lr = 5e-4
mb_lr = 1e-3
model_batch_size = 100
num_ensemble_models = 5
model_iter = 15

# How often to check the progress of the network training
# e.g. lambda it, episode: (it + 1) % max(3, (ep+1)*2) == 0
# dynamic_wait_time = lambda it, ep: (it + 1) % 5 == 0  #
dynamic_wait_time = lambda it, ep: (it + 1) % 3 == 0  #
# dynamic_wait_time = lambda it, ep: (it + 1) % 1 == 0  #

# Create the logging directory:
project_directory = 'Data_logging/ME_TRPO/mt_1/'


hyp_str_all = '-nr_steps_' + str(steps_per_env) + '-cr_lr' + str(cr_lr) + '-crit_it_' + str(
    critic_iter) + '-d_' + str(delta) + '-conj_iters_' + str(conj_iters) + '-n_ep_' + str(num_epochs) + \
              '-mini_bs_' + str(minibatch_size) + '-m_bs_' + str(model_batch_size) + \
              '-mb_lr_' + str(mb_lr) + \
              '-sim_steps_' + str(simulated_steps) + \
              '-m_iter_' + str(model_iter) + '-ensnr_' + str(num_ensemble_models) + '-init_' + str(
    init_random_steps) + '/'
project_directory = project_directory + hyp_str_all

# To label the plots:
hyp_str_all = '-nr_steps_' + str(steps_per_env) + '-cr_lr' + str(cr_lr) + '-crit_it_' + str(
    critic_iter) + '-d_' + str(delta) + '-conj_iters_' + str(conj_iters) + '-n_ep_' + str(num_epochs) + \
              '\n-mini_bs_' + str(minibatch_size) + '-m_bs_' + str(model_batch_size) + \
              '-mb_lr_' + str(mb_lr) + \
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

    def __init__(self, env,  **kwargs):
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


    def reset(self, **kwargs):
        init_obs = self.env.reset(**kwargs)
        self.current_buffer.new_trajectory(init_obs)
        init_obs = self.scale_state_env(init_obs)
        # print('Menv: ', init_obs)
        return init_obs

    def step(self, action):
        # print('a', action)
        action = self.descale_action_env(action)
        # print('as', action)
        ob, reward, done, info = self.env.step(action)
        self.current_buffer.store_step(obs=ob, act=action, rew=reward, done=done)
        ob = self.scale_state_env(ob)
        reward = self.rew_scale(reward)
        # print('Menv: ', ob, reward, done, info)
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

    def descale_action_env(self, act):
        scale = (self.env.action_space.high - self.env.action_space.low)
        return (scale * act + self.env.action_space.high + self.env.action_space.low) / 2

    def rew_scale(self, rew):

        if not(self.test_env_flag):
            '''Rescale reward from [-1,0] to [-1,1] for the training of the network'''
            rew = rew * 2 + 1
        return rew

    def save_current_buffer(self, info=''):
        self.current_buffer = self.data_dict.get(self.environment_usage)
        self.current_buffer.save_buffer(info=info)
        print('Saved current buffer', self.environment_usage)

    def set_directory(self, directory):
        self.directory = directory


env_monitored = MonitoringEnv(env=real_env)


def make_env(**kwargs):
    '''Create the environement'''
    return MonitoringEnv(env=real_env, **kwargs)


def mlp(x, hidden_layers, output_layer, activation=tf.tanh, last_activation=None):
    '''
    Multi-layer perceptron
    '''
    for l in hidden_layers:
        x = tf.layers.dense(x, units=l, activation=activation)
    return tf.layers.dense(x, units=output_layer, activation=last_activation)


def softmax_entropy(logits):
    '''
    Softmax Entropy
    '''
    return -tf.reduce_sum(tf.nn.softmax(logits, axis=-1) * tf.nn.log_softmax(logits, axis=-1), axis=-1)


def gaussian_log_likelihood(ac, mean, log_std):
    '''
    Gaussian Log Likelihood
    '''
    log_p = ((ac - mean) ** 2 / (tf.exp(log_std) ** 2 + 1e-9) + 2 * log_std) + np.log(2 * np.pi)
    return -0.5 * tf.reduce_sum(log_p, axis=-1)


def conjugate_gradient(A, b, x=None, iters=10):
    '''
    Conjugate gradient method: approximate the solution of Ax=b
    It solve Ax=b without forming the full matrix, just compute the matrix-vector product (The Fisher-vector product)
    NB: A is not the full matrix but is a useful matrix-vector product between the averaged Fisher information matrix and arbitrary vectors
    Descibed in Appendix C.1 of the TRPO paper
    '''
    if x is None:
        x = np.zeros_like(b)

    r = A(x) - b
    p = -r
    for _ in range(iters):
        a = np.dot(r, r) / (np.dot(p, A(p)) + 1e-8)
        x += a * p
        r_n = r + a * A(p)
        b = np.dot(r_n, r_n) / (np.dot(r, r) + 1e-8)
        p = -r_n + b * p
        r = r_n
    return x


def gaussian_DKL(mu_q, log_std_q, mu_p, log_std_p):
    '''
    Gaussian KL divergence in case of a diagonal covariance matrix
    '''
    return tf.reduce_mean(tf.reduce_sum(
        0.5 * (log_std_p - log_std_q + tf.exp(log_std_q - log_std_p) + (mu_q - mu_p) ** 2 / tf.exp(log_std_p) - 1),
        axis=1))


def backtracking_line_search(Dkl, delta, old_loss, p=0.8):
    '''
    Backtracking line searc. It look for a coefficient s.t. the constraint on the DKL is satisfied
    It has both to
     - improve the non-linear objective
     - satisfy the constraint
    '''
    ## Explained in Appendix C of the TRPO paper
    a = 1
    it = 0

    new_dkl, new_loss = Dkl(a)
    while (new_dkl > delta) or (new_loss > old_loss):
        a *= p
        it += 1
        new_dkl, new_loss = Dkl(a)

    return a


def GAE(rews, v, v_last, gamma=0.99, lam=0.95):
    '''
    Generalized Advantage Estimation
    '''
    assert len(rews) == len(v)
    vs = np.append(v, v_last)
    d = np.array(rews) + gamma * vs[1:] - vs[:-1]
    gae_advantage = discounted_rewards(d, 0, gamma * lam)
    return gae_advantage


def discounted_rewards(rews, last_sv, gamma):
    '''
    Discounted reward to go
    Parameters:
    ----------
    rews: list of rewards
    last_sv: value of the last state
    gamma: discount value
    '''
    rtg = np.zeros_like(rews, dtype=np.float32)
    rtg[-1] = rews[-1] + gamma * last_sv
    for i in reversed(range(len(rews) - 1)):
        rtg[i] = rews[i] + gamma * rtg[i + 1]
    return rtg


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
    for _ in range(num_games):
        d = False
        game_r = 0
        o = env_test.reset()
        game_length = 0
        while not d:
            a_s, _ = agent_op([o])
            o, r, d, _ = env_test.step(a_s[0])
            game_r += r
            game_length += 1

        games_r.append(game_r)
        games_length.append(game_length)
    return np.mean(games_r), np.std(games_r), np.mean(games_length)


class Buffer():
    '''
    Class to store the experience from a unique policy
    '''

    def __init__(self, gamma=0.99, lam=0.95):
        self.gamma = gamma
        self.lam = lam
        self.adv = []
        self.ob = []
        self.ac = []
        self.rtg = []

    def store(self, temp_traj, last_sv):
        '''
        Add temp_traj values to the buffers and compute the advantage and reward to go
        Parameters:
        -----------
        temp_traj: list where each element is a list that contains: observation, reward, action, state-value
        last_sv: value of the last state (Used to Bootstrap)
        '''
        # store only if there are temporary trajectories
        if len(temp_traj) > 0:
            self.ob.extend(temp_traj[:, 0])
            rtg = discounted_rewards(temp_traj[:, 1], last_sv, self.gamma)
            self.adv.extend(GAE(temp_traj[:, 1], temp_traj[:, 3], last_sv, self.gamma, self.lam))
            self.rtg.extend(rtg)
            self.ac.extend(temp_traj[:, 2])

    def get_batch(self):
        # standardize the advantage values
        norm_adv = (self.adv - np.mean(self.adv)) / (np.std(self.adv) + 1e-10)
        return np.array(self.ob), np.array(np.expand_dims(self.ac, -1)), np.array(norm_adv), np.array(self.rtg)

    def __len__(self):
        assert (len(self.adv) == len(self.ob) == len(self.ac) == len(self.rtg))
        return len(self.ob)


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

    def generate_random_dataset(self):
        rnd = np.arange(len(self.obs))
        np.random.shuffle(rnd)
        self.valid_idx = rnd[: int(len(self.obs) / 5)]
        self.train_idx = rnd[int(len(self.obs) / 5):]
        print('Train set:', len(self.train_idx), 'Valid set:', len(self.valid_idx))

    def get_training_batch(self):
        return np.array(self.obs)[self.train_idx], np.array(np.expand_dims(self.act, -1))[self.train_idx], \
               np.array(self.rew)[self.train_idx], np.array(self.nxt_obs)[self.train_idx], np.array(self.done)[
                   self.train_idx]

    def get_valid_batch(self):
        return np.array(self.obs)[self.valid_idx], np.array(np.expand_dims(self.act, -1))[self.valid_idx], \
               np.array(self.rew)[self.valid_idx], np.array(self.nxt_obs)[self.valid_idx], np.array(self.done)[
                   self.valid_idx]

    def __len__(self):
        assert (len(self.rew) == len(self.obs) == len(self.act) == len(self.nxt_obs) == len(self.done))
        return len(self.obs)


def simulate_environment(env, policy, simulated_steps):
    '''Lists to store rewards and length of the trajectories completed'''
    buffer = Buffer(0.99, 0.95)
    steps = 0
    number_episodes = 0

    while steps < simulated_steps:
        temp_buf = []
        obs = env.reset()
        number_episodes += 1
        done = False

        while not done:
            act, val = policy([obs])

            obs2, rew, done, _ = env.step([act])

            temp_buf.append([obs.copy(), rew, np.squeeze(act), np.squeeze(val)])

            obs = obs2.copy()
            steps += 1

            if done:
                buffer.store(np.array(temp_buf), 0)
                temp_buf = []

            if steps == simulated_steps:
                break

        buffer.store(np.array(temp_buf), np.squeeze(policy([obs])[1]))

    print('Sim ep:', number_episodes, end=' \n')

    return buffer.get_batch(), number_episodes


class NetworkEnv(gym.Wrapper):
    '''
    Wrapper to handle the network interaction
    '''

    def __init__(self, env, model_func=None, done_func=None, number_models=1):
        gym.Wrapper.__init__(self, env)

        self.model_func = model_func
        self.done_func = done_func
        self.number_models = number_models
        self.len_episode = 0
        self.threshold = 0
        self.max_steps = env.max_steps

    def reset(self, **kwargs):
        self.threshold = -0.05 * 2 + 1  # rescaled [-1,1]
        self.len_episode = 0
        # kwargs['simulation'] = True
        # action = self.env.reset(**kwargs)
        if self.model_func is not None:
            obs = np.random.uniform(-1, 1, self.env.observation_space.shape)
            # print('reset', obs)
            # Todo: remove
            # obs = self.env.reset()
        else:
            # obs = self.env.reset(**kwargs)
            pass
        # Does this work?
        self.obs = np.clip(obs, -1.0, 1.0)
        self.obs_init = self.obs.copy()
        # if self.test_phase:
        #     print('test reset', self.obs)
        # print('init: ',self.obs)
        return self.obs

    def step(self, action):
        if self.model_func is not None:
            # predict the next state on a random model
            obs, rew = self.model_func(self.obs, [np.squeeze(action)], np.random.randint(0, self.number_models))
            # # Todo: remove
            # self.env.state = self.obs
            # done = rew > self.threshold

            self.len_episode += 1
            done = self.len_episode >= self.max_steps
            self.obs = np.clip(obs, -1.0, 1.0)
            # # Todo: remove
            # obs_true, rew_true, done_true, _ = self.env.step(action)
            # print(np.linalg.norm(self.obs-obs_true),np.linalg.norm(rew-rew_true), done, done_true)
            # print(self.len_episode, 'state ', obs, 'a ', np.squeeze(action), 'r ', rew)
            # print('step:', self.obs, rew, done, "")
            rew = (rew - 1) / 2
            if rew > -0.05:
                done = True
            return self.obs, rew, done, ""
        else:
            # self.obs, rew, done, _ = real_env.step(action)
            # return self.obs, rew, done, ""
            pass
        # return env.step(action)

    def set_test(self, test_flag):
        real_env.test = test_flag


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


# def episode_done(rew):
#     # TODO: new
#     threshold = -0.1
#     # if rew> threshold:
#     #     print('done---'*10, rew)
#     # return np.abs(np.arcsin(np.squeeze(ob[3]))) > .2
#     return rew > threshold


# def final_reward(ob, ac):
#     # TODO: new
#     # return -np.sqrt(np.mean(np.square(ob)))
#     pass


def restore_model(old_model_variables, m_variables):
    # variable used as index for restoring the actor's parameters
    it_v2 = tf.Variable(0, trainable=False)
    restore_m_params = []

    for m_v in m_variables:
        upd_m_rsh = tf.reshape(old_model_variables[it_v2: it_v2 + tf.reduce_prod(m_v.shape)], shape=m_v.shape)
        restore_m_params.append(m_v.assign(upd_m_rsh))
        it_v2 += tf.reduce_prod(m_v.shape)

    return tf.group(*restore_m_params)


def METRPO(env_name, hidden_sizes=[32, 32], cr_lr=5e-3, num_epochs=50, gamma=0.99, lam=0.95, number_envs=1,
           critic_iter=10, steps_per_env=100, delta=0.05, algorithm='TRPO', conj_iters=10, minibatch_size=1000,
           mb_lr_start=0.0001, model_batch_size=512, simulated_steps=1000, num_ensemble_models=2, model_iter=15,
           init_random_steps=steps_per_env):
    '''
    Model Ensemble Trust Region Policy Optimization
    Parameters:
    -----------
    env_name: Name of the environment
    hidden_sizes: list of the number of hidden units for each layer
    cr_lr: critic learning rate
    num_epochs: number of training epochs
    gamma: discount factor
    lam: lambda parameter for computing the GAE
    number_envs: number of "parallel" synchronous environments
        # NB: it isn't distributed across multiple CPUs
    critic_iter: Number of SGD iterations on the critic per epoch
    steps_per_env: number of steps per environment
            # NB: the total number of steps per epoch will be: steps_per_env*number_envs
    delta: Maximum KL divergence between two policies. Scalar value
    algorithm: type of algorithm. Either 'TRPO' or 'NPO'
    conj_iters: number of conjugate gradient iterations
    minibatch_size: Batch size used to train the critic
    mb_lr: learning rate of the environment model
    model_batch_size: batch size of the environment model
    simulated_steps: number of simulated steps for each policy update
    num_ensemble_models: number of models
    model_iter: number of iterations without improvement before stopping training the model
    '''
    # TODO: add ME-TRPO hyperparameters

    tf.reset_default_graph()

    # Create a few environments to collect the trajectories

    # envs = [StructEnv(gym.make(env_name)) for _ in range(number_envs)]
    envs = [StructEnv(make_env()) for _ in range(number_envs)]
    env_test = StructEnv(make_env(test_env = True))
    # env_test = gym.make(env_name)
    print('env_test' * 4)

    # env_test = make_env(test=True)
    # env_test = gym.wrappers.Monitor(env_test, "VIDEOS/", force=True, video_callable=lambda x: x%10 == 0)
    # to be changed in real test
    # env_test = FelLocalEnv(tango=tango)
    # env_test.test = True
    # env_test_1 = FelLocalEnv(tango=tango)
    # env_test_1.test = True

    # If the scaling is not perfomed this has to be changed
    low_action_space = -1  # envs[0].action_space.low
    high_action_space = 1  # envs[0].action_space.high

    obs_dim = envs[0].observation_space.shape

    act_dim = envs[0].action_space.shape[0]

    # print(envs[0].action_space, envs[0].observation_space, low_action_space,
    #       high_action_space)

    # Placeholders
    act_ph = tf.placeholder(shape=(None, act_dim), dtype=tf.float32, name='act')
    obs_ph = tf.placeholder(shape=(None, obs_dim[0]), dtype=tf.float32, name='obs')
    # NEW
    nobs_ph = tf.placeholder(shape=(None, obs_dim[0]), dtype=tf.float32, name='nobs')
    rew_ph = tf.placeholder(shape=(None, 1), dtype=tf.float32, name='rew')

    ret_ph = tf.placeholder(shape=(None,), dtype=tf.float32, name='ret')
    adv_ph = tf.placeholder(shape=(None,), dtype=tf.float32, name='adv')
    old_p_log_ph = tf.placeholder(shape=(None,), dtype=tf.float32, name='old_p_log')
    old_mu_ph = tf.placeholder(shape=(None, act_dim), dtype=tf.float32, name='old_mu')
    old_log_std_ph = tf.placeholder(shape=(act_dim), dtype=tf.float32, name='old_log_std')
    p_ph = tf.placeholder(shape=(None,), dtype=tf.float32, name='p_ph')

    mb_lr_ = tf.placeholder("float", None)#, name='mb_lr')


    # result of the conjugate gradient algorithm
    cg_ph = tf.placeholder(shape=(None,), dtype=tf.float32, name='cg')

    #########################################################
    ######################## POLICY #########################
    #########################################################

    old_model_variables = tf.placeholder(shape=(None,), dtype=tf.float32, name='old_model_variables')

    # Neural network that represent the policy
    with tf.variable_scope('actor_nn'):
        p_means = mlp(obs_ph, hidden_sizes, act_dim, tf.tanh, last_activation=tf.tanh)
        p_means = tf.clip_by_value(p_means, low_action_space, high_action_space)
        log_std = tf.get_variable(name='log_std', initializer=np.ones(act_dim, dtype=np.float32))

    # Neural network that represent the value function
    with tf.variable_scope('critic_nn'):
        s_values = mlp(obs_ph, hidden_sizes, 1, tf.tanh, last_activation=None)
        s_values = tf.squeeze(s_values)

        # Add "noise" to the predicted mean following the Gaussian distribution with standard deviation e^(log_std)
    p_noisy = p_means + tf.random_normal(tf.shape(p_means), 0, 1) * tf.exp(log_std)
    # Clip the noisy actions
    a_sampl = tf.clip_by_value(p_noisy, low_action_space, high_action_space)

    # Compute the gaussian log likelihood
    p_log = gaussian_log_likelihood(act_ph, p_means, log_std)

    # Measure the divergence
    diverg = tf.reduce_mean(tf.exp(old_p_log_ph - p_log))

    # ratio
    ratio_new_old = tf.exp(p_log - old_p_log_ph)
    # TRPO surrogate loss function
    p_loss = - tf.reduce_mean(ratio_new_old * adv_ph)

    # MSE loss function
    v_loss = tf.reduce_mean((ret_ph - s_values) ** 2)
    # Critic optimization
    v_opt = tf.train.AdamOptimizer(cr_lr).minimize(v_loss)

    def variables_in_scope(scope):
        # get all trainable variables in 'scope'
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)

    # Gather and flatten the actor parameters
    p_variables = variables_in_scope('actor_nn')
    p_var_flatten = flatten_list(p_variables)

    # Gradient of the policy loss with respect to the actor parameters
    p_grads = tf.gradients(p_loss, p_variables)
    p_grads_flatten = flatten_list(p_grads)

    ########### RESTORE ACTOR PARAMETERS ###########
    p_old_variables = tf.placeholder(shape=(None,), dtype=tf.float32, name='p_old_variables')
    # variable used as index for restoring the actor's parameters
    it_v1 = tf.Variable(0, trainable=False)
    restore_params = []

    for p_v in p_variables:
        upd_rsh = tf.reshape(p_old_variables[it_v1: it_v1 + tf.reduce_prod(p_v.shape)], shape=p_v.shape)
        restore_params.append(p_v.assign(upd_rsh))
        it_v1 += tf.reduce_prod(p_v.shape)

    restore_params = tf.group(*restore_params)

    # gaussian KL divergence of the two policies
    dkl_diverg = gaussian_DKL(old_mu_ph, old_log_std_ph, p_means, log_std)

    # Jacobian of the KL divergence (Needed for the Fisher matrix-vector product)
    dkl_diverg_grad = tf.gradients(dkl_diverg, p_variables)

    dkl_matrix_product = tf.reduce_sum(flatten_list(dkl_diverg_grad) * p_ph)
    print('dkl_matrix_product', dkl_matrix_product.shape)
    # Fisher vector product
    # The Fisher-vector product is a way to compute the A matrix without the need of the full A
    Fx = flatten_list(tf.gradients(dkl_matrix_product, p_variables))

    ## Step length
    beta_ph = tf.placeholder(shape=(), dtype=tf.float32, name='beta')
    # NPG update
    npg_update = beta_ph * cg_ph

    ## alpha is found through line search
    alpha = tf.Variable(1., trainable=False)
    # TRPO update
    trpo_update = alpha * npg_update

    ####################   POLICY UPDATE  ###################
    # variable used as an index
    it_v = tf.Variable(0, trainable=False)
    p_opt = []
    # Apply the updates to the policy
    for p_v in p_variables:
        print(p_v)
        upd_rsh = tf.reshape(trpo_update[it_v: it_v + tf.reduce_prod(p_v.shape)], shape=p_v.shape)
        p_opt.append(p_v.assign_sub(upd_rsh))
        it_v += tf.reduce_prod(p_v.shape)

    p_opt = tf.group(*p_opt)

    #########################################################
    ######################### MODEL #########################
    #########################################################

    m_opts = []
    m_losses = []

    nobs_pred_m = []
    act_obs = tf.concat([obs_ph, act_ph], 1)
    # TODO: Variable learning rate injected


    # computational graph of N models
    for i in range(num_ensemble_models):
        with tf.variable_scope('model_' + str(i) + '_nn'):
            # TODO: Add variable size of network
            hidden_sizes = 100
            nobs_pred = mlp(act_obs, [100, 100], obs_dim[0] + 1, tf.nn.tanh, last_activation=None)
            nobs_pred_m.append(nobs_pred)

        m_loss = tf.reduce_mean((tf.concat([nobs_ph, rew_ph], 1) - nobs_pred) ** 2)
        m_losses.append(m_loss)

        m_opts.append(tf.train.AdamOptimizer(learning_rate=mb_lr_).minimize(m_loss))

    ##################### RESTORE MODEL ######################
    initialize_models = []
    models_variables = []
    for i in range(num_ensemble_models):
        m_variables = variables_in_scope('model_' + str(i) + '_nn')
        initialize_models.append(restore_model(old_model_variables, m_variables))
        models_variables.append(flatten_list(m_variables))

    # Time
    now = datetime.now()
    clock_time = "{}_{}_{}_{}".format(now.day, now.hour, now.minute, now.second)
    print('Time:', clock_time)

    # Set scalars and hisograms for TensorBoard
    tf.summary.scalar('p_loss', p_loss, collections=['train'])
    tf.summary.scalar('v_loss', v_loss, collections=['train'])
    tf.summary.scalar('p_divergence', diverg, collections=['train'])
    tf.summary.scalar('ratio_new_old', tf.reduce_mean(ratio_new_old), collections=['train'])
    tf.summary.scalar('dkl_diverg', dkl_diverg, collections=['train'])
    tf.summary.scalar('alpha', alpha, collections=['train'])
    tf.summary.scalar('beta', beta_ph, collections=['train'])
    tf.summary.scalar('p_std_mn', tf.reduce_mean(tf.exp(log_std)), collections=['train'])
    tf.summary.scalar('s_values_mn', tf.reduce_mean(s_values), collections=['train'])
    tf.summary.histogram('p_log', p_log, collections=['train'])
    tf.summary.histogram('p_means', p_means, collections=['train'])
    tf.summary.histogram('s_values', s_values, collections=['train'])
    tf.summary.histogram('adv_ph', adv_ph, collections=['train'])
    tf.summary.histogram('log_std', log_std, collections=['train'])
    scalar_summary = tf.summary.merge_all('train')

    tf.summary.scalar('old_v_loss', v_loss, collections=['pre_train'])
    tf.summary.scalar('old_p_loss', p_loss, collections=['pre_train'])
    pre_scalar_summary = tf.summary.merge_all('pre_train')

    hyp_str = '-spe_' + str(steps_per_env) + '-envs_' + str(number_envs) + '-cr_lr' + str(cr_lr) + '-crit_it_' + str(
        critic_iter) + '-delta_' + str(delta) + '-conj_iters_' + str(conj_iters)

    file_writer = tf.summary.FileWriter('log_dir/' + env_name + '/' + algorithm + '_' + clock_time + '_' + hyp_str,
                                        tf.get_default_graph())

    # create a session
    sess = tf.Session()
    # initialize the variables
    sess.run(tf.global_variables_initializer())

    def action_op(o):
        return sess.run([p_means, s_values], feed_dict={obs_ph: o})

    def action_op_noise(o):
        return sess.run([a_sampl, s_values], feed_dict={obs_ph: o})

    def model_op(o, a, md_idx):
        # TODO: Modified code by Simon
        mo = sess.run(nobs_pred_m[md_idx], feed_dict={obs_ph: [o], act_ph: [a[0]]})
        return np.squeeze(mo[:, :-1]), np.squeeze(mo[:, -1])

    def run_model_loss(model_idx, r_obs, r_act, r_nxt_obs, r_rew):
        # print({'obs_ph': r_obs.shape, 'act_ph': r_act.shape, 'nobs_ph': r_nxt_obs.shape})
        # TODO: Modified code by Simon
        r_act = np.squeeze(r_act, axis=2)
        # print(r_act.shape)
        r_rew = np.reshape(r_rew, (-1, 1))
        # print(r_rew.shape)
        return_val = sess.run(m_losses[model_idx],
                              feed_dict={obs_ph: r_obs, act_ph: r_act, nobs_ph: r_nxt_obs, rew_ph: r_rew})
        return return_val

    def run_model_opt_loss(model_idx, r_obs, r_act, r_nxt_obs, r_rew, mb_lr):
        # TODO: Modified code by Simon
        r_act = np.squeeze(r_act, axis=2)
        r_rew = np.reshape(r_rew, (-1, 1))
        return sess.run([m_opts[model_idx], m_losses[model_idx]],
                        feed_dict={obs_ph: r_obs, act_ph: r_act, nobs_ph: r_nxt_obs, rew_ph: r_rew, mb_lr_: mb_lr })
    def model_assign(i, model_variables_to_assign):
        '''
        Update the i-th model's parameters
        '''
        return sess.run(initialize_models[i], feed_dict={old_model_variables: model_variables_to_assign})

    def policy_update(obs_batch, act_batch, adv_batch, rtg_batch, it):
        # log probabilities, logits and log std of the "old" policy
        # "old" policy refer to the policy to optimize and that has been used to sample from the environment
        # TODO: Modified code by Simon
        act_batch = np.squeeze(act_batch, axis=2)
        old_p_log, old_p_means, old_log_std = sess.run([p_log, p_means, log_std],
                                                       feed_dict={obs_ph: obs_batch, act_ph: act_batch,
                                                                  adv_ph: adv_batch, ret_ph: rtg_batch})
        # get also the "old" parameters
        old_actor_params = sess.run(p_var_flatten)
        if it < 1:
            std_vals = sess.run([log_std], feed_dict={log_std: np.ones(act_dim)})
            # print(std_vals)
        # old_p_loss is later used in the line search
        # run pre_scalar_summary for a summary before the optimization
        old_p_loss, summary = sess.run([p_loss, pre_scalar_summary],
                                       feed_dict={obs_ph: obs_batch, act_ph: act_batch, adv_ph: adv_batch,
                                                  ret_ph: rtg_batch, old_p_log_ph: old_p_log})
        file_writer.add_summary(summary, step_count)

        file_writer.add_summary(summary, step_count)
        file_writer.flush()

        def H_f(p):
            '''
            Run the Fisher-Vector product on 'p' to approximate the Hessian of the DKL
            '''
            return sess.run(Fx,
                            feed_dict={old_mu_ph: old_p_means, old_log_std_ph: old_log_std, p_ph: p, obs_ph: obs_batch,
                                       act_ph: act_batch, adv_ph: adv_batch, ret_ph: rtg_batch})

        g_f = sess.run(p_grads_flatten,
                       feed_dict={old_mu_ph: old_p_means, obs_ph: obs_batch, act_ph: act_batch, adv_ph: adv_batch,
                                  ret_ph: rtg_batch, old_p_log_ph: old_p_log})
        ## Compute the Conjugate Gradient so to obtain an approximation of H^(-1)*g
        # Where H in reality isn't the true Hessian of the KL divergence but an approximation of it computed via Fisher-Vector Product (F)
        conj_grad = conjugate_gradient(H_f, g_f, iters=conj_iters)

        # Compute the step length
        beta_np = np.sqrt(2 * delta / (1e-10 + np.sum(conj_grad * H_f(conj_grad))))

        def DKL(alpha_v):
            '''
            Compute the KL divergence.
            It optimize the function to compute the DKL. Afterwards it restore the old parameters.
            '''
            sess.run(p_opt, feed_dict={beta_ph: beta_np, alpha: alpha_v, cg_ph: conj_grad, obs_ph: obs_batch,
                                       act_ph: act_batch, adv_ph: adv_batch, old_p_log_ph: old_p_log})
            a_res = sess.run([dkl_diverg, p_loss],
                             feed_dict={old_mu_ph: old_p_means, old_log_std_ph: old_log_std, obs_ph: obs_batch,
                                        act_ph: act_batch, adv_ph: adv_batch, ret_ph: rtg_batch,
                                        old_p_log_ph: old_p_log})
            sess.run(restore_params, feed_dict={p_old_variables: old_actor_params})
            return a_res

        # Actor optimization step
        # Different for TRPO or NPG
        # Backtracing line search to find the maximum alpha coefficient s.t. the constraint is valid
        best_alpha = backtracking_line_search(DKL, delta, old_p_loss, p=0.8)
        sess.run(p_opt, feed_dict={beta_ph: beta_np, alpha: best_alpha,
                                   cg_ph: conj_grad, obs_ph: obs_batch, act_ph: act_batch,
                                   adv_ph: adv_batch, old_p_log_ph: old_p_log})

        lb = len(obs_batch)
        shuffled_batch = np.arange(lb)
        np.random.shuffle(shuffled_batch)

        # Value function optimization steps
        for _ in range(critic_iter):
            # shuffle the batch on every iteration
            np.random.shuffle(shuffled_batch)
            for idx in range(0, lb, minibatch_size):
                minib = shuffled_batch[idx:min(idx + minibatch_size, lb)]
                sess.run(v_opt, feed_dict={obs_ph: obs_batch[minib], ret_ph: rtg_batch[minib]})

    def train_model(tr_obs, tr_act, tr_nxt_obs, tr_rew, v_obs, v_act, v_nxt_obs, v_rew, step_count, model_idx, mb_lr):

        # Get validation loss on the old model
        mb_valid_loss1 = run_model_loss(model_idx, v_obs, v_act, v_nxt_obs, v_rew)

        # Restore the random weights to have a new, clean neural network
        model_assign(model_idx, initial_variables_models[model_idx])

        mb_valid_loss = run_model_loss(model_idx, v_obs, v_act, v_nxt_obs, v_rew)

        acc_m_losses = []
        last_m_losses = []
        md_params = sess.run(models_variables[model_idx])
        best_mb = {'iter': 0, 'loss': mb_valid_loss, 'params': md_params}
        it = 0

        lb = len(tr_obs)
        shuffled_batch = np.arange(lb)
        np.random.shuffle(shuffled_batch)

        while best_mb['iter'] > it - model_iter:

            # update the model on each mini-batch
            last_m_losses = []
            for idx in range(0, lb, model_batch_size):
                minib = shuffled_batch[idx:min(idx + minibatch_size, lb)]

                if len(minib) != minibatch_size:
                    _, ml = run_model_opt_loss(model_idx, tr_obs[minib], tr_act[minib], tr_nxt_obs[minib],
                                               tr_rew[minib], mb_lr=mb_lr)
                    acc_m_losses.append(ml)
                    last_m_losses.append(ml)
                else:
                    pass
                    # print('Warning!')

            # Check if the loss on the validation set has improved
            mb_valid_loss = run_model_loss(model_idx, v_obs, v_act, v_nxt_obs, v_rew)
            if mb_valid_loss < best_mb['loss']:
                best_mb['loss'] = mb_valid_loss
                best_mb['iter'] = it
                best_mb['params'] = sess.run(models_variables[model_idx])

            it += 1
            # if it>=10000:
            #     break
            # print('iteration: ', it)

        # Restore the model with the lower validation loss
        model_assign(model_idx, best_mb['params'])

        print('Model:{}, iter:{} -- Old Val loss:{:.6f}  New Val loss:{:.6f} -- New Train loss:{:.6f}'.format(model_idx,
                                                                                                              it,
                                                                                                              mb_valid_loss1,
                                                                                                              best_mb[
                                                                                                                  'loss'],
                                                                                                              np.mean(
                                                                                                                  last_m_losses)))
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
        ax.plot(tests_all[0])
        ax.fill_between(x, tests_all[0] - tests_all[1], tests_all[0] + tests_all[1],
                        alpha=0.5)
        ax.set_ylabel('rewards tests vs. ground truth')
        # plt.tw
        ax2 = ax.twinx()

        color = 'lime'
        ax2.set_ylabel('entropy', color=color)  # we already handled the x-label with ax1
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.plot(entropy_all, color=color)
        # fig.show()
        plt.show()

    def save_data(data, **kwargs):
        '''logging functon'''
        # if 'directory_name' in kwargs:
        #     project_directory = kwargs.get('directory_name')
        now = datetime.now()
        clock_time = f'{now.month:0>2}_{now.day:0>2}_{now.hour:0>2}_{now.minute:0>2}_{now.second:0>2}'
        out_put_writer = open(project_directory + clock_time + '_training_observables', 'wb')
        pickle.dump(data, out_put_writer, -1)
        out_put_writer.close()

    # variable to store the total number of steps
    step_count = 0
    model_buffer = FullBuffer()
    print('Env batch size:', steps_per_env, ' Batch size:', steps_per_env * number_envs)

    # Create a simulated environment
    sim_env = NetworkEnv(make_env(), model_op, None, num_ensemble_models)

    # Get the initial parameters of each model
    # These are used in later epochs when we aim to re-train the models anew with the new dataset
    initial_variables_models = []
    for model_var in models_variables:
        initial_variables_models.append(sess.run(model_var))
    total_iterations = 0

    converged_flag = False
    # save_data = save_data(clock_time)
    sim_rewards_all = []
    sim_rewards_std_all = []
    entropy_all = []
    tests_all = []
    tests_std_all = []
    batch_rews_all = []
    batch_rews_std_all = []
    step_counts_all = []
    	
    for ep in range(num_epochs):
        if (converged_flag):
            print('Converged!!!!')
            break
        # lists to store rewards and length of the trajectories completed
        batch_rew = []
        batch_len = []
        print('============================', ep, '============================')
        # Execute in serial the environment, storing temporarily the trajectories.
        for env in envs:
            # Todo: Test randomization stronger if reward lower...we need a good scheme
            # target_threshold ?????
            init_log_std = np.ones(act_dim) * np.log(np.random.rand() * 1)
            env.reset()

            # iterate over a fixed number of steps
            steps_train = init_random_steps if ep == 0 else steps_per_env
            # steps_train = steps_per_env
            for _ in range(steps_train):
                # found = False
                # while not(found):
                # run the policy

                if ep == 0:
                    # Sample random action during the first epoch
                    act = np.random.uniform(-1, 1, size=env.action_space.shape[-1])

                else:

                    act = sess.run(a_sampl, feed_dict={obs_ph: [env.n_obs], log_std: init_log_std})
                    act = np.clip(act + np.random.randn(act.shape[0], act.shape[1]) * 0.1, -1, 1)

                act = np.squeeze(act)
                # print('act', act*12)
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
                    init_log_std = np.ones(act_dim) * np.log(np.random.rand() * 1)

                # if ep == 0:
                #     # try:
                #     # Initialize randomly a training and validation set
                #     model_buffer.generate_random_dataset()
                #     # get both datasets
                #     train_obs, train_act, train_rew, train_nxt_obs, _ = model_buffer.get_training_batch()
                #     valid_obs, valid_act, valid_rew, valid_nxt_obs, _ = model_buffer.get_valid_batch()
                #     target_threshold = max(max(valid_rew), max(train_rew))
                #     # print('-- '*38, target_threshold)
                #     found = target_threshold>=-0.1 and step_count>=191
                #     # except:
                #     #     pass

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

        # Initialize randomly a training and validation set
        model_buffer.generate_random_dataset()

        # get both datasets
        train_obs, train_act, train_rew, train_nxt_obs, _ = model_buffer.get_training_batch()
        valid_obs, valid_act, valid_rew, valid_nxt_obs, _ = model_buffer.get_valid_batch()
        std_vals = sess.run(log_std)
        print('Log Std policy:', std_vals, np.mean(std_vals))

        target_threshold = max(max(valid_rew), max(train_rew))
        sim_env.threshold = target_threshold  # min(target_threshold, -0.05)
        print('maximum: ', sim_env.threshold)

        # Learning rate as function of ep
        lr = lambda ep: 1e-3-ep/num_epochs*(1e-3-5e-4)
        mb_lr = 1e-3 #if ep<10 else 5e-5
        # simulated_steps = simulated_steps if ep<10 else 10000
        print('mb_lr: ', mb_lr)
        for i in range(num_ensemble_models):
            # train the dynamic model on the datasets just sampled
            train_model(train_obs, train_act, train_nxt_obs, train_rew, valid_obs, valid_act, valid_nxt_obs, valid_rew,
                        step_count, i, mb_lr=mb_lr)

        ############################################################
        ###################### POLICY LEARNING ######################
        ############################################################

        best_sim_test = -1e16 * np.ones(num_ensemble_models)

        # plot_results(env_test, f'Total {total_iterations}, '
        #                        f'data points: {len(model_buffer.train_idx) + len(model_buffer.valid_idx)}, '
        #                        f'modelit: {ep}')

        for it in range(max_training_iterations):
            if converged_flag:
                break
            total_iterations += 1
            print('\t Policy it', it, end='..')

            ##################### MODEL SIMLUATION #####################
            # obs_batch, act_batch, adv_batch, rtg_batch = simulate_environment(sim_env, action_op_noise, simulated_steps)
            batch, ep_length = simulate_environment(sim_env, action_op_noise, simulated_steps)
            # verification_simulate_environment(sim_env, env_test, action_op_noise, 50)
            obs_batch, act_batch, adv_batch, rtg_batch = batch

            ################# TRPO UPDATE ################
            policy_update(obs_batch, act_batch, adv_batch, rtg_batch, it)
            std_vals = sess.run(log_std)
            print('Log Std policy inner:', np.mean(std_vals))
            if np.mean(std_vals) < -5:
                converged_flag = True
            # Testing the policy on a real environment
            # mn_test, mn_test_std, mn_length = test_agent(env_test, action_op, num_games=1)
            # plot_results(env_test, 'ME-TRPO')
            # print(' Test score: ', np.round(mn_test, 2), np.round(mn_test_std, 2), np.round(mn_length, 2))
            # mn_test, mn_test_std, mn_length = test_agent(env_test, action_op, num_games=1)
            # summary = tf.Summary()
            # summary.value.add(tag='test/performance', simple_value=mn_test)
            # file_writer.add_summary(summary, step_count)
            # file_writer.flush()

            # Test the policy on simulated environment.
            # dynamic_wait_time_count = dynamic_wait_time(ep)
            if dynamic_wait_time(it, ep):
                print('Iterations: ', total_iterations)


                # perform test! -----------------------------
                # env_test.env.set_usage('test')
                # mn_test, mn_test_std, mn_length = test_agent(env_test, action_op, num_games=25)
                # perform test! -----------------------------
                label = f'Total {total_iterations}, ' + \
                        f'data points: {len(model_buffer.train_idx) + len(model_buffer.valid_idx)}, ' + \
                        f'ep: {ep}, it: {it}\n' + hyp_str_all

                # plot results of test -----------------------------
                # plot_results(env_test, label=label)

                # env_test.save_current_buffer(info=label)
                # print(' Test score: ', np.round(mn_test, 2), np.round(mn_test_std, 2), np.round(mn_length, 2))

                # save the data for plotting the tests
                # tests_all.append(mn_test)
                # tests_std_all.append(mn_test_std)

                # perform test end! -----------------------------
                env_test.env.set_usage('default')


                print('Simulated test:', end=' ** ')

                sim_rewards = []
                for i in range(num_ensemble_models):
                    sim_m_env = NetworkEnv(make_env(), model_op, None, i + 1)
                    mn_sim_rew, _, _ = test_agent(sim_m_env, action_op, num_games=10)
                    sim_rewards.append(mn_sim_rew)
                    print(mn_sim_rew, end=' ** ')

                print("")

                entropy_all.append(np.mean(std_vals))
                step_counts_all.append(step_count)

                sim_rewards = np.array(sim_rewards)
                sim_rewards_all.append(np.mean(sim_rewards))
                sim_rewards_std_all.append(np.std(sim_rewards))

                batch_rews_all.append(np.mean(batch_rew))
                batch_rews_std_all.append(np.std(batch_rew))

                data = dict(sim_rewards_all=[sim_rewards_all, sim_rewards_std_all],
                            entropy_all=entropy_all,
                            step_counts_all=step_counts_all,
                            batch_rews_all=[batch_rews_all, batch_rews_std_all],
                            tests_all=[tests_all, tests_std_all],
                            info=label)

                # save the data for plotting the progress -------------------
                save_data(data=data)

                # plotting the progress -------------------
                # plot_observables(data=data, label=label)

                # stop training if the policy hasn't improved
                if (np.sum(best_sim_test >= sim_rewards) > int(num_ensemble_models * 0.7)):
                    # or (len(sim_rewards[sim_rewards >= 990]) > int(num_ensemble_models * 0.7)):
                    if it > delay_before_convergence_check and ep<num_epochs-1:
                        # Test the entropy measure as convergence criterion
                        # if np.diff(entropy_all)[-1] < 0:
                        #     print('break')
                        #     break
                        print('break')
                        break
                else:
                    best_sim_test = sim_rewards
    # Final verification:
    # env_final = FelLocalEnv(tango=tango)
    # env_final.test = True
    # env.TOTAL_COUNTER = len(model_buffer.train_idx) + len(model_buffer.valid_idx)
    # mn_test, mn_test_std, mn_length = test_agent(env_final, action_op, num_games=100)
    # plot_results(env_final, 'ME-TRPO', save_name='Fermi')

    env_test.env.set_usage('final')
    mn_test, mn_test_std, mn_length = test_agent(env_test, action_op, num_games=50)

    label = f'Verification : total {total_iterations}, '+\
                           f'data points: {len(model_buffer.train_idx) + len(model_buffer.valid_idx)}, '+\
                           f'ep: {ep}, it: {it}\n'+\
                           f'rew: {mn_test}, std: {mn_test_std}'
    plot_results(env_test, label=label)

    env_test.save_current_buffer(info=label)

    # env_test.env.set_usage('default')

    # closing environments..
    for env in envs:
        env.close()
    file_writer.close()


if __name__ == '__main__':
    METRPO('', hidden_sizes=hidden_sizes, cr_lr=cr_lr, gamma=gamma, lam=lam, num_epochs=num_epochs,
           steps_per_env=steps_per_env,
           number_envs=number_envs, critic_iter=critic_iter, delta=delta, algorithm='TRPO', conj_iters=conj_iters,
           minibatch_size=minibatch_size,
           mb_lr_start=mb_lr, model_batch_size=model_batch_size, simulated_steps=simulated_steps,
           num_ensemble_models=num_ensemble_models, model_iter=model_iter, init_random_steps=init_random_steps)
    # plot the results

# important notes:
# Storage
# Hyperparameters
# Scaling

# Changes:
# No init steps and less step per env 31 instead of 51 and the number of iterations is dynamic

