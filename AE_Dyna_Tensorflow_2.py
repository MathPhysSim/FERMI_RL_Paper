import os
import pickle
import random
import sys
from datetime import datetime

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf

import gym
import numpy as np
from SAC_TFlayers import SAC, ReplayBuffer
from inverted_pendulum import PendulumEnv
from utilities import DynamicsModel, FullBuffer, StructEnv, MonitoringEnv, \
    NetworkEnv, test_agent, plot_observables, plot_results, train_agent


def aedyna(real_env, num_epochs=50, steps_per_env=100, algorithm='SAC', simulated_episodes=10,
           num_ensemble_models=2, model_batch_size=512, init_random_steps=200, **kwargs):
    if 'max_steps' in kwargs:
        max_steps = kwargs.get('max_steps')
    else:
        max_steps = 200

    def make_env(**kwargs):
        '''Create the environement'''
        return MonitoringEnv(env=real_env, project_directory=project_directory, max_steps=max_steps,
                             **kwargs)

    try:
        env_name = real_env.__name__
    except:
        env_name = 'default'

    # Create a few environments to collect the trajectories
    env = StructEnv(make_env())
    env_test = StructEnv(make_env(verification=True))

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    action_range = env.action_space.high

    if algorithm == 'SAC':
        REPLAY_BUFFER_SIZE = 5e5  # size of the replay buffer
        HIDDEN_DIM = 32  # size of hidden layers for networks
        SOFT_Q_LR = 3e-4  # q_net learning rate
        POLICY_LR = 3e-4  # policy_net learning rate
        ALPHA_LR = 3e-4  # alpha learning rate
        replay_buffer = ReplayBuffer(REPLAY_BUFFER_SIZE)
        agent = SAC(obs_dim, act_dim, action_range, HIDDEN_DIM, replay_buffer, SOFT_Q_LR, POLICY_LR, ALPHA_LR)
        # Store initial set-up
        agent.save()

    #########################################################
    ######################### MODEL #########################
    #########################################################

    # computational graph of N models and the correct losses for the anchor method
    m_classes = []

    for i in range(num_ensemble_models):
        m_class = DynamicsModel(obs_dim, act_dim)
        m_classes.append(m_class)

    # Time stamp for logging
    now = datetime.now()
    clock_time = "{}_{}_{}_{}".format(now.day, now.hour, now.minute, now.second)
    print('Time:', clock_time)

    def model_op(o, a, md_idx):
        """Calculate the predictions of the dynamics model"""
        next_obs, rew = m_classes[md_idx].predict({'states': np.array([o]), 'actions': np.array([a])})

        # return np.squeeze(mo[:, :-1]), float(np.squeeze(mo[:, -1]))
        return np.squeeze(next_obs), np.squeeze(rew)

    def train_model(train_set_x, train_set_y, n_data, model_idx=0):

        # Restore the initial random weights to have a new, clean neural network
        # initial_variables_models - list stored before already in the code below -
        # important for the anchor method
        m_classes[model_idx].restore()

        # Get validation loss on the now initialized model
        mb_valid_loss = m_classes[model_idx].evaluate(train_set_x, train_set_y)

        results = m_classes[model_idx].train_model(train_set_x, train_set_y, n_data, batch_size=model_batch_size)
        print(f'Model: {model_idx}, loss: {results.history["loss"][-1]}')

    # def plot_results(env_wrapper, label=None, **kwargs):
    #     """ Plot the validation episodes"""
    #     rewards = env_wrapper.env.current_buffer.get_data()['rews']
    #
    #     iterations = []
    #     finals = []
    #     means = []
    #     stds = []
    #
    #     for i in range(len(rewards)):
    #         if len(rewards[i]) > 1:
    #             finals.append(rewards[i][-1])
    #             # means.append(np.mean(rewards[i][1:]))
    #             means.append(np.sum(rewards[i][1:]))
    #             stds.append(np.std(rewards[i][1:]))
    #             iterations.append(len(rewards[i]))
    #     x = range(len(iterations))
    #     iterations = np.array(iterations)
    #     finals = np.array(finals)
    #     means = np.array(means)
    #     stds = np.array(stds)
    #
    #     plot_suffix = label
    #
    #     fig, axs = plt.subplots(2, 1, sharex=True)
    #
    #     ax = axs[0]
    #     ax.plot(x, iterations)
    #     ax.set_ylabel('Iterations (1)')
    #     ax.set_title(plot_suffix)
    #
    #     if 'data_number' in kwargs:
    #         ax1 = plt.twinx(ax)
    #         color = 'lime'
    #         ax1.set_ylabel('Mean reward', color=color)  # we already handled the x-label with ax1
    #         ax1.tick_params(axis='y', labelcolor=color)
    #         ax1.plot(x, kwargs.get('data_number'), color=color)
    #
    #     ax = axs[1]
    #     color = 'blue'
    #     ax.set_ylabel('Final reward', color=color)  # we already handled the x-label with ax1
    #     ax.tick_params(axis='y', labelcolor=color)
    #     ax.plot(x, finals, color=color)
    #
    #     ax.set_title('Final reward per episode')  # + plot_suffix)
    #     ax.set_xlabel('Episodes (1)')
    #
    #     ax1 = plt.twinx(ax)
    #     color = 'lime'
    #     ax1.set_ylabel('Mean reward', color=color)  # we already handled the x-label with ax1
    #     ax1.tick_params(axis='y', labelcolor=color)
    #     ax1.fill_between(x, means - stds, means + stds,
    #                      alpha=0.5, edgecolor=color, facecolor='#FF9848')
    #     ax1.plot(x, means, color=color)
    #
    #     if 'save_name' in kwargs:
    #         plt.savefig(kwargs.get('save_name') + '.pdf')
    #     plt.show()

    # def plot_observables(data, label, **kwargs):
    #     """plot observables during the test"""
    #
    #     sim_rewards_all = np.array(data.get('sim_rewards_all'))
    #     step_counts_all = np.array(data.get('step_counts_all'))
    #     batch_rews_all = np.array(data.get('batch_rews_all'))
    #     tests_all = np.array(data.get('tests_all'))
    #
    #     fig, axs = plt.subplots(2, 1, sharex=True)
    #     x = np.arange(len(batch_rews_all[0]))
    #     ax = axs[0]
    #     ax.step(x, batch_rews_all[0])
    #     ax.fill_between(x, batch_rews_all[0] - batch_rews_all[1], batch_rews_all[0] + batch_rews_all[1],
    #                     alpha=0.5)
    #     ax.set_ylabel('rews per batch')
    #
    #     ax.set_title(label)
    #
    #     ax2 = ax.twinx()
    #
    #     color = 'lime'
    #     ax2.set_ylabel('data points', color=color)  # we already handled the x-label with ax1
    #     ax2.tick_params(axis='y', labelcolor=color)
    #     ax2.step(x, step_counts_all, color=color)
    #
    #     ax = axs[1]
    #     ax.plot(sim_rewards_all[0], ls=':', label='sim')
    #     ax.fill_between(x, sim_rewards_all[0] - sim_rewards_all[1], sim_rewards_all[0] + sim_rewards_all[1],
    #                     alpha=0.5)
    #     try:
    #         ax.plot(tests_all[0], label='real')
    #         ax.fill_between(x, tests_all[0] - tests_all[1], tests_all[0] + tests_all[1],
    #                         alpha=0.5)
    #         ax.axhline(y=np.max(tests_all[0]), c='orange')
    #     except:
    #         pass
    #     ax.set_ylabel('rewards tests')
    #     ax.legend(loc="upper right")
    #     # plt.tw
    #     ax.grid(True)
    #     ax2 = ax.twinx()
    #
    #     color = 'lime'
    #     ax2.set_ylabel('success', color=color)  # we already handled the x-label with ax1
    #     ax2.tick_params(axis='y', labelcolor=color)
    #     ax2.plot(length_all, color=color)
    #
    #     fig.align_labels()
    #     plt.show()

    def save_data(data, **kwargs):
        '''logging function to save results to pickle'''
        now = datetime.now()
        clock_time = f'{now.month:0>2}_{now.day:0>2}_{now.hour:0>2}_{now.minute:0>2}_{now.second:0>2}'
        out_put_writer = open(project_directory + clock_time + '_training_observables', 'wb')
        pickle.dump(data, out_put_writer, -1)
        out_put_writer.close()

        # variable to store the total number of steps

    step_count = 0
    model_buffer = FullBuffer()
    # print('Env batch size:', steps_per_env, ' Batch size:', steps_per_env)

    # Create a simulated environment
    sim_env = NetworkEnv(make_env(), model_op, None,
                         num_ensemble_models, project_directory=project_directory)

    total_iterations = 0

    sim_rewards_all = []
    sim_rewards_std_all = []
    length_all = []
    tests_all = []
    tests_std_all = []
    batch_rews_all = []
    batch_rews_std_all = []
    step_counts_all = []

    # agent = SAC(obs_dim, act_dim, action_range, HIDDEN_DIM, replay_buffer, SOFT_Q_LR, POLICY_LR, ALPHA_LR)
    # init the agent
    # agent.load_weights()
    for ep in range(num_epochs):

        # lists to store rewards and length of the trajectories completed
        batch_rew = []
        batch_len = []
        print('============================', ep, '============================')
        # Execute in serial the environment, storing temporarily the trajectories.
        env.reset()

        # iterate over a fixed/variable number of steps
        # steps_train = init_random_steps if ep == 0 else steps_per_env // ep
        steps_train = init_random_steps if ep == 0 else steps_per_env

        for _ in range(steps_train):
            # run the policy
            if ep == 0:
                # Sample random action during the first epoch
                act = agent.policy_net.sample_action()
            else:
                # Add artificial noise
                noise = 0.01 * np.random.randn(act_dim)
                act = agent.policy_net.get_action(env.n_obs.astype(np.float32))
                act = np.clip(np.squeeze(act) + noise, -1, 1)
            # take a step in the environment
            obs2, rew, done, _ = env.step(act)
            # add the new transition to the temporary buffer
            model_buffer.store(env.n_obs.copy(), act, rew.copy(), obs2.copy(), done)

            env.n_obs = obs2.copy()
            step_count += 1

            if done:
                batch_rew.append(env.get_episode_reward())
                batch_len.append(env.get_episode_length())

                env.reset()

        # save the data for plotting the collected data for the model
        # env.save_current_buffer()

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

        for i in range(num_ensemble_models):
            # Initialize randomly a training and validation set
            train_set_x, train_set_y, n_data = model_buffer.generate_random_dataset()
            train_model(train_set_x, train_set_y, n_data=n_data, model_idx=i)

        ############################################################
        ###################### POLICY LEARNING #####################
        ############################################################
        data = model_buffer.get_maximum()
        # print(data)
        label = f'Total {total_iterations}, ' + \
                f'data points: {len(model_buffer)}, ' + \
                f'ep: {ep}, max: {data}\n' + hyp_str_all
        # sim_env.visualize(data=data, label=label)

        best_sim_test = -1e16 * np.ones(num_ensemble_models)
        replay_buffer = ReplayBuffer(REPLAY_BUFFER_SIZE)
        agent = SAC(obs_dim, act_dim, action_range, HIDDEN_DIM, replay_buffer, SOFT_Q_LR, POLICY_LR, ALPHA_LR)
        frame_idx = 0
        # init the agent
        # agent.load_weights()

        for it in range(max_training_iterations):
            total_iterations += 1
            print('\t Policy it', it, end='..\n')

            ################# Agent UPDATE ################
            frame_idx = train_agent(sim_env, agent, TRAIN_EPISODES=simulated_episodes,
                                    frame_idx_in=frame_idx, replay_buffer=replay_buffer)
            ################# Agent Test time ################
            # if dynamic_wait_time(it, ep):
            if it >= 1:
                print('Iterations: ', total_iterations)
                label = f'Total {total_iterations}, ' + \
                        f'data points: {len(model_buffer)}, ' + \
                        f'ep: {ep}, it: {it}\n' + hyp_str_all
                env_test.env.set_usage('test')
                mn_test, mn_test_std, mn_length, mn_success = test_agent(env_test, agent.policy_net.get_action,
                                                                         num_games=10)
                print(' Test score on real environment: ', np.round(mn_test, 2), np.round(mn_test_std, 2),
                      np.round(mn_length, 2), np.round(mn_success, 2))

                # plot results of test
                # plot_results(env_test, label=label)
                # env_test.save_current_buffer(info=label)

                # Save the data for plotting the tests
                tests_all.append(mn_test)
                tests_std_all.append(mn_test_std)
                length_all.append(mn_length)

                env_test.env.set_usage('default')

                print('Simulated test individual networks:', end=' ** ')

                sim_rewards = []
                for i in range(num_ensemble_models):
                    def model_op_new(o, a):
                        return model_op(o, a, i)

                    # sim_m_env = NetworkEnv(make_env(), lambda o, a: model_op(o, a, i), None, number_models=i,
                    #                        verification=True, project_directory=project_directory)

                    sim_m_env = NetworkEnv(make_env(), model_op_new, None, number_models=i,
                                           verification=True, project_directory=project_directory)

                    mn_sim_rew, _, _, _ = test_agent(sim_m_env, agent.policy_net.get_action, num_games=10)
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
                # save_data(data=data)

                # plotting the progress -------------------
                # if it % 10 == 0:
                fig = plot_observables(data=data, label=label, length_all=length_all)

                # stop training if the policy hasn't improved
                if (np.sum(best_sim_test >= sim_rewards) >= int(num_ensemble_models * 0.7)):
                    if it > delay_before_convergence_check and ep < num_epochs - 1:
                        print('break-no improvement in', int(num_ensemble_models * 0.7), ' models')
                        break
                else:
                    best_sim_test = sim_rewards

    # Store the observables at the end
    fig.savefig('Final_Observables.png')
    # Final verification:
    env_test.env.set_usage('final')
    mn_test, mn_test_std, mn_length, _ = test_agent(env_test, agent.policy_net.get_action, num_games=50)

    label = f'Verification : total {total_iterations}, ' + \
            f'data points: {len(model_buffer.train_idx) + len(model_buffer.valid_idx)}, ' + \
            f'ep: {ep}, it: {it}\n' + \
            f'rew: {mn_test}, std: {mn_test_std}'
    plot_results(env_test, label=label)

    # env_test.save_current_buffer(info=label)

    env_test.env.set_usage('default')

    # closing environments..
    env.close()
    # file_writer.close()


if __name__ == '__main__':

    ############################################################
    # Hyperparameters
    ############################################################
    # Steps in real environment each epoch for the model training
    steps_per_epoch = 200
    total_steps = 1400
    # Initial steps in real environment at the beginning for the model training
    init_random_steps = 200
    num_epochs = int((total_steps - init_random_steps) / steps_per_epoch) + 1

    # number of epochs - total time of retraining the model
    # num_epochs = 6
    print('Number of epochs: ', num_epochs)

    # maximum number of iteration if the policy does not converge fast enough
    max_training_iterations = 15
    # delay before check since the rl-algorithm needs time to perform
    delay_before_convergence_check = 2

    # number of simulated episodes before checking the performance
    simulated_episodes = 6

    # parameters for the model network
    model_batch_size = 64
    num_ensemble_models = 3

    # How often to check the progress of the network training
    # e.g. lambda it, episode: (it + 1) % max(3, (ep+1)*2) == 0
    dynamic_wait_time = lambda it, ep: (it + 1) % 2 == 0  #

    # Set max episode length manually here for the pendulum
    max_steps = 200

    try:
        random_seed = int(sys.argv[2])
    except:
        random_seed = 25
    try:
        file_name = sys.argv[1] + '_' + str(random_seed)
    except:
        file_name = 'defaultexp_noise_' + str(random_seed) + '_'

    # reproducible

    random.seed(random_seed)
    np.random.seed(random_seed)
    tf.random.set_seed(random_seed)

    try:
        root_dir = sys.argv[3]
    except:
        root_dir = 'Data/Simulation/'

    directory = root_dir + file_name + '/'

    if not os.path.exists(directory):
        os.makedirs(directory)

    try:
        # clipped_double_q
        index = int(sys.argv[4])
        parameter_list = [
            dict(noise=0.0, data_noise=0, models=1),
            dict(noise=0.0, data_noise=0, models=3),
            dict(noise=0.0, data_noise=0, models=5),
            dict(noise=0.0, data_noise=0, models=10)
        ]
        parameters = parameter_list[index]
        print('Running...', parameters)
    except:
        parameters = dict(noise=0.0, data_noise=0.0, models=num_ensemble_models)
        print('Running default...', parameters)

    directory = root_dir + file_name + '/'

    # Create the logging directory:
    project_directory = directory  # 'Data_logging/Simulation/'

    num_ensemble_models = parameters.get('models')
    hyp_str_all = 'nr_steps_' + str(steps_per_epoch) + '-n_ep_' + str(num_epochs) + \
                  '-m_bs_' + str(model_batch_size) + \
                  '-sim_steps_' + str(simulated_episodes) + \
                  '-ensnr_' + str(num_ensemble_models) + '-init_' + str(
        init_random_steps) + '/'
    project_directory = project_directory + hyp_str_all

    # To label the plots:
    hyp_str_all = '-nr_steps_' + str(steps_per_epoch) + '-n_ep_' + str(num_epochs) + \
                  '-m_bs_' + str(model_batch_size) + \
                  '-sim_steps_' + str(simulated_episodes) + \
                  '\n-ensnr_' + str(num_ensemble_models)

    if not os.path.isdir(project_directory):
        os.makedirs(project_directory)
        print("created folder : ", project_directory)


    ############################################################
    # Loading the environment
    ############################################################
    class TestWrapperEnv(gym.Wrapper):
        """
        Gym Wrapper to add noise and visualise.
        """

        def __init__(self, env, render=False, **kwargs):
            """
            :param env: open gym environment
            :param kwargs: noise
            :param render: flag to render
            """
            self.showing_render = render
            self.current_step = 0
            if 'noise' in kwargs:
                self.noise = kwargs.get('noise')
            else:
                self.noise = 0.0

            gym.Wrapper.__init__(self, env)

        def reset(self, **kwargs):
            self.current_step = 0
            obs = self.env.reset(**kwargs) + self.noise * np.random.randn(self.env.observation_space.shape[-1])
            return obs

        def step(self, action):
            self.current_step += 1
            obs, reward, done, info = self.env.step(action)
            if self.current_step >= 200:
                done = True
            if self.showing_render:
                # Simulate and visualise the environment
                self.env.render()
            obs = obs + self.noise * np.random.randn(self.env.observation_space.shape[-1])
            # reward = reward / 10
            return obs, reward, done, info


    # To test the noise robustness
    # env = TestWrapperEnv(PendulumEnv(), render=False, noise=parameters.get('noise'))
    env = PendulumEnv()
    env.seed(random_seed)
    # if a video should be made
    # real_env = gym.wrappers.Monitor(env, "recordings_new", force=True)

    real_env = env

    aedyna(real_env=real_env, num_epochs=num_epochs,
           steps_per_env=steps_per_epoch, algorithm='SAC', model_batch_size=model_batch_size,
           simulated_episodes=simulated_episodes,
           num_ensemble_models=num_ensemble_models, init_random_steps=init_random_steps, max_steps=max_steps)
