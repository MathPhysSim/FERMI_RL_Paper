import os
import pickle
import shutil
import matplotlib.pyplot as plt
import gym


'''This is a version of the NAF Article (Gu2016)
 Gu, S.; Lillicrap, T.; Sutskever, I. & Levine, S.
 Continuous Deep Q-Learning with Model-based Acceleration 2016
with some modifications as explained in the paper.
We use tensorflow 2.3'''

import tensorflow as tf
from tensorflow import keras
# Turn off warnings form tensorflow
tf.get_logger().setLevel('ERROR')

tf.keras.backend.set_floatx('float64')
import numpy as np
from tqdm import tqdm

class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for NAF_debug agents.
    """

    def __init__(self, obs_dim, act_dim, size):
        self.obs1_buf = np.zeros([size, obs_dim], dtype=np.float64)
        self.obs2_buf = np.zeros([size, obs_dim], dtype=np.float64)
        self.acts_buf = np.zeros([size, act_dim], dtype=np.float64)
        self.rews_buf = np.zeros(size, dtype=np.float64)
        self.done_buf = np.zeros(size, dtype=np.float64)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        self.obs1_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size=32):
        if self.size < batch_size:
            idxs = np.arange(self.size)
        else:
            idxs = np.random.randint(0, self.size, size=batch_size)

        return dict(obs1=self.obs1_buf[idxs],
                    obs2=self.obs2_buf[idxs],
                    acts=self.acts_buf[idxs],
                    rews=self.rews_buf[idxs],
                    done=self.done_buf[idxs])

    def save_to_pkl(self, name, directory):
        idxs = np.arange(self.size)
        buffer_data = dict(obs1=self.obs1_buf[idxs],
                           obs2=self.obs2_buf[idxs],
                           acts=self.acts_buf[idxs],
                           rews=self.rews_buf[idxs],
                           done=self.done_buf[idxs])
        f = open(directory + name, "wb")
        pickle.dump(buffer_data, f)
        f.close()

    def read_from_pkl(self, name, directory):
        with open(directory + name, 'rb') as f:
            buffer_data = pickle.load(f)
        obs1s, obs2s, acts, rews, dones = [buffer_data[key] for key in buffer_data]
        for i in range(len(obs1s)):
            self.store(obs1s[i], acts[i], rews[i], obs2s[i], dones[i])
        # print(self.size)

def basic_loss_function(y_true, y_pred):
    return tf.math.reduce_mean(y_true - y_pred)


class NormalizeEnv(gym.Wrapper):
    '''
    Gym Wrapper to normalize the environment
    '''

    def __init__(self, env, **kwargs):
        gym.Wrapper.__init__(self, env)

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

    def reset(self, **kwargs):
        return self.scale_state_env(self.env.reset(**kwargs))

    def step(self, action):
        # TODO: check the dimensions
        ob, reward, done, info = self.env.step(self.descale_action_env(action)[0])
        return self.scale_state_env(ob), reward, done, info

    def descale_action_env(self, act):
        scale = (self.env.action_space.high - self.env.action_space.low)
        return_value = (scale * act + self.env.action_space.high + self.env.action_space.low) / 2
        return return_value

    def scale_state_env(self, ob):
        scale = (self.env.observation_space.high - self.env.observation_space.low)
        return (2 * ob - (self.env.observation_space.high + self.env.observation_space.low)) / scale


class QModel:
    """Artificial neural net holding the state-action value function in a simple analytical form"""

    def __init__(self, obs_dim=2, act_dim=2, **kwargs):
        if 'directory' in kwargs:
            self.directory = kwargs.get('directory')

        if 'save_frequency' in kwargs:
            self.save_frequency = kwargs.get('save_frequency')
        else:
            self.save_frequency = 500

        if 'hidden_sizes' in kwargs:
            self.hidden_sizes = kwargs.get('hidden_sizes')
        else:
            self.hidden_sizes = (100, 100)

        if 'early_stopping' in kwargs:
            self.callback = tf.keras.callbacks.EarlyStopping(monitor='mae',
                                                             patience=kwargs.get('early_stopping'))
        else:
            self.callback = tf.keras.callbacks.EarlyStopping(monitor='mae', patience=2)

        if 'name' in kwargs:
            self.__name__ = kwargs.get('name')
            print(self.__name__)

        if 'learning_rate' in kwargs:
            self.learning_rate = kwargs.get('learning_rate')
            del kwargs['learning_rate']
        else:
            self.learning_rate = 1e-3

        if 'directory' in kwargs:
            self.directory = kwargs.get('directory')
        else:
            self.directory = None

        if 'clipped_double_q' in kwargs:
            self.clipped_double_q = kwargs.get('clipped_double_q')
        else:
            self.clipped_double_q = False

        if 'kernel_initializer' in kwargs:
            self.kernel_initializer = kwargs.get('kernel_initializer')
        else:
            self.kernel_initializer = tf.compat.v1.random_uniform_initializer(-0.01, 0.01)

        self.init = True

        self.act_dim = act_dim
        self.obs_dim = obs_dim

        # Define the network inputs (state-action)
        inputs_state = keras.Input(shape=(self.obs_dim,), name="state_input")
        inputs_action = keras.Input(shape=(self.act_dim,), name="action_input")

        # create a shared network for the variables
        h = inputs_state
        for hidden_dim in self.hidden_sizes:
            h = self.fc(h, hidden_dim, kernel_initializer=self.kernel_initializer)

        # Output - state-value function, where the reward is assumed to be negative
        V = tf.scalar_mul(-1, self.fc(h, 1, activation=tf.nn.leaky_relu,
                                      kernel_initializer=self.kernel_initializer, name='V'))
        # Output - for the matrix L
        l = self.fc(h, (self.act_dim * (self.act_dim + 1) / 2),
                    kernel_initializer=self.kernel_initializer, name='l')
        # Output - policy pi
        mu = self.fc(h, self.act_dim, kernel_initializer=self.kernel_initializer, name='mu')
        self.value_model = keras.Model([inputs_state], V, name='value_model')
        self.action_model = keras.Model([inputs_state], mu, name='action_model')

        pivot = 0
        rows = []
        for idx in range(self.act_dim):
            count = self.act_dim - idx
            diag_elem = tf.exp(tf.slice(l, (0, pivot), (-1, 1)))
            non_diag_elems = tf.slice(l, (0, pivot + 1), (-1, count - 1))
            row = tf.pad(tensor=tf.concat((diag_elem, non_diag_elems), 1), paddings=((0, 0), (idx, 0)))
            rows.append(row)
            pivot += count
        L = tf.transpose(a=tf.stack(rows, axis=1), perm=(0, 2, 1))
        P = tf.matmul(L, tf.transpose(a=L, perm=(0, 2, 1)))
        tmp = tf.expand_dims(inputs_action - mu, -1)
        # The advantage function
        A = -tf.multiply(tf.matmul(tf.transpose(a=tmp, perm=[0, 2, 1]),
                                   tf.matmul(P, tmp)), tf.constant(0.5, dtype=tf.float64))
        A = tf.reshape(A, [-1, 1])

        # The state-action-value function
        Q = tf.add(A, V)

        # We use a customized way to train the model:
        self.optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.q_model = self.CustomModel(inputs=[inputs_state, inputs_action], outputs=Q, mother_class=self)
        self.q_model.compile(optimizer=self.optimizer, loss="mse", metrics=["mae"])

        self.storage_management()

    def storage_management(self):
        checkpoint_dir = self.directory + self.__name__ + "/"
        self.ckpt = tf.train.Checkpoint(step=tf.Variable(1), model=self.q_model)
        self.manager = tf.train.CheckpointManager(self.ckpt, checkpoint_dir, max_to_keep=3)
        self.ckpt.restore(self.manager.latest_checkpoint)
        if self.manager.latest_checkpoint:
            print("Restored from {}".format(self.manager.latest_checkpoint))
        else:
            print("Initializing from scratch.")

    def fc(self, x, hidden_size, activation=tf.nn.tanh,
           kernel_initializer=tf.compat.v1.random_uniform_initializer(-0.01, 0.01),
           name=None):
        layer = keras.layers.Dense(hidden_size, activation=activation,
                                   kernel_initializer=kernel_initializer,
                                   kernel_regularizer=None,
                                   bias_initializer=tf.compat.v1.constant_initializer(0.0), name=name)
        return layer(x)

    def get_action(self, state):
        return self.action_model.predict(np.array(state))

    def get_value_estimate(self, state):
        return self.value_model.predict(np.array(state))

    def set_polyak_weights(self, weights, polyak=0.999, **kwargs):
        weights_old = self.get_weights()
        weights_new = [polyak * weights_old[i] + (1 - polyak) * weights[i] for i in range(len(weights))]
        self.q_model.set_weights(weights=weights_new)

    def get_weights(self):
        return self.q_model.get_weights()

    def save_model(self, directory):
        try:
            self.q_model.save(filepath=directory, overwrite=True)
        except:
            print('Saving failed')

    def set_target_models(self, q_target_1, q_target_2=None):
        self.q_target_first = q_target_1
        if q_target_2 is not None:
            self.q_target_second = q_target_2

    class CustomModel(keras.Model):

        def __init__(self, *args, **kwargs):
            self.mother_class = kwargs.pop('mother_class')
            self.__name__ = self.mother_class.__name__

            super().__init__(*args, **kwargs)

        def train_step(self, batch):
            self.discount = self.mother_class.discount
            self.polyak = self.mother_class.polyak

            v_1 = self.mother_class.q_target_first.value_model(batch['obs2'])  # , training=False)
            if self.mother_class.clipped_double_q:
                v_2 = self.mother_class.q_target_second.value_model(batch['obs2'])  # , training=False)
                v = tf.squeeze(tf.where(tf.math.less(v_1, v_2), v_1, v_2))
            else:
                v = tf.squeeze(v_1)

            y_target = tf.add(tf.multiply(tf.math.scalar_mul(self.discount, v),
                                          tf.add(tf.constant(1, dtype=tf.float64),
                                                 tf.math.scalar_mul(-1, batch['done']))), batch['rews'])

            # Double Q implementation
            # a_1 = self.mother_class.q_target_first.action_model(batch['obs2'])
            # q_2 = self.mother_class.q_target_second.q_model([batch['obs2'], a_1])
            #
            # y_target = tf.add(tf.multiply(tf.math.scalar_mul(self.discount, q_2),
            #                               tf.add(tf.constant(1, dtype=tf.float64),
            #                                      tf.math.scalar_mul(-1, batch['done']))), batch['rews'])

            with tf.GradientTape() as tape:
                # Run the forward pass of the layer.
                # The operations that the layer applies
                # to its inputs are going to be recorded
                # on the GradientTape.
                y_pred = self([batch['obs1'], batch['acts']], training=True)
                # Compute the loss value for this minibatch.
                loss = self.compiled_loss(y_target, y_pred)
            # Use the gradient tape to automatically retrieve
            # the gradients of the trainable variables with respect to the loss.
            # Compute gradients
            trainable_vars = self.trainable_weights
            gradients = tape.gradient(loss, trainable_vars)
            # Run one step of gradient descent by updating
            # the value of the variables to minimize the loss.
            # Update weights
            self.optimizer.apply_gradients(zip(gradients, trainable_vars))
            # Update the metrics.
            # Metrics are configured in `compile()`.
            self.compiled_metrics.update_state(y_target, y_pred)
            # Apply weights to target network
            return {m.name: m.result() for m in self.metrics}

    class CustomCallback(keras.callbacks.Callback):

        def __init__(self, patience=0):
            # super(self.CustomCallback, self).__init__()
            super().__init__()
            self.patience = patience
            # best_weights to store the weights at which the minimum loss occurs.
            self.best_weights = None
            self.q_target = None

        def on_train_begin(self, logs=None):
            # The number of epoch it has waited when loss is no longer minimum.
            self.wait = 0
            # The epoch the training stops at.
            self.stopped_epoch = 0
            # Initialize the best as infinity.
            self.best = np.Inf

        def on_epoch_end(self, epoch, logs=None):
            current = logs.get("loss")
            # if np.less(current, self.best):
            #     self.best = current
            #     self.wait = 0
            #     # Record the best weights if current results is better (less).
            #     self.best_weights = self.model.get_weights()
            # else:
            #     self.wait += 1
            #     if self.wait >= self.patience:
            #         self.stopped_epoch = epoch
            #         self.model.stop_training = True
            #         # print("Restoring model weights from the end of the best epoch.")
            #         self.model.set_weights(self.best_weights)
            # Apply weights to target network
            # self.q_target.set_polyak_weights(self.model.get_weights(),
            #                                  self.model.polyak, name=self.model.__name__)

        def on_train_batch_end(self, batch, logs=None):
            # Apply weights to target network
            self.q_target.set_polyak_weights(self.model.get_weights(),
                                             self.model.polyak, name=self.model.__name__)
        # def on_train_end(self, logs=None):
        #     if self.stopped_epoch > 0:
        #         print("Epoch %05d: early stopping" % (self.stopped_epoch + 1))
        # self.q_target.set_polyak_weights(self.model.get_weights(),
        #                                  polyak=0.999)
        # print('end of training')

        # def on_train_batch_end(self, batch, logs=None):
        #     keys = list(logs.keys())
        #
        #     # self.q_target.set_polyak_weights(self.model.get_weights(),
        #     #                                               polyak=0.999)
        #     # print('updated', self.q_target.__name__)
        #     print("...Training: end of batch {}; got log keys: {}".format(batch, keys))
        #     # print(self.model.y_target)

    def set_training_parameters(self, **kwargs):
        # Filter kwargs for keras
        if 'polyak' in kwargs:
            self.polyak = kwargs.pop('polyak')
        else:
            self.polyak = 0.999
        if 'discount' in kwargs:
            self.discount = kwargs.pop('discount')
        else:
            self.discount = 0.999
        if 'steps_per_batch' in kwargs:
            self.steps_per_batch = kwargs.pop('steps_per_batch')
        else:
            self.steps_per_batch = 1
        if 'batch_size' in kwargs:
            self.batch_size = kwargs.pop('batch_size')
        else:
            self.batch_size = 1

        self.callback = self.CustomCallback(patience=0)
        self.callback.q_target = self.q_target_first

        self.training_params = kwargs

    def train_model(self, **kwargs):

        # for key in kwargs:
        #     self.training_params[key] = kwargs.get(key)

        batch = self.replay_buffer.sample_batch(batch_size=self.batch_size)
        # Here we decide how often to iterate over the data
        dataset = tf.data.Dataset.from_tensor_slices(batch)  # .repeat(1).shuffle(buffer_size=10000)
        train_dataset = dataset.batch(self.steps_per_batch)

        hist = self.q_model.fit(train_dataset,
                                verbose=0,
                                callbacks=[self.callback],
                                shuffle=True,
                                **self.training_params)

        if int(self.ckpt.step) % self.save_frequency == 0:
            save_path = self.manager.save()
            print("Saved checkpoint for step {}: {}".format(int(self.ckpt.step), save_path))
            save_path_target = self.q_target_first.manager.save()
            print("Saved checkpoint for step {}: {}".format(int(self.ckpt.step), save_path_target))
        self.ckpt.step.assign_add(1)
        return_value = hist.history['loss']

        return return_value

    def create_buffers(self, buffer=None):
        if buffer is None:
            self.replay_buffer = ReplayBuffer(obs_dim=self.obs_dim, act_dim=self.act_dim, size=int(1e6))
            try:
                files = []
                directory = self.directory + 'data/'
                for f in os.listdir(directory):
                    if 'buffer_data' in f and 'pkl' in f:
                        files.append(f)
                files.sort()
                self.replay_buffer.read_from_pkl(name=files[-1], directory=directory)
                print('Buffer data loaded for ' + self.__name__, files[-1])
            except:
                print('Buffer data empty for ' + self.__name__, files)
        else:
            self.replay_buffer = buffer


class NAF(object):
    def __init__(self, env, training_info=dict(), pre_tune=None,
                 noise_info=dict(), save_frequency=500, directory=None, is_continued=False,
                 clipped_double_q=False, q_smoothing_sigma=0.02, q_smoothing_clip=0.05, **nafnet_kwargs):
        """
        :param env: open gym environment to be solved
        :dict training_info: dictionary containing info for the training of the network
        :tuple pre_tune: list of tuples (state action reward next state done)
        :param noise_info: dict with noise function for decay of gaussian noise
        :param save_frequency: frequency to save the weights of the network
        :param directory: directory were weights are saved
        :param is_continued: continue a training, otherwise given directory deleted if existing
        :param clipped_double_q: use the clipped double q trick with switching all clipped_double_q steps
        :param q_smoothing_clip: add small noise on actions to smooth the training
        :param q_smoothing: add small noise on actions to smooth the training
        :param nafnet_kwargs: keywords to handle the network and training
        """

        self.clipped_double_q = clipped_double_q
        self.q_smoothing_clip = q_smoothing_clip
        self.q_smoothing_sigma = q_smoothing_sigma

        self.losses2 = []
        self.vs2 = []
        self.model_switch = 1

        self.directory = directory
        self.save_frequency = save_frequency

        self.losses = []
        self.pre_tune = pre_tune

        self.env = NormalizeEnv(env)

        if 'noise_function' in noise_info:
            self.noise_function = noise_info.get('noise_function')
        else:
            self.noise_function = lambda action, nr: action + np.random.randn(self.action_size) * 1 / (nr + 1)

        self.action_size = self.env.action_space.shape[0]
        self.observation_size = self.env.observation_space.shape[0]

        self.max_steps = 1000

        self.idx_episode = None
        self.vs = []

        self.training_info = training_info

        if 'learning_rate' in training_info:
            learning_rate = training_info.get('learning_rate')
            del training_info['learning_rate']
        else:
            learning_rate = 1e-3

        if not is_continued:
            shutil.rmtree(self.directory)
            os.makedirs(self.directory)
            os.makedirs(self.directory + "data/")
        else:
            if not os.path.exists(self.directory):
                print('Creating directory: ', self.directory)
                os.makedirs(self.directory)
            if not os.path.exists(self.directory + "data/"):
                print('Creating directory: ', self.directory + "data/")
                os.makedirs(self.directory + "data/")

        self.q_main_model_1 = QModel(obs_dim=self.observation_size, act_dim=self.action_size,
                                     learning_rate=learning_rate,
                                     name='q_main_model_1',
                                     directory=self.directory,
                                     save_frequency=self.save_frequency,
                                     clipped_double_q=self.clipped_double_q,
                                     **nafnet_kwargs)
        # Create replay buffer
        self.q_main_model_1.create_buffers()

        # Set same initial values in all networks
        self.q_target_model_1 = QModel(obs_dim=self.observation_size, act_dim=self.action_size,
                                       name='q_target_model_1',
                                       directory=self.directory,
                                       **nafnet_kwargs)
        self.q_target_model_1.q_model.set_weights(weights=self.q_main_model_1.q_model.get_weights())

        if self.clipped_double_q:
            self.q_main_model_2 = QModel(obs_dim=self.observation_size, act_dim=self.action_size,
                                         learning_rate=learning_rate,
                                         name='q_main_model_2',
                                         directory=self.directory,
                                         save_frequency=self.save_frequency,
                                         clipped_double_q=self.clipped_double_q,
                                         **nafnet_kwargs)
            # Copy buffer from first model
            self.q_main_model_2.create_buffers(buffer=self.q_main_model_1.replay_buffer)

            self.q_target_model_2 = QModel(obs_dim=self.observation_size, act_dim=self.action_size,
                                           name='q_target_model_2',
                                           directory=self.directory,
                                           **nafnet_kwargs)
            self.q_target_model_2.q_model.set_weights(weights=self.q_main_model_2.q_model.get_weights())

            # Set the target models
            self.q_main_model_1.set_target_models(self.q_target_model_1, self.q_target_model_2)
            self.q_main_model_2.set_target_models(self.q_target_model_2, self.q_target_model_1)
        else:
            self.q_main_model_1.set_target_models(self.q_target_model_1)

        self.counter = 0

    def predict(self, model, state, is_train):

        if is_train and model.replay_buffer.size < self.warm_up_steps:
            action = np.random.uniform(-1, 1, self.action_size)
            return np.array(action)

        # Add small noise on the controller
        elif is_train:
            action = self.noise_function(np.squeeze(model.get_action([state])),self.idx_episode)
            if self.q_smoothing_clip is None:
                return_value = np.clip(action, -1, 1)
            else:
                return_value = np.clip(action + np.clip(self.sigma * np.random.randn(self.action_size),
                                                        -self.q_smoothing_clip, self.q_smoothing_clip), -1, 1)
            return return_value
        else:
            action = model.get_action([state])
            return action

    def verification(self, **kwargs):
        print('Verification phase')
        if 'environment' in kwargs:
            self.env = kwargs.get('environment')
        if 'max_episodes' in kwargs:
            self.max_episodes = kwargs.get('max_episodes')
        if 'max_steps' in kwargs:
            self.max_steps = kwargs.get('max_steps')
        self.run(is_train=False)
        self.q_main_model_1.replay_buffer.save_to_pkl(name='buffer_data_verification.pkl', directory=self.directory)
        print('Saving verification buffer...')

    def training(self, **kwargs):
        print('Training phase')
        if 'warm_up_steps' in kwargs:
            self.warm_up_steps = kwargs.get('warm_up_steps')
        else:
            self.warm_up_steps = 0

        if 'initial_episode_length' in kwargs:
            self.initial_episode_length = kwargs.get('initial_episode_length')
        else:
            self.initial_episode_length = 5
        if 'environment' in kwargs:
            self.env = kwargs.get('environment')
        if 'max_episodes' in kwargs:
            self.max_episodes = kwargs.get('max_episodes')
        if 'max_steps' in kwargs:
            self.max_steps = kwargs.get('max_steps')
        self.q_main_model_1.set_training_parameters(**self.training_info)
        if self.clipped_double_q:
            self.q_main_model_2.set_training_parameters(**self.training_info)

        self.run(is_train=True)

    def run(self, is_train=True):
        for index in tqdm(range(0, self.max_episodes)):
            self.idx_episode = index
            # self.visualize(f'index: {index}')

            o = self.env.reset()
            for t in range(0, self.max_steps):
                # 1. predict

                a_1 = np.squeeze(self.predict(self.q_main_model_1, o, is_train))
                # Double Q implementation
                # a_2 = np.squeeze(self.predict(self.q_main_model_2, o, is_train))
                # a = (a_1 + a_2) / 2

                a = a_1
                o2, r, d, _ = self.env.step(a)
                if is_train:
                    self.q_main_model_1.replay_buffer.store(o, a, r, o2, d)
                o = o2
                d = False if t == self.max_steps - 1 else d

                if t > 0 and t % self.initial_episode_length == 0 and \
                        self.q_main_model_1.replay_buffer.size <= self.warm_up_steps:
                    o = self.env.reset()
                    print('Initial reset at ', t)

                # 2. train maybe not every step
                if t % 1 == 0:
                    if is_train and self.q_main_model_1.replay_buffer.size > self.warm_up_steps:
                        self.update_q(self.q_main_model_1)
                        if self.clipped_double_q:
                            self.update_q(self.q_main_model_2)
                    # Double Q implementation
                    # if is_train and self.q_main_model_1.replay_buffer.size > self.warm_up_steps:
                    #     # try:
                    #     if np.random.uniform(-1, 1, 1) < 0:
                    #         self.update_q(self.q_main_model_1)
                    #     else:
                    #         self.update_q(self.q_main_model_2)
                if d:
                    break

    def train_model(self, model, **kwargs):
        # Generate batch for monitoring the performance
        v = self.q_target_model_1.value_model(model.replay_buffer.sample_batch(20)['obs2'])
        loss = model.train_model(**kwargs)[-1]
        return v, loss

    def update_q(self, model, **kwargs):
        vs = []
        losses = []
        self.counter += 1
        v, loss = self.train_model(model=model, **kwargs)
        if model == self.q_main_model_1:
            vs.append(v)
            losses.append(loss)

            if self.counter % self.save_frequency == 0:
                number = str(self.counter).zfill(4)
                self.q_main_model_1.replay_buffer.save_to_pkl(name=f'buffer_data_' + number + '.pkl',
                                                              directory=self.directory + "data/")
                print('Saving buffer...')
            self.vs.append(np.mean(vs))
            self.losses.append(np.mean(losses))

    def visualize(self, label=None, **kwargs):
        # action = [np.zeros(self.env.action_space.shape)]
        state = np.zeros(self.env.observation_space.shape)

        delta = 0.05
        theta = np.arange(-1, 1, delta)
        theta_dot = np.arange(-1, 1, delta)
        X, Y = np.meshgrid(theta, theta_dot)

        Nr = 1
        Nc = 2
        fig, axs = plt.subplots(Nr, Nc)
        fig.subplots_adjust(hspace=0.3)

        rewards = np.zeros(X.shape)
        actions = np.zeros(X.shape)
        for i1 in range(len(theta)):
            for j1 in range(len(theta_dot)):
                state[0] = np.sin(theta[i1])
                state[1] = np.cos(theta[i1])
                state[2] = theta_dot[j1]

            rewards[i1, j1] = self.q_target_model_1.get_value_estimate([state])
            actions[i1, j1] = self.q_target_model_1.get_action([state])

        axs[0].contour(X, Y, rewards, alpha=1)
        axs[0].set_title('Value estimate')

        axs[1].contour(X, Y, actions, alpha=1)
        axs[0].set_title('Policy estimate')

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

        # plt.title(maximum)
        # plt.title(label)
        # plt.colorbar()
        fig.show()
        # else:
        #     pass
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


if __name__ == '__main__':
    print('start')
    # test_state = np.random.random((1, 2))
    #
    # q_main_model = QModel(2, 2)
    # q_target_model = QModel(2, 2)
    #
    # print('main', q_main_model.get_action(test_state))
    # print('main', q_main_model.get_value_estimate(test_state))
    #
    # print('target', q_target_model.get_action(test_state))
    # print('target', q_target_model.get_value_estimate(test_state))
    #
    # q_target_model.set_weights(q_main_model.get_weights())
    #
    # print('target', q_target_model.get_action(test_state))
    # print('target', q_target_model.get_value_estimate(test_state))
    #
    # batch_x = np.random.random((5, 4))
    # batch_y = np.random.random((5, 4))
    # hist = q_main_model.q_model.fit(batch_x, batch_y)
    # print(hist.history['loss'])
    #
    # print('main', q_main_model.get_action(test_state))
    # print('main', q_main_model.get_value_estimate(test_state))
    #
    # print('target', q_target_model.get_action(test_state))
    # print('target', q_target_model.get_value_estimate(test_state))
    #
    #
    # q_target_model.set_weights(q_main_model.get_weights())
    #
    # print('target', q_target_model.get_action(test_state))
    # print('target', q_target_model.get_value_estimate(test_state))
    #
    # weights = (q_target_model.get_weights())
    # keras.utils.plot_model(model, 'my_first_model.png')
    # keras.utils.plot_model(model_get_action, 'model_get_action.png')
