import os
import pickle
import shutil
import time

import tensorflow as tf
from tensorflow import keras

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


class QModel:

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

        if 'clipped_double_q' in kwargs:
            self.clipped_double_q = kwargs.get('clipped_double_q')
        else:
            self.clipped_double_q = False
            # print(self.__name__ )
        if 'kernel_initializer' in kwargs:
            self.kernel_initializer = kwargs.get('kernel_initializer')
        else:
            self.kernel_initializer = tf.compat.v1.random_uniform_initializer(-0.01, 0.01)

        self.init = True

        self.act_dim = act_dim
        self.obs_dim = obs_dim

        # create a shared network for the variables
        inputs_state = keras.Input(shape=(self.obs_dim,), name="state_input")
        inputs_action = keras.Input(shape=(self.act_dim,), name="action_input")

        # h = inputs[:, 0:obs_dim]
        h = inputs_state
        for hidden_dim in self.hidden_sizes:
            h = self.fc(h, hidden_dim, kernel_initializer=self.kernel_initializer)
        V = self.fc(h, 1, activation=None, kernel_initializer=self.kernel_initializer, name='V')

        l = self.fc(h, (self.act_dim * (self.act_dim + 1) / 2),
                    kernel_initializer=self.kernel_initializer, name='l')
        mu = self.fc(h, self.act_dim, kernel_initializer=self.kernel_initializer, name='mu')

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
        A = -tf.multiply(tf.matmul(tf.transpose(a=tmp, perm=[0, 2, 1]),
                                   tf.matmul(P, tmp)), tf.constant(0.5, dtype=tf.float64))
        A = tf.reshape(A, [-1, 1])
        Q = tf.add(A, V)

        if 'learning_rate' in kwargs:
            self.learning_rate = kwargs.get('learning_rate')
            del kwargs['learning_rate']
        else:
            self.learning_rate = 1e-3
        if 'directory' in kwargs:
            self.directory = kwargs.get('directory')
        else:
            self.directory = None

        if 'discount' in kwargs:
            self.discount = tf.constant(kwargs.get('discount'), dtype=tf.float64)
            del kwargs['discount']
        else:
            self.discount = tf.constant(0.999, dtype=tf.float64)

        self.optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)

        self.q_model = self.CustomModel(inputs=[inputs_state, inputs_action], outputs=Q, mother_class=self)

        self.q_model.compile(optimizer=self.optimizer, loss="mse", metrics=["mae"])

        # Action output
        self.model_get_action = keras.Model(inputs=[inputs_state, inputs_action],
                                            outputs=self.q_model.get_layer(name='mu').output)

        # Value output
        self.model_value_estimate = keras.Model(inputs=[inputs_state, inputs_action],
                                                outputs=self.q_model.get_layer(name='V').output)

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
        state = np.array([state], dtype='float64')
        actions = tf.zeros(shape=(tf.shape(state)[0], self.act_dim), dtype=tf.float64)
        return self.model_get_action.predict([state, actions])

    def get_value_estimate(self, state):
        actions = tf.zeros(shape=(tf.shape(state)[0], self.act_dim), dtype=tf.float64)
        return self.model_value_estimate.predict([state, actions])

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
            self.q_target.set_polyak_weights(self.model.get_weights(),
                                             polyak=0.999, name=self.model.__name__)

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

    def train_model(self, **kwargs):
        if 'polyak' in kwargs:
            self.polyak = kwargs.get('polyak')
            del kwargs['polyak']
        else:
            self.polyak = 0.999
        if 'batch_size' in kwargs:
            batch_size = kwargs.get('batch_size')
            del kwargs['batch_size']
        else:
            batch_size = 100
        if 'epochs' not in kwargs:
            kwargs['epochs'] = 1
        if 'steps_per_batch' in kwargs:
            steps_per_batch = kwargs.get('steps_per_batch')
            del kwargs['steps_per_batch']
        else:
            steps_per_batch = 1
        if 'discount' in kwargs:
            del kwargs['discount']


        batch = self.replay_buffer.sample_batch(batch_size=batch_size)

        # Here we decide how often to iterate over the data
        dataset = tf.data.Dataset.from_tensor_slices(batch)  # .repeat(1).shuffle(buffer_size=10000)
        train_dataset = dataset.batch(steps_per_batch)
        self.callback = self.CustomCallback(patience=0)
        self.callback.q_target = self.q_target_first

        hist = self.q_model.fit(train_dataset,
                                verbose=0,
                                callbacks=[self.callback],
                                shuffle=True,
                                **kwargs)
        if int(self.ckpt.step) % self.save_frequency == 0:
            save_path = self.manager.save()
            print("Saved checkpoint for step {}: {}".format(int(self.ckpt.step), save_path))
            save_path_target = self.q_target_first.manager.save()
            print("Saved checkpoint for step {}: {}".format(int(self.ckpt.step), save_path_target))
        self.ckpt.step.assign_add(1)
        return_value = hist.history['loss']

        return return_value

    def set_models(self, q_target_1, q_target_2=None):
        self.q_target_first = q_target_1
        if q_target_2 is not None:
            self.q_target_second = q_target_2

    class CustomModel(keras.Model):

        def __init__(self, *args, **kwargs):
            self.mother_class = kwargs.get('mother_class')
            self.__name__ = self.mother_class.__name__
            self.discount = self.mother_class.discount
            del kwargs['mother_class']
            super().__init__(*args, **kwargs)

        def train_step(self, batch):
            o = batch['obs1']
            o2 = batch['obs2']
            a = batch['acts']
            r = batch['rews']
            d = batch['done']
            v_1 = self.mother_class.q_target_first.model_value_estimate([o2, a])  # , training=False)
            if self.mother_class.clipped_double_q:
                v_2 = self.mother_class.q_target_second.model_value_estimate([o2, a])  # , training=False)
                v = tf.squeeze(tf.where(tf.math.less(v_1, v_2), v_1, v_2))
            else:
                v = tf.squeeze(v_1)
            y_target = tf.add(tf.multiply(tf.math.scalar_mul(self.discount, v),
                                          tf.add(tf.constant(1, dtype=tf.float64),
                                                 tf.math.scalar_mul(-1, d))), r)

            with tf.GradientTape() as tape:
                # Run the forward pass of the layer.
                # The operations that the layer applies
                # to its inputs are going to be recorded
                # on the GradientTape.
                y_pred = self([o, a], training=True)  # Logits for this minibatch
                # Compute the loss value for this minibatch.
                loss = self.compiled_loss(
                    y_target,
                    y_pred,
                )
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
            return {m.name: m.result() for m in self.metrics}

    def create_buffers(self):
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

class NAF(object):
    def __init__(self, env, training_info=dict(), pretune=None,
                 noise_info=dict(), save_frequency=500, directory=None, is_continued=False,
                 clipped_double_q=2, q_smoothing=0.01, **nafnet_kwargs):
        '''
        :param env: open gym environment to be solved
        :param directory: directory were weigths are saved
        :param stat: statistic class to handle tensorflow and statitics
        :param discount: discount factor
        :param batch_size: batch size for the training
        :param learning_rate: learning rate
        :param max_steps: maximal steps per episode
        :param update_repeat: iteration per step of training
        :param max_episodes: maximum number of episodes
        :param polyak: polyac averaging
        :param pretune: list of tuples of state action reward next state done
        :param nafnet_kwargs: keywords to handle the network and training
        :param noise_info: dict with noise_function
        :param clipped_double_q: use the clipped double q trick with switching all clipped_double_q steps
        :param q_smoothing: add small noise on actions to smooth the training
        '''
        self.rewards = []
        self.states = []
        self.actions = []
        self.dones = []

        self.clipped_double_q = clipped_double_q
        self.q_smoothing = q_smoothing
        self.losses2 = []
        self.vs2 = []
        self.model_switch = 1

        self.directory = directory
        self.save_frequency = save_frequency

        self.losses = []
        self.pretune = pretune

        self.env = env

        if 'noise_function' in noise_info:
            self.noise_function = noise_info.get('noise_function')
        else:
            self.noise_function = lambda nr: 1 / (nr + 1)

        self.action_size = env.action_space.shape[0]
        self.observation_size = env.observation_space.shape[0]

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
            self.q_main_model_2.create_buffers()
            self.q_target_model_2 = QModel(obs_dim=self.observation_size, act_dim=self.action_size,
                                           name='q_target_model_2',
                                           directory=self.directory,
                                           **nafnet_kwargs)
            self.q_target_model_2.q_model.set_weights(weights=self.q_main_model_2.q_model.get_weights())

            self.q_main_model_1.set_models(self.q_target_model_1, self.q_target_model_2)
            self.q_main_model_2.set_models(self.q_target_model_2, self.q_target_model_1)
        else:
            self.q_main_model_1.set_models(self.q_target_model_1)

        self.counter = 0

    def predict(self, model, state, is_train):

        if is_train and model.replay_buffer.size < self.warm_up_steps:
            action = np.random.uniform(-1, 1, self.action_size)
            return np.array(action)

        # Add small noise on the controller
        elif is_train:
            action = model.get_action(state=state)
            noise = self.noise_function(self.idx_episode) * np.random.randn(self.action_size)
            if self.q_smoothing is None:
                return_value = np.clip(action + noise, -1, 1)
            else:
                sigma = 0.01
                return_value = np.clip(action + noise + np.clip(sigma * np.random.randn(
                    self.action_size), -self.q_smoothing, self.q_smoothing), -1, 1)
            return np.array(return_value)
        else:
            action = model.get_action(state=state)
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

        self.run(is_train=True)

    def add_trajectory_data(self, state, action, reward, done):
        index = self.idx_episode
        self.rewards[index].append(reward)
        self.actions[index].append(action)
        self.states[index].append(state)
        self.dones[index].append(done)

    def store_trajectories_to_pkl(self, name, directory):
        out_put_writer = open(directory + name, 'wb')
        pickle.dump(self.states, out_put_writer, -1)
        pickle.dump(self.actions, out_put_writer, -1)
        pickle.dump(self.rewards, out_put_writer, -1)
        pickle.dump(self.dones, out_put_writer, -1)
        out_put_writer.close()

    def init_trajectory_data(self, state):
        self.rewards.append([])
        self.actions.append([])
        self.states.append([])
        self.dones.append([])
        self.add_trajectory_data(state=state, action=None, done=None, reward=None)

    def run(self, is_train=True):
        for index in tqdm(range(0, self.max_episodes)):
            self.idx_episode = index

            o = self.env.reset()
            # For the trajectory storage
            self.init_trajectory_data(state=o)

            for t in range(0, self.max_steps):
                # 1. predict
                a = np.squeeze(self.predict(self.q_main_model_1, o, is_train))
                o2, r, d, _ = self.env.step(a)
                self.add_trajectory_data(state=o2, action=a, done=d, reward=r)
                if is_train:
                    self.q_main_model_1.replay_buffer.store(o, a, r, o2, d)
                    if self.clipped_double_q:
                        self.q_main_model_2.replay_buffer.store(o, a, r, o2, d)
                o = o2
                d = False if t == self.max_steps - 1 else d

                if t > 0 and t % self.initial_episode_length == 0 and \
                        self.q_main_model_1.replay_buffer.size <= self.warm_up_steps:
                    o = self.env.reset()
                    self.init_trajectory_data(state=o)
                    print('Initial reset at ', t)

                # 2. train maybe not every step
                if t % 1 == 0:
                    if is_train and self.q_main_model_1.replay_buffer.size > self.warm_up_steps:
                        # try:
                        self.update_q(self.q_main_model_1)
                        if self.clipped_double_q:
                            self.update_q(self.q_main_model_2)
                if d:
                    break

    def train_model(self, model, **kwargs):
        # Generate batch for monitoring the performance
        batch = model.replay_buffer.sample_batch(200)

        o2 = batch['obs2']
        a = batch['acts']
        v = self.q_target_model_1.model_value_estimate([o2, a])
        loss = model.train_model(**self.training_info)[-1]
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
                self.store_trajectories_to_pkl(name=f'trajectory_data_' + number + '.pkl',
                                               directory=self.directory + "data/")
                print('Saving buffer...')
            self.vs.append(np.mean(vs))
            self.losses.append(np.mean(losses))


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