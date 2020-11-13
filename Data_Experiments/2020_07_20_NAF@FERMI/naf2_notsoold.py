import os
import pickle
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
        obs1s,obs2s,acts,rews,dones = [buffer_data[key] for key in buffer_data]
        for i in range(len(obs1s)):
            self.store(obs1s[i], acts[i], rews[i], obs2s[i], dones[i])
        # print(self.size)



# class ReplayBufferPER(PrioritizedReplayBuffer):
#     """
#     A simple FIFO experience replay buffer for NAF_debug agents.
#     """
#
#     def __init__(self, obs_dim, act_dim, size, prio_info):
#         self.alpha = prio_info.get('alpha')
#         self.beta = prio_info.get('beta')
#         super(ReplayBufferPER, self).__init__(size, self.alpha)
#         self.ptr, self.size, self.max_size = 0, 0, size
#
#     def store(self, obs, act, rew, next_obs, done):
#         super(ReplayBufferPER, self).add(obs, act, rew, next_obs, done, 1)
#         self.ptr = (self.ptr + 1) % self.max_size
#         self.size = min(self.size + 1, self.max_size)
#
#     def sample_normal(self, batch_size):
#         if self.size < batch_size:
#             batch_size = self.size
#         obs1, acts, rews, obs2, done, gammas, weights, idxs = super(ReplayBufferPER, self).sample_normal_rand(
#             batch_size)
#         return dict(obs1=obs1,
#                     obs2=obs2,
#                     acts=acts,
#                     rews=rews,
#                     done=done), [weights, idxs]
#
#     def sample_batch(self, batch_size=32, **kwargs):
#         if 'beta' in kwargs:
#             self.beta = kwargs.get('beta')
#         if self.size < batch_size:
#             batch_size = self.size
#             obs1, acts, rews, obs2, done, gammas, weights, idxs = super(ReplayBufferPER, self).sample_normal_rand(
#                 batch_size)
#         else:
#             obs1, acts, rews, obs2, done, gammas, weights, idxs = super(ReplayBufferPER, self).sample(batch_size,
#                                                                                                       self.beta)
#         return dict(obs1=obs1,
#                     obs2=obs2,
#                     acts=acts,
#                     rews=rews,
#                     done=done), [weights, idxs]


def basic_loss_function(y_true, y_pred):
    return tf.math.reduce_mean(y_true - y_pred)


# obs_dim = 2
# act_dim = 2
# action = tf.Variable(np.ones(act_dim), dtype=float)
hidden_sizes = (100, 100)


class QModel:

    def __init__(self, obs_box=2, act_box=2, **kwargs):
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

        self.init = True

        self.act_box = act_box
        self.obs_box = obs_box
        self.act_dim = act_box.shape[0]
        self.obs_dim = obs_box.shape[0]

        # create a shared network for the variables
        inputs_state = keras.Input(shape=(self.obs_dim,), name="state_input")
        inputs_action = keras.Input(shape=(self.act_dim,), name="action_input")

        # h = inputs[:, 0:obs_dim]
        h = self.normalize(inputs_state, box=self.obs_box)
        for hidden_dim in hidden_sizes:
            h = self.fc(h, hidden_dim)
        V = self.fc(h, 1, name='V')

        l = self.fc(h, (self.act_dim * (self.act_dim + 1) / 2))
        mu = self.fc(h, self.act_dim, name='mu')

        # action = inputs[:, obs_dim:]
        action = self.normalize(inputs_action, box=self.act_box)
        # rescale action to tanh

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
        tmp = tf.expand_dims(action - mu, -1)
        A = -tf.multiply(tf.matmul(tf.transpose(a=tmp, perm=[0, 2, 1]),
                                   tf.matmul(P, tmp)), tf.constant(0.5, dtype=tf.float64))
        A = tf.reshape(A, [-1, 1])
        Q = tf.add(A, V)

        if 'learning_rate' in kwargs:
            self.learning_rate = kwargs.get('learning_rate')
            del kwargs['learning_rate']
        else:
            self.learning_rate = 1e-3
        # print('learning rate', self.learning_rate )
        if 'directory' in kwargs:
            self.directory = kwargs.get('directory')
        else:
            self.directory = None

        initial_learning_rate = 0.005
        lr_schedule = keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate, decay_steps=1000, decay_rate=0.99, staircase=True
        )
        lr_schedule = self.learning_rate
        self.optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)

        self.q_model = self.CustomModel(inputs=[inputs_state, inputs_action], outputs=Q, mother_class=self)
        # self.q_model.compile(keras.optimizers.Adam(learning_rate=self.learning_rate), loss=MSE)
        # optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.q_model.compile(optimizer=self.optimizer, loss="mse", metrics=["mae"])

        # Action output
        # self.model_get_action = keras.Model(inputs=self.q_model.layers[0].input,
        #                                     outputs=self.q_model.get_layer(name='mu').output)
        self.model_get_action = keras.Model(inputs=[inputs_state, inputs_action],
                                            outputs=self.q_model.get_layer(name='mu').output)

        # Value output
        self.model_value_estimate = keras.Model(inputs=[inputs_state, inputs_action],
                                                outputs=self.q_model.get_layer(name='V').output)

        self.storage_management()
        # self.q_model.summary()

    def storage_management(self):
        checkpoint_dir = self.directory + self.__name__ + "/"
        self.ckpt = tf.train.Checkpoint(step=tf.Variable(1), model=self.q_model)
        self.manager = tf.train.CheckpointManager(self.ckpt, checkpoint_dir, max_to_keep=3)
        self.ckpt.restore(self.manager.latest_checkpoint)
        if self.manager.latest_checkpoint:
            print("Restored from {}".format(self.manager.latest_checkpoint))
        else:
            print("Initializing from scratch.")

    def normalize(self, input, box):
        low = tf.convert_to_tensor(box.low, dtype=tf.float64)
        high = tf.convert_to_tensor(box.high, dtype=tf.float64)
        return tf.math.scalar_mul(tf.convert_to_tensor(2, dtype=tf.float64),
                                  tf.math.add(tf.convert_to_tensor(-0.5, dtype=tf.float64),
                                              tf.multiply(tf.math.add(input, -low), 1 / (high - low))))

    def de_normalize(self, input, box):
        low = tf.convert_to_tensor(box.low, dtype=tf.float64)
        high = tf.convert_to_tensor(box.high, dtype=tf.float64)
        input = tf.convert_to_tensor(input, dtype=tf.float64)
        return tf.math.add(
            tf.multiply(tf.math.add(tf.math.scalar_mul(tf.convert_to_tensor(0.5, dtype=tf.float64), input),
                                    tf.convert_to_tensor(0.5, dtype=tf.float64)),
                        (high - low)), low)

    def fc(self, x, hidden_size, name=None):
        layer = keras.layers.Dense(hidden_size, activation=tf.nn.tanh,
                                   kernel_initializer=tf.compat.v1.random_uniform_initializer(-0.01, 0.01),
                                   kernel_regularizer=None,
                                   bias_initializer=tf.compat.v1.constant_initializer(0.0), name=name)
        return layer(x)

    def get_action(self, state):
        state = np.array([state], dtype='float64')
        actions = tf.zeros(shape=(tf.shape(state)[0], self.act_dim), dtype=tf.float64)
        return self.de_normalize(self.model_get_action.predict([state, actions]), self.act_box)

    def get_value_estimate(self, state):
        actions = tf.zeros(shape=(tf.shape(state)[0], self.act_dim), dtype=tf.float64)
        return self.model_value_estimate.predict([state, actions])

    def set_polyak_weights(self, weights, polyak=0.999, **kwargs):
        if 'name' in kwargs:
            print(10 * ' updating:', kwargs.get('name'))
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

    # def train_model(self, batch_s, batch_a, batch_y, **kwargs):
    #     # batch_x = np.concatenate((batch_s, batch_a), axis=1)
    #     n_split = int(5 * len(batch_s) / 5)
    #     batch_s_train, batch_a_train, batch_y_train = batch_s[:n_split], batch_a[:n_split], batch_y[:n_split]
    #     batch_s_val, batch_a_val, batch_y_val = batch_s[n_split:], batch_a[n_split:], batch_y[n_split:]
    #     x_batch_train = tf.keras.layers.concatenate([batch_s_train, batch_a_train],
    #                                                 axis=1, dtype=tf.float64)
    #     y_batch_train = tf.convert_to_tensor(batch_y_train, dtype=tf.float64)
    #
    #     train_dataset = tf.data.Dataset.from_tensor_slices((x_batch_train, y_batch_train))
    #     train_dataset = train_dataset.repeat(50).shuffle(buffer_size=1024, reshuffle_each_iteration=True).batch(15)
    #
    #     # x_batch_train = tf.keras.layers.concatenate([batch_s_val, batch_a_val],
    #     #                                             axis=1, dtype=tf.float64)
    #     # y_batch_train = tf.convert_to_tensor(batch_y_val, dtype=tf.float64)
    #
    #     # val_dataset = tf.data.Dataset.from_tensor_slices((x_batch_train, y_batch_train))
    #     # val_dataset = val_dataset.repeat(50).shuffle(buffer_size=1024, reshuffle_each_iteration=True).batch(10)
    #
    #     # if x_batch_train.shape[0]<50 else 25
    #     epochs = 6
    #     self.callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2)
    #     batch_size = 10  # x_batch_train.shape[0]
    #     hist = self.q_model.fit(x_batch_train, y_batch_train,
    #                             validation_split=0.1,
    #                             steps_per_epoch=2,
    #                             verbose=0,
    #                             batch_size=batch_size,
    #                             callbacks=[self.callback],
    #                             shuffle=True,
    #                             epochs=epochs,
    #                             # validation_data=val_dataset,
    #                             # validation_steps=3,
    #                             **kwargs)
    #     return_value = hist.history['loss']
    #     return return_value

    # def train_model(self, batch_s, batch_a, y_batch_train, **kwargs):
    #     # x_batch_train = np.concatenate((batch_s, batch_a), axis=1)
    #     x_batch_train = tf.keras.layers.concatenate([batch_s, batch_a],
    #                                                 axis=1, dtype=tf.float64)
    #     y_batch_train = tf.convert_to_tensor(y_batch_train, dtype=tf.float64)
    #
    #     return self.train_step(x_batch_train, y_batch_train, **kwargs)

    # @tf.function(experimental_relax_shapes=True)

    class CustomCallback(keras.callbacks.Callback):

        def __init__(self, patience=0):
            # super(self.CustomCallback, self).__init__()
            super().__init__()
            self.patience = patience
            # best_weights to store the weights at which the minimum loss occurs.
            self.best_weights = None

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
                                             polyak=0.9995)
            # print('updating...', self.model.__name__)

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

    # def get_estimate(self, o2, d, r):
    #     self.discount = 0.999
    #     v_1 = self.q_target_first.get_value_estimate(o2)
    #     v_2 = self.q_target_second.get_value_estimate(o2)
    #     v = tf.where(v_1 < v_2, v_1, v_2)
    #     return self.discount * tf.squeeze(v) * (1 - d) + r

    def train_model(self, **kwargs):
        if 'polyak' in kwargs:
            self.polyak = kwargs.get('polyak')
            del kwargs['polyak']
        else:
            self.polyak = 0.999
        if 'batch_size' in kwargs:
            self.batch_size = kwargs.get('batch_size')
        else:
            self.batch_size = 100
        if 'epochs' not in kwargs:
            # self.epochs = kwargs.get('epochs')
            kwargs['epochs'] = 2
        if 'steps_per_epoch' not in kwargs:
            # self.steps_per_epoch = kwargs.get('steps_per_epoch')
            kwargs['steps_per_epoch'] = 10
        # else:
        #     self.steps_per_epoch = 10
        if 'discount' in kwargs:
            self.polyak = kwargs.get('discount')
            del kwargs['discount']
        else:
            self.polyak = 0.999
        batch = self.replay_buffer.sample_batch(batch_size=1000000)
        # batch, prios = self.replay_buffer.sample_batch(batch_size=batch_size)
        # nr = self.replay_buffer.size
        #
        # beta = lambda nr: max(1e-16, 1 - nr / 1000)
        # decay_function = lambda nr: max(0, 1 - nr / 1000)
        # beta_decay = beta(nr)
        # print(beta_decay)
        # batch, priority_info = self.replay_buffer.sample_batch(batch_size=30, beta=beta_decay)
        # sample_weights = priority_info[0].astype('float64')
        # batch['sample_weights'] = sample_weights
        #
        batch['obs1'] = batch['obs1'].astype('float64')
        batch['obs2'] = batch['obs2'].astype('float64')
        batch['acts'] = batch['acts'].astype('float64')
        batch['rews'] = batch['rews'].astype('float64')
        batch['done'] = np.where(batch['done'], 1, 0).astype('float64')

        # batch['y_target'] = self.get_estimate(o2, d, r)
        # batch['x_batch_train'] = x_batch_train
        # print(batch)
        dataset = tf.data.Dataset.from_tensor_slices(batch).repeat(200).shuffle(buffer_size=10000)
        train_dataset = dataset.batch(self.batch_size)
        # print([element['obs1'] for element in train_dataset.take(2)])

        # val_dataset = dataset.take(10)

        # if False:
        #     # if self.replay_buffer.size % 50 == 0 or self.init:
        #     epochs = 50
        #     dataset = tf.data.Dataset.from_tensor_slices(batch).shuffle(buffer_size=1024)
        #     train_dataset = dataset.batch(10)
        #     self.callback = self.CustomCallback(patience=0)
        #     self.callback.q_target = self.q_target_first
        #
        #     # self.callback.q_model = self.q_model
        #     early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=2, verbose=1)
        #
        #     hist = self.q_model.fit(train_dataset,
        #                             # validation_split=0.1,
        #                             verbose=1,
        #                             # batch_size=batch_size,
        #                             callbacks=[self.callback, early_stop],
        #                             shuffle=True,
        #                             epochs=epochs,
        #                             # validation_data=val_dataset,
        #                             # validation_steps=2,
        #                             **kwargs)
        #     self.init = False
        # else:
        #     # epochs = 2
        self.callback = self.CustomCallback(patience=0)
        self.callback.q_target = self.q_target_first
        # self.save_frequency = 5

        # checkpoint_callback = [
        #     keras.callbacks.ModelCheckpoint(
        #         # Path where to save the model
        #         filepath=self.directory+self.__name__ +"/mymodel.tf",
        #         save_weights_only=True,
        #         save_freq=self.save_frequency,
        #         # save_best_only=True,  # Only save a model if `val_loss` has improved.
        #         # monitor="loss",
        #         verbose=1,
        #     )
        # ]
        # self.callback.q_model = self.q_model
        # early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, verbose=0)
        # TODO: implement saving
        hist = self.q_model.fit(train_dataset,
                                # sample_weights=sample_weights,
                                # validation_split=0.1,
                                # steps_per_epoch=self.steps_per_epoch,
                                verbose=0,
                                # batch_size=batch_size,
                                callbacks=[self.callback],# , checkpoint_callback],
                                shuffle=True,
                                # epochs=self.epochs,
                                # validation_data=val_dataset,
                                # validation_steps=2,
                                **kwargs)
        # update the targets
        # self.q_target_first.set_polyak_weights(self.q_model.get_weights(),
        #                                  polyak=0.999)

        # loss = model.train_model(o, a, target_y, sample_weight=sample_weights)[-1]
        if int(self.ckpt.step) % self.save_frequency == 0:
            save_path = self.manager.save()
            print("Saved checkpoint for step {}: {}".format(int(self.ckpt.step), save_path))
            save_path_target = self.q_target_first.manager.save()
            print("Saved checkpoint for step {}: {}".format(int(self.ckpt.step), save_path_target))
        self.ckpt.step.assign_add(1)
        return_value = hist.history['loss']
        # decay = decay_function(nr)
        # update_prios = (return_value[-1] * decay + 1e-16) * np.ones(priority_info[0].shape)
        # self.replay_buffer.update_priorities(idxes=priority_info[1], priorities=update_prios)

        return return_value

    def set_models(self, q_target_1, q_target_2=None):
        self.q_target_first = q_target_1
        if q_target_2 is not None:
            self.q_target_second = q_target_2

    class CustomModel(keras.Model):

        def __init__(self, *args, **kwargs):
            self.mother_class = kwargs.get('mother_class')
            self.__name__ = self.mother_class.__name__
            del kwargs['mother_class']
            super().__init__(*args, **kwargs)
            if 'discount' in kwargs:
                self.discount = tf.constant(kwargs.get('discount'), dtype=tf.float64)
                del kwargs['discount']
            else:
                self.discount = tf.constant(0.999, dtype=tf.float64)

        def train_step(self, batch):
            o = batch['obs1']
            o2 = batch['obs2']
            a = batch['acts']
            r = batch['rews']
            d = batch['done']
            # target_y = self.mother_class.get_estimate(o2, d, r)
            # y_target = batch['y_target']

            # a_zero = tf.multiply(a, tf.constant(0, dtype=tf.float64))
            v_1 = self.mother_class.q_target_first.model_value_estimate([o2, a])  # , training=False)
            if self.mother_class.clipped_double_q:
                v_2 = self.mother_class.q_target_second.model_value_estimate([o2, a])  # , training=False)
                v = tf.squeeze(tf.where(tf.math.less(v_1, v_2), v_1, v_2))
                # print('double', self.mother_class.__name__)
            else:
                v = tf.squeeze(v_1)
                # print('single', self.mother_class.__name__)
            y_target = tf.add(tf.multiply(tf.math.scalar_mul(self.discount, v),
                                          tf.add(tf.constant(1, dtype=tf.float64),
                                                 tf.math.scalar_mul(-1, d))), r)

            # print('target', tf.reduce_mean(y_target))

            # Iterate over the batches of the dataset.
            # if 'sample_weight' in kwargs:
            #     sample_weight = kwargs.get('sample_weight')
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
                    # sample_weight=batch['sample_weights']
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

            # self.mother_class.q_target.set_polyak_weights(self.mother_class.q_model.get_weights(),
            # polyak=0.999)
            return {m.name: m.result() for m in self.metrics}

    def create_buffers(self, per_flag, prio_info):
        if not (per_flag):
            self.replay_buffer = ReplayBuffer(obs_dim=self.obs_dim, act_dim=self.act_dim, size=int(1e6))

        try:
            self.replay_buffer.read_from_pkl(name='buffer_data.pkl', directory=self.directory)
            print('Buffer data loaded for ' + self.__name__)
        except:
            print('Buffer data empty for ' + self.__name__)

        # else:
        #     self.replay_buffer = ReplayBufferPER(obs_dim=self.obs_dim, act_dim=self.act_dim, size=int(1e6),
        #                                          prio_info=prio_info)


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
        :param prio_info: parameters to handle the prioritizing of the buffer
        :param nafnet_kwargs: keywords to handle the network and training
        :param noise_info: dict with noise_function
        :param clipped_double_q: use the clipped double q trick with switching all clipped_double_q steps
        :param q_smoothing: add small noise on actions to smooth the training
        '''
        self.clipped_double_q = clipped_double_q
        self.q_smoothing = q_smoothing
        self.losses2 = []
        self.vs2 = []
        self.model_switch = 1

        self.directory = directory
        self.save_frequency = save_frequency

        self.losses = []
        self.pretune = pretune
        # self.prio_info = prio_info
        self.prio_info = dict()
        self.per_flag = bool(self.prio_info)
        # self.per_flag = False
        print('PER is:', self.per_flag)

        self.env = env

        if 'noise_function' in noise_info:
            self.noise_function = noise_info.get('noise_function')
        else:
            self.noise_function = lambda nr: 1 / (nr + 1)


        self.action_box = env.action_space
        self.action_size = self.action_box.shape[0]
        self.obs_box = env.observation_space

        self.max_steps = 1000
        # self.update_repeat = update_repeat

        self.idx_episode = None
        self.vs = []
        if 'decay_function' in self.prio_info:
            self.decay_function = self.prio_info.get('decay_function')
        else:
            if 'beta' in self.prio_info:
                self.decay_function = lambda nr: self.prio_info.get('beta')
            else:
                self.decay_function = lambda nr: 1.

        if 'beta_decay' in self.prio_info:
            self.beta_decay_function = self.prio_info.get('beta_decay')
        # elif self.per_flag:
        #     self.beta_decay_function = lambda nr: max(1e-12, prio_info.get('beta_start') - nr / 100)
        else:
            self.beta_decay_function = lambda nr: 1

        self.training_info = training_info

        if 'learning_rate' in training_info:
            learning_rate = training_info.get('learning_rate')
            del training_info['learning_rate']
        else:
            learning_rate = 1e-3

        self.q_main_model_1 = QModel(obs_box=self.obs_box, act_box=self.action_box, learning_rate=learning_rate,
                                     name='q_main_model_1',
                                     directory=self.directory,
                                     save_frequency=self.save_frequency,
                                     clipped_double_q=self.clipped_double_q,
                                     **nafnet_kwargs)
        self.q_main_model_1.create_buffers(per_flag=self.per_flag, prio_info=self.prio_info)

        # self.current_model = self.q_main_model_1
        # Set same initial values in all networks
        # self.q_main_model_2.q_model.set_weights(weights=self.q_main_model_1.q_model.get_weights())
        # Set same initial values in all networks
        self.q_target_model_1 = QModel(obs_box=self.obs_box, act_box=self.action_box,
                                       name='q_target_model_1',
                                       directory=self.directory,
                                       **nafnet_kwargs)
        self.q_target_model_1.q_model.set_weights(weights=self.q_main_model_1.q_model.get_weights())

        if self.clipped_double_q:
            self.q_main_model_2 = QModel(obs_box=self.obs_box, act_box=self.action_box, learning_rate=learning_rate,
                                         name='q_main_model_2',
                                         directory=self.directory,
                                         save_frequency=self.save_frequency,
                                         clipped_double_q=self.clipped_double_q,
                                         **nafnet_kwargs)
            self.q_main_model_2.create_buffers(per_flag=self.per_flag, prio_info=self.prio_info)
            self.q_target_model_2 = QModel(obs_box=self.obs_box, act_box=self.action_box,
                                           name='q_target_model_2',
                                           directory=self.directory,)
            self.q_target_model_2.q_model.set_weights(weights=self.q_main_model_2.q_model.get_weights())

            # TODO: change to one network
            # if self.clipped_double_q:
            self.q_main_model_1.set_models(self.q_target_model_1, self.q_target_model_2)
            self.q_main_model_2.set_models(self.q_target_model_2, self.q_target_model_1)
        else:
            self.q_main_model_1.set_models(self.q_target_model_1)

        if not(is_continued):
        #     try:
        #         print(self.directory)
        #         self.q_target_model_1.q_model = tf.keras.models.load_model(filepath=self.directory)
        #         print('Successfully loaded', 10 * ' -')
        #     except:
        #         print('Failed to load', 10 * ' *')
        #         if not os.path.exists(self.directory):
        #             os.makedirs(self.directory)
        # else:
            if not os.path.exists(self.directory):
                os.makedirs(self.directory)
            elif not (self.directory):
                for f in os.listdir(self.directory):
                    print('Deleting: ', self.directory + '/' + f)
                    os.remove(self.directory + '/' + f)
                time.sleep(.5)
        else:
            if not os.path.exists(self.directory):
                print('Creating directory: ', self.directory)
                os.makedirs(self.directory)

        self.counter = 0

    def predict(self, model, state, is_train):

        if is_train and model.replay_buffer.size < self.warm_up_steps:
            print(10 * 'inits ')
            action = model.de_normalize(np.random.uniform(-1, 1, self.action_size), model.act_box)
            # print(action)
            return np.array(action)
        elif is_train:
            action = model.normalize(model.get_action(state=state), model.act_box)
            noise = self.noise_function(self.idx_episode) * np.random.randn(self.action_size)
            if self.q_smoothing is None:
                return_value = model.de_normalize(np.clip(action + noise, -1, 1), model.act_box)
            else:
                return_value = model.de_normalize(np.clip(np.clip(action +
                                                                  noise, -1, 1) + np.clip(self.q_smoothing *
                                                                                          np.random.randn(
                                                                                              self.action_size),
                                                                                          -self.q_smoothing,
                                                                                          self.q_smoothing),
                                                          -1, 1),
                                                  model.act_box)
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

    def run(self, is_train=True):
        for index in tqdm(range(0, self.max_episodes)):
            self.idx_episode = index
            # if self.clipped_double_q is not None:
            #     self.model_switcher(self.idx_episode)

            o = self.env.reset()
            # if self.training_stop is not None:
            #     is_train = False if self.training_stop < self.idx_episode else True
                # print("starting the tests")
            for t in range(0, self.max_steps):
                # 1. predict
                # if self.q_smoothing is not None:
                #     a = np.clip(self.predict(self.q_main_model_1, o, is_train)[0] +
                #                 np.clip(self.q_smoothing * np.random.randn(self.action_size),
                #                         -self.q_smoothing, self.q_smoothing), -1, 1)
                # else:
                a = np.squeeze(self.predict(self.q_main_model_1, o, is_train))
                o2, r, d, _ = self.env.step(a)
                if is_train:
                    self.q_main_model_1.replay_buffer.store(o, a, r, o2, d)
                    if self.clipped_double_q:
                        self.q_main_model_2.replay_buffer.store(o, a, r, o2, d)
                o = o2
                d = False if t == self.max_steps - 1 else d

                if t % self.initial_episode_length == 0 and self.q_main_model_1.replay_buffer.size <= self.warm_up_steps:
                    o = self.env.reset()
                    print('Initial reset at ', t)

                # 2. train
                if is_train and self.q_main_model_1.replay_buffer.size > self.warm_up_steps:
                    # try:
                    self.update_q(self.q_main_model_1)
                    if self.clipped_double_q:
                        self.update_q(self.q_main_model_2)
                    # except:
                    #     print('wait:', self.q_main_model_1.replay_buffer.size)
                if d:
                    break

    def train_model(self, model):
        # beta_decay = self.beta_decay_function(self.idx_episode)
        # decay = self.decay_function(self.idx_episode)

        # if self.per_flag:
        #     if True:  # model is self.q_main_model_1:
        #         batch, priority_info = model.replay_buffer.sample_batch(batch_size=self.batch_size, beta=beta_decay)
        #     else:
        #         batch, priority_info = model.replay_buffer.sample_normal(batch_size=self.batch_size)
        # else:
        batch = model.replay_buffer.sample_batch(200)

        # o = batch['obs1']
        o2 = batch['obs2']
        a = batch['acts']
        # r = batch['rews']
        # d = batch['done']
        #
        # v = self.get_v(o2)
        # target_y = self.discount * np.squeeze(v) * (1 - d) + r
        # input = tf.keras.layers.concatenate([o2, a], axis=1, dtype=tf.float64)
        v = self.q_target_model_1.model_value_estimate([o2, a])
        # v_2 = self.q_target_model_2.model_value_estimate([o2, a])
        # print(tf.reduce_mean(v))
        # print(tf.reduce_mean(v_2))

        # if self.per_flag:
        #     if True:  # model is self.q_main_model_1:
        #         sample_weights = tf.convert_to_tensor(priority_info[0], dtype=tf.float64)
        #         loss = model.train_model(o, a, target_y, sample_weight=sample_weights)[-1]
        #         update_prios = (loss * decay + 1e-16) * np.ones(priority_info[0].shape)
        #         model.replay_buffer.update_priorities(idxes=priority_info[1], priorities=update_prios)
        #     else:
        #         loss = model.train_model_1(10000)[-1]
        # else:
        # loss = model.train_model(o, a, target_y)[-1]
        loss = model.train_model(**self.training_info)[-1]

        # model.set_polyak_weights(self.q_main_model_1.get_weights(), polyak=self.polyak)
        # if self.clipped_double_q is not None:
        #     model.set_polyak_weights(self.q_main_model_2.get_weights(), polyak=self.polyak)

        return v, loss

    # def model_switcher(self, number):
    #     if number % 1 == 0:
    #         self.model_switch = 2 if self.model_switch == 1 else 1
    #         if self.model_switch == 1:
    #             self.current_model = self.q_main_model_1
    #         else:
    #             self.current_model = self.q_main_model_2

    def update_q(self, model):
        vs = []
        losses = []
        self.counter += 1

        # for i in range(self.update_repeat):
        # print('i', i, model)
        v, loss = self.train_model(model=model)
        if model == self.q_main_model_1:
            vs.append(v)
            losses.append(loss)

            if (self.counter) % self.save_frequency == 0:
                # self.q_target_model_1.save_model(directory=self.directory)
                self.q_main_model_1.replay_buffer.save_to_pkl(name='buffer_data.pkl', directory=self.directory)
                print('Saving buffer...')
        # if model == self.q_main_model_1:
            self.vs.append(np.mean(vs))
            self.losses.append(np.mean(losses))

    # def get_v(self, o2):
    #     v_1 = self.q_target_model_1.get_value_estimate(o2)
    #     if self.clipped_double_q is not None:
    #         v_2 = self.q_target_model_2.get_value_estimate(o2)
    #         v = np.where(v_1 < v_2, v_1, v_2)
    #
    #         # print('vs: ', np.mean(o2), np.mean(v_1), np.mean(v_2))
    #     else:
    #         v = v_1
    #     return v


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
