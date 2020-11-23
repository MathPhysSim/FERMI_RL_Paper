import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def fc(x, hidden_size, activation=tf.nn.tanh,
       kernel_initializer=tf.compat.v1.random_uniform_initializer(-0.01, 0.01),
       name=None):
    layer = keras.layers.Dense(hidden_size, activation=activation,
                               kernel_initializer=kernel_initializer,
                               kernel_regularizer=None,
                               bias_initializer=tf.compat.v1.constant_initializer(0.0), name=name)
    return layer(x)


kernel_initializer = tf.compat.v1.random_uniform_initializer(-0.01, 0.01)

obs_dim = 2
act_dim = 2
hidden_sizes = (100, 100)
inputs_state = keras.Input(shape=(obs_dim,), name="state_input")

# h = inputs[:, 0:obs_dim]
h = inputs_state
for hidden_dim in hidden_sizes:
    h = fc(h, hidden_dim, kernel_initializer=kernel_initializer)
V = fc(h, 1, activation=None, kernel_initializer=kernel_initializer, name='V')

l = fc(h, (act_dim * (act_dim + 1) / 2),
       kernel_initializer=kernel_initializer, name='l')
mu = fc(h, act_dim, kernel_initializer=kernel_initializer, name='mu')
state_out_put = [V, l, mu]

value_model = keras.Model(inputs_state, V, name='value_model')
action_model = keras.Model(inputs_state, mu, name='action_model')

inputs_action = keras.Input(shape=(act_dim,), name="action_input")

pivot = 0
rows = []
for idx in range(act_dim):
    count = act_dim - idx
    diag_elem = tf.exp(tf.slice(l, (0, pivot), (-1, 1)))
    non_diag_elems = tf.slice(l, (0, pivot + 1), (-1, count - 1))
    row = tf.pad(tensor=tf.concat((diag_elem, non_diag_elems), 1), paddings=((0, 0), (idx, 0)))
    rows.append(row)
    pivot += count
L = tf.transpose(a=tf.stack(rows, axis=1), perm=(0, 2, 1))
P = tf.matmul(L, tf.transpose(a=L, perm=(0, 2, 1)))
tmp = tf.expand_dims(inputs_action - mu, -1)
A = -tf.multiply(tf.matmul(tf.transpose(a=tmp, perm=[0, 2, 1]),
                           tf.matmul(P, tmp)), tf.constant(0.5))
A = tf.reshape(A, [-1, 1])
Q = tf.add(A, V)

action_state_model = keras.Model(inputs=[inputs_state, inputs_action], outputs=[Q], name='action_state_model')
action_state_model.summary()

tf.print(value_model.predict([[0, 0],[0, 0],[0, 0],[0, 0]]))
tf.print(action_model.predict([[0, 0]]))

tf.print(action_state_model.predict([np.array([[0, 0]]), np.array([[0, 0]])]))

#
# x = layers.Reshape((4, 4, 1))(encoder_output)
# x = layers.Conv2DTranspose(16, 3, activation="relu")(x)
# x = layers.Conv2DTranspose(32, 3, activation="relu")(x)
# x = layers.UpSampling2D(3)(x)
# x = layers.Conv2DTranspose(16, 3, activation="relu")(x)
# decoder_output = layers.Conv2DTranspose(1, 3, activation="relu")(x)
#
# autoencoder = keras.Model(encoder_input, decoder_output, name="autoencoder")
# autoencoder.summary()
