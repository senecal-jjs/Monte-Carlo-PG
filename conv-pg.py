import gym
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Reshape, Flatten
from keras.optimizers import Adam
from keras.layers.convolutional import Convolution2D


def build_model(self):
    model = Sequential()
    model.add(Reshape((1, 80, 80), input_shape=(self.state_size,)))
    model.add(Convolution2D(32, 6, 6, subsample=(3, 3), border_mode='same',
                            activation='relu', init='he_uniform'))
    model.add(Flatten())
    model.add(Dense(64, activation='relu', init='he_uniform'))
    model.add(Dense(32, activation='relu', init='he_uniform'))
    model.add(Dense(self.action_size, activation='softmax'))
    opt = Adam(lr=self.learning_rate)

    # Using categorical crossentropy as a loss is a trick to easily
    # implement the policy gradient. Categorical cross entropy is defined
    # H(p, q) = sum(p_i * log(q_i)). For the action taken, a, you set
    # p_a = advantage. q_a is the output of the policy network, which is
    # the probability of taking the action a, i.e. policy(s, a).
    # All other p_i are zero, thus we have H(p, q) = A * log(policy(s, a))
    model.compile(loss='categorical_crossentropy', optimizer=opt)
    return model
