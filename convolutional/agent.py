import numpy as np
import tensorflow as tf


class Agent():
    def __init__(self, session, action_size, optimizer=tf.train.AdamOptimizer(1e-4)):
        # session:  the tensorflow session
        # action_size: the number of game actions possible
        self.action_size = action_size
        self.optimizer = optimizer
        self.sess = session

        with tf.variable_scope('network'):
            # Store the state, and policy from the network:
            self.state, self.policy = self.build_model(80, 80, 1)

            # Get the weights for the network
            self.weights = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='network')

            # Placeholdersfor the action, and advantage values
            self.action = tf.placeholder('int32', [None], name='action')
            self.advantages = tf.placeholder('float32',[None], name='advantages')

        with tf.variable_scope('optimizer'):
            # Calculate the one hot vector for each action
            one_hot_action = tf.one_hot(self.action, self.action_size, 1.0, 0.0)

            # Clip the policy output to avoid zeros and ones -- these don't play well with taking log.
            self.log_policy = tf.log(tf.clip_by_value(self.policy, 0.000001, 0.999999))

            # For a given state and action, compute the log of the policy at
            # that action for that state. This also works on batches.
            self.log_pi_for_action = tf.reduce_sum(tf.multiply(self.log_policy, one_hot_action), reduction_indices=1)

            # We want to do gradient ascent on the expected discounted reward.
            # The gradient of the expected discounted reward is the gradient of
            # log pi * (R), where R is the discounted reward from the
            # given state following the policy pi. Since we want to maximise
            # this, we define the policy loss as the negative and get tensorflow
            # to do the automatic differentiation for us.
            self.policy_loss = -tf.reduce_mean(self.log_pi_for_action * self.advantages)

            tvars = tf.trainable_variables()
            self.gradient_holders = []
            for idx,var in enumerate(tvars):
                placeholder = tf.placeholder(tf.float32,name=str(idx)+'_holder')
                self.gradient_holders.append(placeholder)

            self.gradients = tf.gradients(self.loss,tvars)

            optimizer = tf.train.AdamOptimizer(learning_rate=lr)
            self.train_op = optimizer.apply_gradients(zip(self.gradient_holders,tvars))

            # # Compute the gradient of the loss with respect to all the weights,
            # # and create a list of tuples consisting of the gradient to apply to
            # # the weight.
            # grads = tf.gradients(self.policy_loss, tvars)
            # # grads, _ = tf.clip_by_global_norm(grads, 40.0)
            # grads_vars = list(zip(grads, tvars))
            #
            # # Create an operator to apply the gradients using the optimizer.
            # # Note that apply_gradients is the second part of minimize() for the
            # # optimizer, so will minimize the loss.
            # self.train_op = optimizer.apply_gradients(grads_vars)

    def build_model(self, h, w, channels):
          state = tf.placeholder('float32', shape=(None, h, w, channels), name='state')
          # First convolutional layer
          with tf.variable_scope('conv1'):
              conv1 = tf.contrib.layers.convolution2d(inputs=state,
              num_outputs=16, kernel_size=[8,8], stride=[4,4], padding="VALID",
              activation_fn=tf.nn.relu,
              weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
              biases_initializer=tf.zeros_initializer())

          # Second convolutional layer
          with tf.variable_scope('conv2'):
              conv2 = tf.contrib.layers.convolution2d(inputs=conv1, num_outputs=32,
              kernel_size=[4,4], stride=[2,2], padding="VALID",
              activation_fn=tf.nn.relu,
              weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
              biases_initializer=tf.zeros_initializer())

          # Flatten the network
          with tf.variable_scope('flatten'):
              flatten = tf.contrib.layers.flatten(inputs=conv2)

          # Fully connected layer with 256 hidden units
          with tf.variable_scope('fc1'):
              fc1 = tf.contrib.layers.fully_connected(inputs=flatten, num_outputs=256,
              activation_fn=tf.nn.relu,
              weights_initializer=tf.contrib.layers.xavier_initializer(),
              biases_initializer=tf.zeros_initializer())

          # The policy output
          with tf.variable_scope('policy'):
              policy = tf.contrib.layers.fully_connected(inputs=fc1,
              num_outputs=self.action_size, activation_fn=tf.nn.softmax,
              weights_initializer=tf.contrib.layers.xavier_initializer(),
              biases_initializer=None)

          return state, policy

    def get_policy(self, state):
        policy = self.sess.run(self.policy, {self.state: state})
        return policy.flatten()

    def train(self, feed_dict_in):
        # Training
        self.sess.run(self.train_op, feed_dict=feed_dict_in)
