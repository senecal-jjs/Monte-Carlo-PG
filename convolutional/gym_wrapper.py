from scipy.misc import imresize
import gym
import numpy as np
import random


class CustomGym():
    def __init__(self, game, skip_actions=4, num_frames=1, width=80, height=80):
        self.env = gym.make(game)
        self.num_frames = num_frames
        self.skip_actions = skip_actions
        self.width = width
        self.height = height

        if game == 'SpaceInvaders-v0':
            self.action_space = [1, 2, 3]
        elif game == 'Pong-v0':
            self.action_space = [1, 2, 3]
        elif game == 'Breakout-v0':
            self.action_space = [1, 4, 5]
        else:
            # Use the actions specified by open AI
            self.action_space = range(env.action_space.n)

        self.action_size = len(self.action_space)
        self.observation_shape = self.env.observation_space.shape

        self.state = None
        self.game = game

    def preprocess(self, I):
        I = I[35:195] # crop
        I = I[::2,::2,0] # downsample by factor of 2
        I[I == 144] = 0 # erase background (background type 1)
        I[I == 109] = 0 # erase background (background type 2)
        I[I != 0] = 1 # everything else (paddles, ball) just set to 1
        return I.reshape(1, I.shape[0], I.shape[1], 1)

    def render(self):
        self.env.render()

    def reset(self):
        return self.preprocess(self.env.reset())

    def step(self, action_index):
        action = self.action_space[action_index]
        state, reward, terminal, _ = self.env.step(action)
        return self.preprocess(state), reward, terminal, _
        # accum_reward = 0
        # prev_state = None
        #
        # for _ in range(self.skip_actions):
        #     state, reward, terminal, _ = self.env.step(action)
        #     accum_reward += reward
        #     if terminal:
        #         break
        #     prev_state = state
        #
        # # Takes maximum value for each pixel value over the current and previous
        # # frame. Used to handle flickering ATARI sprites Mnih et al. 2015
        # if self.game == 'SpaceInvaders-v0' and prev_state is not None:
        #     s = np.maximum.reduce([state, prev_state])
        # return self.preprocess(state), accum_reward, terminal, _
