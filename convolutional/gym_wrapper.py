from scipy.misc import imresize
import gym
import numpy as np
import random


class CustomGym():
    def __init__(self, game, skip_actions=4, num_frames=4, width=84, height=84):
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

    def preprocess(self, observation, is_start=False):
        grayscale = observation.astype('float32').mean(2)
        s = imresize(grayscale, (self.width, self.height)).astype('float32') * (1.0/255.0)  #255 is the max pixel value in atari
        s = s.reshape(1, s.shape[0], s.shape[1], 1)

        if is_start or self.state is None:
            self.state = np.repeat(s, self.num_frames, axis=3)
        else:
            self.state = np.append(s, self.state[:,:,:,:self.num_frames-1], axis=3)

        return self.state

    def render(self):
        self.env.render()

    def reset(self):
        return self.preprocess(self.env.reset(), is_start=True)

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
