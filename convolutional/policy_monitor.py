import os
import datetime
import csv
import numpy as np
import tensorflow as tf
from gym.wrappers import Monitor
import gym

from gym_wrapper import CustomGym


class PolicyMonitor():
    '''
    Runs an episode in an environment, and saves a video
    '''
    def __init__(self, game, save_path, agent):
        self.env = CustomGym(game=game, monitor=True)
        self.video_dir = os.path.join(save_path, "videos")
        self.agent = agent

        try:
            os.makedirs(self.video_dir)
        except FileExistsError:
            pass


    def evaluate(self):
        # Run an episode
        done = False
        state = self.env.reset()
        total_reward = 0
        episode_length = 0

        while not done:
            # Probabilistically pick an action given the policy network output
            action_probabilities = self.agent.get_policy(state)

            # Select the best action
            action_index = np.argmax(action_probabilities)

            next_state, reward, done, _ = self.env.step(action_index)
            total_reward += reward
            episode_length += 1
            state = next_state

        # Save score to file
        with open(os.path.join(save_path, "evaluation.csv", a), "a") as myFile:
            writer = csv.writer(myFile)
            writer.writerows([datetime.datetime.now().time(), episode_length, total_reward])
