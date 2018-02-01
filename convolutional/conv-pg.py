import tensorflow as tf
import numpy as np
import gym
from collections import deque

from agent import Agent


def run(env, agent):
    terminal = True

    episode = 0
    reward_sum = 0
    last_50_scores = deque(maxlen=50)
    while True:
        batch_states = []
        batch_rewards = []
        batch_actions = []

        if terminal:
            terminal = False
            state = env.reset()

        while not terminal:
            # Save the current state
            batch_states.append(state)

            # Flip a biased coin to choose an action according
            # to the policy probabilities
            policy = agent.get_policy(state)
            action_index = np.random.choice(agent.action_size, p=policy)

            # Peform the action, get the updated environment values
            state, reward, terminal, _ = env.step(action_index)

            # Clip the reward to be between -1 and 1
            reward = np.clip(reward, -1, 1)
            reward_sum += reward

            # Save the rewards and actions
            batch_rewards.append(reward)
            batch_actions.append(action_index)

        # Calculate the sampled n-step discounted reward
        batch_target_values = []
        for reward in reversed(batch_rewards):
            target_value = reward + discount_factor * target_value
            batch_target_values.append(target_value)

        # Reverse the batch target values so that they are in the correct order
        batch_target_values.reverse()

        agent.train(np.vstack(batch_states), batch_actions, batch_target_values)
        print("Episode: {0}, Score: {1}, Running Avg: {2}".format(episode, reward_sum, np.mean(last_50_scores)))
        last_50_scores.append(reward_sum)
        reward_sum = 0
        episode += 1
