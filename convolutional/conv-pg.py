import tensorflow as tf
import numpy as np
import gym
from collections import deque

from gym_wrapper import CustomGym
from agent import Agent


def discount_rewards(r, gamma):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):
        if r[t] != 0: running_add = 0 # reset the sum, since this was a game boundary (pong specific!)
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r

def run(in_game):
    discount_factor = 0.99
    env = CustomGym(game=in_game)

    with tf.Session() as sess:
        agent = Agent(session=sess, action_size = env.action_size)
        sess.run(tf.global_variables_initializer())

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
            batch_target_values = discount_rewards(np.stack(batch_rewards), discount_factor)
            # target_value = 0
            # batch_target_values = []
            # for reward in reversed(batch_rewards):
            #     target_value = reward + discount_factor * target_value
            #     batch_target_values.append(target_value)

            # Reverse the batch target values so that they are in the correct order
            # batch_target_values.reverse()
            batch_target_values -= np.mean(batch_target_values)
            batch_target_values /= np.std(batch_target_values)

            agent.train(np.vstack(batch_states), batch_actions, batch_target_values)
            print("Episode: {0}, Score: {1}, Running Avg: {2}".format(episode, reward_sum, np.mean(last_50_scores)))
            last_50_scores.append(reward_sum)
            reward_sum = 0
            episode += 1


if __name__ == '__main__':
    game = 'Pong-v0'
    run(game)
