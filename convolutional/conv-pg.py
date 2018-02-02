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

    # Normalize
    mean = np.mean(discounted_r)
    std = np.std(discounted_r)
    discounted_r = (discounted_r - mean) / (std)
    return discounted_r

def run(in_game):
    env = CustomGym(game=in_game)

    with tf.Session() as sess:
        agent = Agent(session=sess, action_size = env.action_size)
        sess.run(tf.global_variables_initializer())

        gradBuffer = sess.run(tf.trainable_variables())
        for ix,grad in enumerate(gradBuffer):
            gradBuffer[ix] = grad * 0

        # Initialize environment
        discount_factor = 0.99
        prev_state = None # used in computing the difference frame
        observation = env.reset()

        # Track episode rewards and scores
        batch_size = 10
        episode = 0
        reward_sum = 0
        running_reward = None
        last_50_scores = deque(maxlen=50)

        # Track episode history
        ep_history = []

        while True:
            # Get the difference frame
            state = observation - prev_state if prev_state is not None else np.zeros(np.shape(observation))
            prev_state = observation

            # Probabilistically pick an action given the policy network output
            action_probabilities = agent.get_policy(state)

            # Flip a biased coin to choose an action
            action_index = np.random.choice(agent.action_size, p=action_probabilities)

            # Take a step in the environment with the selected action
            observation, reward, done, info = env.step(action_index)

            # Save the episode history
            ep_history.append([state, action_index, reward, observation])

            # Add to the current score
            reward_sum += reward

            if done:
                episode += 1

                # Get the discounted rewards
                ep_history = np.array(ep_history)
                ep_history[:,2] = discount_rewards(ep_history[:,2], discount_factor)

                # Prepare the info needed to calculate the gradients
                feed_dict={agent.advantages: ep_history[:,2], agent.action: ep_history[:,1],
                           agent.state: np.vstack(ep_history[:,0])}

                # Calculate the gradients and add to the gradient buffer
                grads = sess.run(agent.gradients, feed_dict=feed_dict)
                for idx,grad in enumerate(grads):
                    gradBuffer[idx] += grad

                # Perform a network parameter update every batch size episodes
                if episode % batch_size == 0:
                    feed_dict = dictionary = dict(zip(agent.gradient_holders, gradBuffer))
                    _ = sess.run(agent.train_op, feed_dict=feed_dict)

                # boring book-keeping
                ep_history = []
                running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
                print('resetting env. episode %d reward total was %f. running mean: %f' % (episode, reward_sum, running_reward))
                reward_sum = 0
                observation = env.reset() # reset env
                prev_state = None


if __name__ == '__main__':
    game = 'Pong-v0'
    run(game)
