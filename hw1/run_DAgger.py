"""
Code to implement DAgger.
The model used for supervised training consists of 1 hidden layer for simplicity.
The optimization method here is SGD.
Example usage:
    python run_DAgger.py expert_data/Humanoid-v2.pkl --render --num_rollouts 20

Author of this script: Haoxiong Liu
"""

import os
import pickle
import tensorflow as tf
import numpy as np
import tf_util
import gym
import argparse
import load_policy


def weight_variable(shape):     # weight initialization
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('expert_policy_file', type=str)
    parser.add_argument('expert_data_file', type=str)
    parser.add_argument('envname', type=str)
    parser.add_argument('--render', action='store_true')
    parser.add_argument("--max_timesteps", type=int)
    parser.add_argument('--num_rollouts', type=int, default=20,
                        help='Number of expert roll outs')
    parser.add_argument('--aggregate_times', type=int, default=3)
    args = parser.parse_args()

    # load expert policy
    policy_fn = load_policy.load_policy(args.expert_policy_file)

    # read expert data
    f = open(args.expert_data_file, 'rb')
    expert_data = pickle.load(f)
    expert_observations = expert_data['observations']
    expert_actions = expert_data['actions']
    input_dim = len(expert_observations[0])
    output_dim = len(expert_actions[0][0])

    # supervised learning model, 1 hidden layer
    ex_obs = tf.placeholder("float", shape=[input_dim])   # ex_obs
    ex_action = tf.placeholder("float", shape=[output_dim])   # ex_action

    # hidden layer
    W1 = weight_variable([input_dim, 16])
    b1 = bias_variable([16])
    h1 = tf.nn.relu(tf.tensordot(ex_obs, W1, [[0], [0]]) + b1)

    # output layer
    W2 = weight_variable([16, output_dim])
    b2 = bias_variable([output_dim])
    output_action = tf.nn.relu(tf.tensordot(h1, W2, [[0], [0]]) + b2)  # output action

    # loss function and optimization method
    learning_rate = 1e-2
    loss = tf.reduce_sum(tf.square(output_action-ex_action))
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    # create gym env
    env = gym.make(args.envname)
    max_steps = args.max_timesteps or env.spec.timestep_limit

    with tf.Session() as sess:
        tf_util.initialize()

        # train
        for _ in range(args.aggregate_times):
            for index in range(len(expert_observations)):
                train_step.run(feed_dict={ex_obs: expert_observations[index],
                                          ex_action: expert_actions[index][0]})

            # aggregate
            observations = []
            actions = []
            obs = env.reset()
            done = False
            steps = 0
            while not done:
                action = sess.run(output_action, feed_dict={ex_obs: obs})
                obs, _, done, _ = env.step(action)
                observations.append(obs)
                actions.append(policy_fn(obs[None, :]))
                steps += 1
                if args.render:
                    env.render()
                if steps >= max_steps:
                    break
            expert_observations = np.concatenate((expert_observations, observations))
            expert_actions = np.concatenate((expert_actions, actions))

        # apply trained policy
        returns = []
        for i in range(args.num_rollouts):
            print('iter', i)
            obs = env.reset()
            done = False
            totalr = 0.
            steps = 0
            while not done:
                action = sess.run(output_action, feed_dict={ex_obs: obs})
                obs, r, done, _ = env.step(action)
                totalr += r
                steps += 1
                if args.render:
                    env.render()
                if steps >= max_steps:
                    break
            returns.append(totalr)

        print('returns', returns)
        print('mean return', np.mean(returns))
        print('std of return', np.std(returns))


if __name__ == '__main__':
    main()
