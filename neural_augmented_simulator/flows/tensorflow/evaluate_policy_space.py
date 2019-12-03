import pdb
import argparse
import tensorflow as tf
import os
import gym
import torch
import numpy as np

from PIL import Image
from neural_augmented_simulator.common.agent.ppo_agent import PPO
from neural_augmented_simulator.common.agent.actor_critic import Memory
from neural_augmented_simulator.arguments import get_args
from neural_augmented_simulator.flows.tensorflow import density_estimator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

color2num = dict(
    gray=30,
    red=31,
    green=32,
    yellow=33,
    blue=34,
    magenta=35,
    cyan=36,
    white=37,
    crimson=38
)


def colorize(string, color, bold=False, highlight=False):
    """
    Colorize a string.
    This function was originally written by John Schulman.
    """
    attr = []
    num = color2num[color]
    if highlight:
        num += 10
    attr.append(str(num))
    if bold:
        attr.append('1')
    return '\x1b[%sm%s\x1b[0m' % (';'.join(attr), string)


def evaluate(model_to_load, args):

    msg = 'Evaluating model seed id: ' + model_to_load
    print(colorize(msg, 'yellow', bold=True))

    env = gym.make(args.env)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    n_episodes = 10
    # max timesteps in one episode
    max_timesteps = 1500

    # constant std for action distribution (Multivariate Normal)
    action_std = 0.5
    # update policy for K epochs
    K_epochs = 80
    # clip parameter for PPO
    eps_clip = 0.2
    # discount factor
    gamma = 0.99

    lr = 0.0003
    betas = (0.9, 0.999)

    memory = Memory()
    ppo = PPO(state_dim, action_dim, action_std,
              lr, betas, gamma, K_epochs, eps_clip)

    ppo.policy_old.load_state_dict(torch.load(
        model_to_load, map_location=torch.device('cpu')))

    data_points = []
    for ep in range(1, n_episodes + 1):
        ep_reward = 0
        state = env.reset()
        for t in range(max_timesteps):
            action = ppo.select_action(state, memory)
            state, reward, done, _ = env.step(action)
            data_points.append(env.unwrapped.get_tip()[0][1:])
            ep_reward += reward
            if args.render:
                env.render()
            if args.save_gif:
                img = env.render(mode='rgb_array')
                img = Image.fromarray(img)
                img.save('./gif/{}.jpg'.format(t))
            if done:
                break

        print('Episode: {}\tReward: {}'.format(ep, int(ep_reward)))
        ep_reward = 0
    env.close()
    return np.asarray(data_points)


def evaluate_points(data_points, learned_dist, sess):

    #data_iterator = density_estimator.create_data_iter(data_points)

    probabilities = sess.run(learned_dist.prob(data_points))
    norm_probs = density_estimator.normalize_probs(probabilities)

    # Calculate stat
    num_slash = 30
    print("-"*num_slash)
    print('State based on probabilities')
    print('Mean: ', np.mean(norm_probs))
    print('Std: ', np.std(norm_probs))
    print('Median: ', np.median(norm_probs))
    print("-"*num_slash)

    log_scale_probs = sess.run(
        density_estimator.shift_log_space(probabilities))
    log_probs_norm = density_estimator.normalize_probs(log_scale_probs)

    # Calculate stat
    print("-"*num_slash)
    print('State based on log probabilities')
    print('Mean: ', np.mean(log_probs_norm))
    print('Std: ', np.std(norm_probs))
    print('Median: ', np.median(log_probs_norm))
    print("-"*num_slash)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--env', type=str, default='ErgoReacherNew-Headless-MultiGoal-Halfdisk-Long-v2')
    parser.add_argument(
        '--stored_env', type=str, default='ErgoReacherAugmented-Headless-MultiGoal-Halfdisk-Long-v2')
    parser.add_argument('--exploration', type=str, default='goal')
    parser.add_argument('--freq', type=int, default=10)
    parser.add_argument('--all-seeds', action='store_true')
    parser.add_argument('--seed', type=int, default=1)

    parser.add_argument('--render', action='store_true')
    parser.add_argument('--save-gif', action='store_true')
    parser.add_argument('--point-scale', type=float, default=2.0)

    #args = parser.parse_args()
    args, unknown = parser.parse_known_args()

    exploration = args.exploration
    freq = args.freq

    current_dir = os.getcwd()
    trained_model_path = current_dir + \
        '/trained_models/ppo/{}-babbling/Variant-{}/'.format(
            exploration, freq)

    density_model_path = current_dir + \
        '/flows/tensorflow/density_models/' + exploration + \
        '_' + str(freq)

    msg = 'Directory for loading models: \n' + trained_model_path + '\n and \n' + \
        density_model_path
    print(colorize(msg, 'green', bold=True))

    env_name = args.stored_env
    trained_model_name = 'ppo_{}_{}_{}-babbling_'.format(env_name,
                                                         freq, exploration)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.InteractiveSession(config=config)

    _, _, learned_dist = density_estimator.setup_and_train_maf(train=False)

    sess.run(tf.global_variables_initializer())

    scope = tf.get_variable_scope()
    saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                             scope=scope.name))

    # Load the density model
    density_model = density_model_path + '/checkpoints'
    saver.restore(sess, density_model)

    aggregated_datapoints = []

    if args.all_seeds:
        # Evaluate per seed as well as data collected across all seeds
        all_trained_models = []
        for i in range(1, 4):
            trained_model_wseed = trained_model_path + \
                trained_model_name + str(i) + '.pth'
            all_trained_models.append(trained_model_wseed)

        for model in all_trained_models:
            data_points = evaluate(model, args)
            # This is a slight hack as the data was trained by this shift scale
            # to avoid negative values. Better fix is to normalize the data
            # and then back.
            data_points += args.point_scale
            aggregated_datapoints.append(data_points)

            msg = 'Stats based on: ' + model
            print(colorize(msg, 'green', bold=True))
            evaluate_points(data_points, learned_dist, sess)

        aggregated_datapoints = np.asarray(aggregated_datapoints)
        msg = 'Stats based aggregated points: '
        print(colorize(msg, 'cyan', bold=True))
        evaluate_points(aggregated_datapoints, learned_dist, sess)
    else:
        model = trained_model_path + \
            trained_model_name + str(args.seed) + '.pth'

        data_points = evaluate(model, args)
        # This is a slight hack as the data was trained by this shift scale
        # to avoid negative values. Better fix is to normalize the data
        # and then back.
        data_points += args.point_scale

        # Load the density model
        density_model = density_model_path + '/checkpoints'
        evaluate_points(data_points, learned_dist, sess)
        msg = 'Stats based on: ' + model
        print(colorize(msg, 'green', bold=True))
