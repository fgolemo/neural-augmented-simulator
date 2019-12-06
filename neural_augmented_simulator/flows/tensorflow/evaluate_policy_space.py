import pdb
import argparse
import tensorflow as tf
import time
import os
import gym
import torch
import matplotlib.pyplot as plt
import numpy as np

from PIL import Image
from neural_augmented_simulator.common.agent.ppo_agent import PPO
from neural_augmented_simulator.common.agent.actor_critic import Memory
from neural_augmented_simulator.arguments import get_args
from neural_augmented_simulator.flows.tensorflow import density_estimator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
plt.rcParams.update({'font.size': 10})


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


def evaluate(model_to_load, args, seed):

    msg = 'Evaluating model seed id: ' + model_to_load
    print(colorize(msg, 'yellow', bold=True))

    env = gym.make(args.env)
    env.seed(seed + 100)
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
                time.sleep(0.05)
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


def evaluate_points(data_points, learned_dist, sess, seed=None):

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

    return np.mean(log_probs_norm)


def setup_and_rollout(exploration, freq, seed):

    os.environ['approach'] = exploration + '-babbling'
    os.environ['variant'] = str(freq)

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

    env_name = args.trained_env
    trained_model_name = 'ppo_{}_{}_{}-babbling_'.format(env_name,
                                                         freq, exploration)

    tf.reset_default_graph()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.InteractiveSession(config=config)

    _, _, learned_dist = density_estimator.setup_and_train_maf(
        train=False)

    sess.run(tf.global_variables_initializer())

    scope = tf.get_variable_scope()
    saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                             scope=scope.name))

    # Load the density model
    density_model = density_model_path + '/checkpoints'
    saver.restore(sess, density_model)

    model = trained_model_path + \
        trained_model_name + str(seed) + '.pth'
    data_points = evaluate(model, args, seed)
    # This is a slight hack as the data was trained by this shift scale
    # to avoid negative values. Better fix is to normalize the data
    # and then normalize back.
    data_points += args.point_scale

    msg = 'Stats based on: ' + model
    print(colorize(msg, 'green', bold=True))

    return learned_dist, sess, data_points


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--env', type=str, default='Nas-ErgoReacherAugmented-Headless-MultiGoal-Halfdisk-Long-v2',
        help='The environment to evaluate on - this can be the headless or graphical version.')
    parser.add_argument(
        '--trained_env', type=str, default='Nas-ErgoReacherAugmented-Headless-MultiGoal-Halfdisk-Long-v2')

    parser.add_argument('--single-seed-eval', action='store_true')
    parser.add_argument('--seed', type=int, default=1,
                        help='Used as the environment seed.')

    parser.add_argument('--density-eval-itr', type=int, default=2,
                        help='Number of rollouts for density evaluation.')

    parser.add_argument('-task', type=str, default='reacher')

    parser.add_argument('--render', action='store_true')
    parser.add_argument('--save-gif', action='store_true')
    parser.add_argument('--point-scale', type=float, default=2.0)

    args, unknown = parser.parse_known_args()

    all_frequencies = [1, 2, 10]
    all_seeds = [1, 2, 3]

    os.environ['task'] = args.task

    if args.single_seed_eval:
        # Go through all seeded policies and evaluate once.
        for seed in all_seeds:
            densities = {}
            for exploration in ['goal', 'motor']:
                for freq in all_frequencies:
                    learned_dist, sess, data_points = setup_and_rollout(
                        exploration, freq, seed)

                    avg_prob = evaluate_points(
                        data_points, learned_dist, sess, seed)
                    col_name = exploration + '_' + str(freq)
                    densities.update({col_name: avg_prob})

            plt.clf()
            plt.bar(*zip(*densities.items()))
            plt.ylabel('Average Probabilities')
            title = 'Rollout point probabilities averaged for seed ' + \
                str(seed)
            plt.suptitle(title)
            figure_name = 'Avg_probs_seed_' + str(seed) + '.png'
            plt.savefig(figure_name)

    else:
        densities = {}

        avg_probs_per_method = {}
        avg_probs_per_method_arr = []

        seed_rollouts_vec = []
        all_seed_rollouts_vec = {}
        _, ax1 = plt.subplots()

        ax1.set_title('Log probabilities across methods.')
        for exploration in ['goal', 'motor']:
            for freq in all_frequencies:
                aggregated_datapoints = []
                average_probabilities = []

                for seed in all_seeds:
                    for eval_itr in range(args.density_eval_itr):
                        learned_dist, sess, data_points = setup_and_rollout(
                            exploration, freq, seed)

                        # Evaluate current seed,rollout
                        seed_per_rollout_prob = evaluate_points(
                            data_points, learned_dist, sess, seed)
                        seed_rollouts_vec.append(seed_per_rollout_prob)

                        aggregated_datapoints.append(data_points)

                # Now plot the box and whiskers per method, per freq
                col_name = exploration + '_' + str(freq)
                all_seed_rollouts_vec.update({col_name: seed_rollouts_vec})
                seed_rollouts_vec = []

        labels, data = [*zip(*all_seed_rollouts_vec.items())]
        ax1.boxplot(data)
        plt.xticks(range(1, len(labels) + 1), labels)
        figure_name = 'box_and_whiskers_for_probs.png'
        plt.savefig(figure_name)

        # aggregated_datapoints = np.concatenate(
        #     aggregated_datapoints, axis=0)
        # msg = 'Stats based aggregated points: '
        # print(colorize(msg, 'cyan', bold=True))
        # evaluate_points(aggregated_datapoints, learned_dist, sess)

        # avg_probs = evaluate_points(
        #     data_points, learned_dist, sess, seed)
        # average_probabilities.append(avg_probs)

        # col_name = exploration + '_' + str(freq)
        # densities.update({col_name: avg_probs})
        # avg_probs_per_method.update({col_name: average_probabilities})
        # pdb.set_trace()
        # avg_probs_per_method_arr.append(average_probabilities)

        # plt.clf()

        # plt.bar(*zip(*densities.items()))
        # plt.ylabel('Average Probabilities')
        # title = 'Rollout point probabilities averaged across all seeds.'
        # plt.suptitle(title)
        # figure_name = 'Avg_probs_across_all_seeds.png'
        # plt.savefig(figure_name)