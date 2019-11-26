"""
Density function estimator on Goal Babble data.
"""
import argparse
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random
import math
import os

from sklearn.preprocessing import MinMaxScaler


tfd = tf.contrib.distributions
tfb = tfd.bijectors
layers = tf.contrib.layers

tf.set_random_seed(0)

# dataset-specific settings
settings = {
    'goal': {
        'batch_size': 100,
        'num_bijectors': 4,
        'train_iters': 2e4
    },
    'motor': {
        'batch_size': 100,
        'num_bijectors': 4,
        'train_iters': 2e4
    }
}


def load_data(freq=10, search_space='goal', points_scale=2.0):
    # Load the data
    if search_space == 'goal':
        file_path = os.getcwd() + f'/data/freq{freq}/goal-babbling/'
        file_name = f'goals_and_positions_freq-{freq}.npz'
        pos_action_noise = np.load(file_path + file_name)
        sampled_goals = pos_action_noise['positions'] + points_scale
        search_space = pos_action_noise['goals'] + points_scale

        return search_space, sampled_goals

    elif search_space == 'motor':
        file_path = os.getcwd() + f'/data/freq{freq}/motor-babbling/'
        file_name = f'motor_positions_freq-{freq}.npy'
        pos_action_noise = np.load(file_path + file_name)

        # plt.scatter(pos_action_noise[:, 0],
        #             pos_action_noise[:, 1], s=10, color='blue')
        # plt.savefig('motor_babble.png')

        positions = pos_action_noise + points_scale
        return positions, None


def plot_data(sampled_goals, search_space,
              fig_path,
              goal_clusters=None,
              goal_locs=None,
              display=False):
    _, (ax1, ax2) = plt.subplots(1, 2)
    ax1.scatter(sampled_goals[:, 0],
                sampled_goals[:, 1], s=10, color='blue')
    ax1.set_title('Goals Babbled')
    ax2.scatter(search_space[:, 0],
                search_space[:, 1], s=10, color='blue')
    ax2.set_title('Search Space')

    for goal_indx, clus in enumerate(goal_clusters):
        # for clus in goal_clusters:
        ax1.scatter(clus[:, 0],
                    clus[:, 1], s=10, color='yellow')
        clus_title = chr(65 + goal_indx)
        ax1.text(goal_locs[goal_indx][0],
                 goal_locs[goal_indx][1],
                 clus_title, {'color': 'black', 'fontsize': 20})

    if display:
        plt.show()
    fig_path = fig_path + '/input_data.png'
    plt.savefig(fig_path)


def create_data_iter(input_data, np_dtype=np.float32, target_density='goal'):
    dataset = tf.data.Dataset.from_tensor_slices(input_data.astype(np_dtype))
    dataset = dataset.repeat()
    dataset = dataset.shuffle(buffer_size=input_data.shape[0])
    dataset = dataset.prefetch(3 * settings[target_density]['batch_size'])
    dataset = dataset.batch(settings[target_density]['batch_size'])
    return dataset.make_one_shot_iterator()


def setup_and_train_maf(train=True, data_iterator=None, target_density='goal', dtype=tf.float32):

    if train:
        x_samples = data_iterator.get_next()

    base_dist = tfd.MultivariateNormalDiag(loc=tf.zeros([2], dtype))

    num_bijectors = settings[target_density]['num_bijectors']
    bijectors = []

    for _ in range(num_bijectors):

        bijectors.append(tfb.MaskedAutoregressiveFlow(
            shift_and_log_scale_fn=tfb.masked_autoregressive_default_template(
                hidden_layers=[512, 512])))
        bijectors.append(tfb.Permute(permutation=[1, 0]))
    # Discard the last Permute layer.
    flow_bijector = tfb.Chain(list(reversed(bijectors[:-1])))

    learned_dist = tfd.TransformedDistribution(
        distribution=base_dist,
        bijector=flow_bijector)

    # For visualization
    x = base_dist.sample(8000)
    samples_A = [x]
    names = [base_dist.name]
    for bijector in reversed(learned_dist.bijector.bijectors):
        x = bijector.forward(x)
        samples_A.append(x)
        names.append(bijector.name)

    if train:
        loss = -tf.reduce_mean(learned_dist.log_prob(x_samples))
        train_op = tf.train.AdamOptimizer(1e-4).minimize(loss)
        return loss, train_op, learned_dist
    else:
        return None, None, learned_dist


def evaluate_model(distribution, eval_data, sess, fig_path):

    probabilities = distribution.prob(eval_data)
    prob_then_log = tf.log(probabilities)

    log_scale_probs = shift_log_space(probabilities)

    log_scale_probs, probabilities, prob_then_log = sess.run(
        [log_scale_probs, probabilities, prob_then_log])

    norm_probs = normalize_probs(probabilities)

    log_probs_norm = normalize_probs(log_scale_probs)

    _, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20, 8))

    ax1.scatter(eval_data[:, 0], eval_data[:, 1],
                c=np.squeeze(log_scale_probs))
    ax1.set_title('Shifted Log Probabilities')

    ax2.scatter(eval_data[:, 0], eval_data[:, 1],
                c=np.squeeze(probabilities))
    ax2.set_title('Probabilities')

    ax3.scatter(eval_data[:, 0], eval_data[:, 1],
                c=np.squeeze(norm_probs))
    ax3.set_title('Normalized Probabilities')

    ax4.scatter(eval_data[:, 0], eval_data[:, 1],
                c=np.squeeze(log_probs_norm))
    ax4.set_title('Normalized Shifted Log Probabilities')

    fig_path = fig_path + '/density_evaluation.png'
    plt.savefig(fig_path)


def make_goal_clusters(distribution, goal_list):
    """
    then we can sample a few of the target domain points (i.e. end
    effector positions during rollouts), then for each one of these
    points, we can test them against the density function individually
    and return a
    mean + min + max over all points or we can do Kolmogorov-Smirnov
    test and return that confidence ? How to do in tf?
    """

    # Generate a cluster of points near the goal regions
    # and evaluate these points

    circle_max_radius = 0.02

    goal_clusters = []

    for goal in goal_list:

        target_circle = make_circle_cluster(goal[0],
                                            goal[1],
                                            circle_max_radius)
        goal_clusters.append(target_circle)

    return goal_clusters


def test_goal_targets(distribution, goal_clusters, sess):

    NUM_SLASH = 30
    for target_idx, target in enumerate(goal_clusters):

        # Evaluate the points
        probabilities = sess.run(distribution.prob(target))

        norm_probs = normalize_probs(probabilities)

        # Calculate stat
        print("-"*NUM_SLASH)
        print('State based on probabilities')
        clus_title = chr(65 + target_idx)
        print(clus_title)
        print('Min: ', np.min(norm_probs))
        print('Max: ', np.max(norm_probs))
        print('Mean: ', np.mean(norm_probs))
        print('Median: ', np.median(norm_probs))
        print("-"*NUM_SLASH)

        log_scale_probs = sess.run(shift_log_space(probabilities))
        log_probs_norm = normalize_probs(log_scale_probs)

        # Calculate stat
        print("-"*NUM_SLASH)
        print('State based on log probabilities')
        clus_title = chr(65 + target_idx)
        print(clus_title)
        print('Min: ', np.min(log_probs_norm))
        print('Max: ', np.max(log_probs_norm))
        print('Mean: ', np.mean(log_probs_norm))
        print('Median: ', np.median(log_probs_norm))
        print("-"*NUM_SLASH)


def normalize_probs(probabilities):
    scaler = MinMaxScaler()
    normalizer = scaler.fit(probabilities.reshape(-1, 1))
    normalized_probs = normalizer.transform(
        probabilities.reshape(-1, 1))
    return normalized_probs


def shift_log_space(probabilities):
    """Add a shift to the log function.
    """
    scale = 5.0
    log_scale_probs = tf.log(probabilities + tf.exp(-scale))
    log_scale_probs = log_scale_probs + scale

    return log_scale_probs


def make_circle_cluster(center_x, center_y,
                        max_radius, cluster_size=500):

    circle_points = []

    for _ in range(cluster_size):
        rando_ang = 2 * math.pi * random.random()
        rando_radius = max_radius * math.sqrt(random.random())

        x_coord = rando_radius * math.cos(rando_ang) + center_x
        y_coord = rando_radius * math.sin(rando_ang) + center_y
        circle_points.append([x_coord, y_coord])

    return np.asarray(circle_points)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--target_density', type=str, default='goal')
    parser.add_argument('--freq', type=int, default=10)
    parser.add_argument('--skip_train', action='store_true')
    args = parser.parse_args()

    dtype = tf.float32
    np_dtype = np.float32
    target_density = args.target_density
    freq = args.freq
    num_steps = int(settings[target_density]['train_iters'])
    PT_SCALE = 2.0
    NUM_SLASH = 30

    path = os.getcwd() + '/density_models/' + target_density + \
        '_' + str(freq)

    model_path = path + '/checkpoints'

    if not os.path.exists(path):
        os.makedirs(path)

    skip_train = args.skip_train

    sess = tf.InteractiveSession()

    search_space, sampled_goals = load_data(freq=freq,
                                            search_space=target_density,
                                            points_scale=PT_SCALE)

    if target_density == 'goal':
        # Given a set of goals, generate cluster around the point
        # evaluate, and generate some stats
        goals = [[2.24, 2.025],
                 [2.1, 2.10],
                 [1.78, 2.20]]

        goal_clusters = make_goal_clusters(None,
                                           goals)

        plot_data(sampled_goals, search_space, path,
                  goal_clusters, goals, display=False)

        data_iterator = create_data_iter(sampled_goals,
                                         np_dtype, target_density)
    elif target_density == 'motor':
        data_iterator = create_data_iter(search_space,
                                         np_dtype, target_density)

    loss, train_op, learned_dist = setup_and_train_maf(data_iterator)

    sess.run(tf.global_variables_initializer())

    # Setup model saving
    scope = tf.get_variable_scope()
    saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                             scope=scope.name))

    if not skip_train:

        global_step = []
        np_losses = []
        print("-"*NUM_SLASH)
        print('Training ...')
        print("-"*NUM_SLASH)
        for i in range(num_steps):
            _, np_loss = sess.run([train_op, loss])
            if i % int(1e2) == 0:
                global_step.append(i)
                np_losses.append(np_loss)
            if i % int(1e3) == 0:
                print('\n', '> Iter: ', i, ' Loss: ', np_loss)
                saver.save(sess, model_path)
        start = 0

        plt.clf()
        plt.plot(np_losses[start:])
        loss_fig_path = path + '/Losses.png'
        plt.savefig(loss_fig_path)
        print("-"*NUM_SLASH)
        print('Evaluating ...')
        print("-"*NUM_SLASH)

    if skip_train:
        saver.restore(sess, model_path)

    evaluate_model(learned_dist,
                   search_space,
                   sess,
                   path)

    if target_density == 'goal':
        test_goal_targets(learned_dist,
                          goal_clusters,
                          sess)


if __name__ == "__main__":
    main()
