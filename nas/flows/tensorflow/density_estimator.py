"""
Density function estimator on Goal Babble data.
"""
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
    'GB': {
        'batch_size': 100,
        'num_bijectors': 4,
        'train_iters': 2e4
    }
}

DTYPE = tf.float32
NP_DTYPE = np.float32
TARGET_DENSITY = 'GB'
FREQ = 10
#NUM_STEPS = 2
NUM_STEPS = int(settings[TARGET_DENSITY]['train_iters'])
PT_SCALE = 2.0
NUM_SLASH = 30


def load_data(freq=10, points_scale=2.0):
    # Load the data
    file_path = os.getcwd() + f'/data/freq{freq}/goal-babbling/'
    file_name = f'goals_and_positions_freq-{freq}.npz'
    pos_action_noise = np.load(file_path + file_name)
    sampled_goals = pos_action_noise['positions'] + points_scale
    search_space = pos_action_noise['goals'] + points_scale

    return sampled_goals, search_space


def plot_data(sampled_goals, search_space,
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
    plt.savefig('input_data.png')


def create_data_iter(input_data):
    dataset = tf.data.Dataset.from_tensor_slices(input_data.astype(NP_DTYPE))
    dataset = dataset.repeat()
    dataset = dataset.shuffle(buffer_size=input_data.shape[0])
    dataset = dataset.prefetch(3 * settings[TARGET_DENSITY]['batch_size'])
    dataset = dataset.batch(settings[TARGET_DENSITY]['batch_size'])
    return dataset.make_one_shot_iterator()


def train_maf(data_iterator):

    x_samples = data_iterator.get_next()

    base_dist = tfd.MultivariateNormalDiag(loc=tf.zeros([2], DTYPE))

    num_bijectors = settings[TARGET_DENSITY]['num_bijectors']
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

    # visualization
    x = base_dist.sample(8000)
    samples_A = [x]
    names = [base_dist.name]
    for bijector in reversed(learned_dist.bijector.bijectors):
        x = bijector.forward(x)
        samples_A.append(x)
        names.append(bijector.name)

    loss = -tf.reduce_mean(learned_dist.log_prob(x_samples))
    train_op = tf.train.AdamOptimizer(1e-4).minimize(loss)

    return loss, train_op, learned_dist


def evaluate_model(distribution, eval_data, sess):

    probabilities = distribution.prob(eval_data)
    log_probabilities = distribution.log_prob(eval_data)
    prob_then_log = tf.log(probabilities)
    #####################################
    # Add a shift-up to the log function.
    #####################################
    scale = 5.0
    log_scale_probs = tf.log(probabilities + tf.exp(-scale))

    log_scale_probs, probabilities, prob_then_log, original_log_probs = sess.run(
        [log_scale_probs, probabilities, prob_then_log, log_probabilities])

    # Note: Taking tf.log(dist.prob(x)) == dist.log_prob(x)
    # np.testing.assert_allclose(prob_then_log, original_log_probs, rtol=1e-7)

    ############
    # Normalizer
    ############
    scaler = MinMaxScaler()
    prob_normalizer = scaler.fit(probabilities.reshape(-1, 1))
    norm_probs = prob_normalizer.transform(probabilities.reshape(-1, 1))

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
                c=np.squeeze(original_log_probs))
    ax4.set_title('tf.log_Probabilities')

    plt.savefig('density_evaluation.png')


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

    for target_idx, target in enumerate(goal_clusters):

        # Evaluate the points
        probabilities = sess.run(distribution.prob(target))

        scaler = MinMaxScaler()
        prob_normalizer = scaler.fit(probabilities.reshape(-1, 1))
        norm_probs = prob_normalizer.transform(probabilities.reshape(-1, 1))

        # Calculate stat
        print("-"*NUM_SLASH)
        clus_title = chr(65 + target_idx)
        print(clus_title)
        print('Min: ', np.min(norm_probs))
        print('Max: ', np.max(norm_probs))
        print('Mean: ', np.mean(norm_probs))
        print('Median: ', np.median(norm_probs))
        print("-"*NUM_SLASH)

        # TODO: STAT Test ?


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

    sess = tf.InteractiveSession()

    sampled_goals, search_space = load_data(points_scale=PT_SCALE)

    # Given a set of goals, generate cluster around the point
    # evaluate, and generate some stats
    goals = [[2.24, 2.025],
             [2.1, 2.10],
             [1.78, 2.20]]

    goal_clusters = make_goal_clusters(None,
                                       goals)

    plot_data(sampled_goals, search_space,
              goal_clusters, goals, display=False)

    data_iterator = create_data_iter(sampled_goals)

    loss, train_op, learned_dist = train_maf(data_iterator)

    sess.run(tf.global_variables_initializer())

    global_step = []
    np_losses = []
    print("-"*NUM_SLASH)
    print('Training ...')
    print("-"*NUM_SLASH)
    for i in range(NUM_STEPS):
        _, np_loss = sess.run([train_op, loss])
        if i % int(1e3) == 0:
            global_step.append(i)
            np_losses.append(np_loss)
        if i % int(1e4) == 0:
            print('\n', '> Iter: ', i, ' Loss: ', np_loss)
    start = 0

    plt.plot(np_losses[start:])
    plt.savefig('Losses.png')
    print("-"*NUM_SLASH)
    print('Evaluating ...')
    print("-"*NUM_SLASH)

    evaluate_model(learned_dist,
                   search_space,
                   sess)

    test_goal_targets(learned_dist,
                      goal_clusters,
                      sess)


if __name__ == "__main__":
    main()
