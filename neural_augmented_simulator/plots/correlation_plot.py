import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial.polynomial import polyfit
import seaborn as sns
import os
import pickle

sns.set(color_codes=True)
with open(os.getcwd() + '/rewards_and_logprobs.pkl', 'rb') as f:
    rewards_and_logprobs = pickle.load(f)
seed = [1, 2, 3]
variants = ['motor_1', 'goal_1', 'motor_2',  'goal_2', 'motor_10', 'goal_10']

markers = ['o', 'x', '+']
fig, ax = plt.subplots(1, 3, sharey='row', figsize=(15, 6))


for i, s in enumerate(seed):
    rewards_per_seed = []
    log_probs_per_seed = []
    for var in variants:
        rewards = rewards_and_logprobs[var]['rewards'][i]
        log_prob = rewards_and_logprobs[var]['log_probs'][i]
        rewards_per_seed.append(rewards)
        log_probs_per_seed.append(log_prob)
        ax[i].scatter(log_prob, rewards, marker=markers[i], s=150)
        ax[i].text(log_prob, rewards, var, transform=ax[i].transData)
        ax[i].set_title(f'Correlation plot | Seed - {s}')
        ax[i].set_xlabel('Log Probabilities')
        ax[i].set_ylabel('Performance on real robot')
    # plt.legend()

        ax[i].set_xlim(0.3, 1)
        ax[i].set_ylim(-5, 45)
    X_plot = np.linspace(ax[i].get_xlim()[0], ax[i].get_xlim()[1], 100)
    b, m = polyfit(log_probs_per_seed, rewards_per_seed, 1)
    ax[i].plot(X_plot, b + m * X_plot, '--')
# plt.tight_layout()
plt.savefig('correlation_plot.png')
plt.show()



