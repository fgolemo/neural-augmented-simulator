import matplotlib.pyplot as plt
import numpy as np
import os


def set_box_color(bp, color):
    for patch, color in zip(bp['boxes'], color):
        patch.set_facecolor(color)


sim_results = np.load(os.getcwd() + '/reacher_final_sim_results.npy')
real_results = np.load(os.getcwd() + '/reacher_final_real_results.npy')[1:, :]
sim_results = list(sim_results)
real_results = list(real_results)

ticks = ["mb-1", "gb-1", "mb-2", "gb-2", "mb-10", "gb-10"]

_, ax1 = plt.subplots()
box_plot1 = ax1.boxplot(sim_results, positions=np.array(range(len(sim_results)))*2.0-0.35, sym='', widths=0.6,
                        patch_artist=True)
box_plot2 = plt.boxplot(real_results,  positions=np.array(range(len(real_results)))*2.0+0.35, sym='', widths=0.6,
                        patch_artist=True)
colors = ['pink', 'lightblue', 'lightgreen']

# adding horizontal grid lines

ax1.yaxis.grid(True)
ax1.set_ylabel('Average Rewards')
ax1.set_title('Average reward plot of simulation and real robot rollouts')
set_box_color(box_plot1, color=['pink']*6) # colors are from http://colorbrewer2.org/
set_box_color(box_plot2, color=['lightblue']*6)
plt.plot([], c='pink', label='Simulation Rollouts')
plt.plot([], c='lightblue', label='Real Robot Rollouts')
plt.legend()
#
plt.xticks(range(0, len(ticks) * 2, 2), ticks)
plt.xlim(-2, len(ticks)*2)
plt.savefig(os.getcwd() + '/sim_real_reward_comparision.png')
plt.show()