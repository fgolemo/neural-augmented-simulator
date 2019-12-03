import time
import random
import os
import numpy as np
from neural_augmented_simulator.recordings.goal_babbling import GoalBabbling
import matplotlib.pyplot as plt
from neural_augmented_simulator.arguments import get_args

args = get_args()

random.seed(args.seed)
np.random.seed(seed=args.seed)

total_steps = args.total_steps
rest_interval = args.rest_interval
freq = args.freq
steps_until_resample = args.num_steps/freq

# Hyper-parameters
SAMPLE_NEW_GOAL = args.goal_sample_freq
NUMBER_OF_RETRIES = 5
ACTION_NOISE = 0.4
K_NEAREST_NEIGHBOURS = 8
EPSILON = 0.3
task = args.task
goal_babbling = GoalBabbling(ACTION_NOISE, NUMBER_OF_RETRIES, task)
# Reset the robot
goal_babbling.reset_robot()

end_pos = []
history = []
max_history_len = 15000
goal_positions = []
count = 0

file_path = os.getcwd()
if not os.path.isdir(file_path + '/data/robot_recordings/{}/freq{}/{}'.format(args.env_name, args.freq, args.approach)):
    os.makedirs(file_path + '/data/robot_recordings/{}/freq{}/{}'.format(args.env_name, args.freq, args.approach))


print('================================================')
print('Approach is : {} | Task is {} | Frequency is : {}'.format(args.approach, task, args.freq))
print('================================================')


# Create numpy arrays to store actions and observations
sim_trajectories = np.zeros((total_steps, 12))
actions = np.zeros((total_steps, goal_babbling.action_len))

for epi in range(total_steps):
    if epi % rest_interval == 0:  # Reset the robot after every rest interval
        print('Taking Rest at {}'.format(epi))
        observation = goal_babbling.reset_robot()

    if epi % steps_until_resample == 0:
        # goal = [random.uniform(-0.1436, 0.22358), random.uniform(0.016000, 0.25002)]  # Reacher goals
        # goal = [random.uniform(-0.135, 0.0), random.uniform(-0.081, 0.135)]  # Pusher goals
        goal = [random.uniform(-0.7, 0.7), random.uniform(-0.7, 0.7)] # goals are not corelating with tip positions. Take care of normalization
        if count < 10:
            action = goal_babbling.sample_action()
        else:
            action = goal_babbling.sample_action() if random.random() < EPSILON \
                else goal_babbling.action_retries(goal, history)
        count += 1
    else:
        if task == 'reacher':
            action[0], action[3] = 0, 0
    _, end_position, observation = goal_babbling.perform_action(action)  # Perform the action and get the observation
    if len(history) >= max_history_len:
        del history[0]
    history.append((action, end_position))  # Store the actions and end positions in buffer
    end_pos.append(end_position)
    goal_positions.append(goal)
    actions[epi, :] = action  # Store the actions
    sim_trajectories[epi, :] = observation  # Store the observations

# Save the end positions, goals, actions and simulation trajectories.
final_pos = np.asarray(end_pos)
final_goals = np.asarray(goal_positions)
np.savez(file_path + '/data/robot_recordings/{}/freq{}/{}/goals_and_positions.npz'.format(args.env_name, args.freq, args.approach)
         , positions=final_pos, goals=final_goals)
np.savez(file_path + '/data/robot_recordings/{}/freq{}/{}/actions_trajectories.npz'.format(args.env_name, args.freq, args.approach),
         actions=actions, sim_trajectories=sim_trajectories)

# Plot the end_pos, goals and 2D histogram of end_pos
fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(12, 6))
ax1.scatter(final_pos[:, 0], final_pos[:, 1], alpha=0.5, linewidths=1)
# ax1.set_xlim(-0.135, 0.0) # Change axis limits | Pusher : x(-0.135, 0.0) | Reacher : -0.1436, 0.22358
# ax1.set_ylim(-0.081, 0.135) # Change axis limits | Pusher : x(-0.081, 0.135) | Reacher (0.016000, 0.25002)
ax1.set_title("End effector positions for {} trajectories".format(total_steps / 100))
ax2.scatter(final_goals[:, 0], final_goals[:, 1], alpha=0.5, linewidths=1)
ax2.set_title('Goals sampled')
plt.savefig(file_path + '/data/robot_recordings/{}/freq{}/{}/positions-goals.png'.format(args.env_name, freq, args.approach))
plt.close()
# Plot the 2D histogram and save it.
plt.hist2d(final_pos[:, 0], final_pos[:, 1], bins=100)
# plt.xlim(-0.135, 0.0)
# plt.ylim(-0.081, 0.135)
plt.title("2D Histogram of end effector positions")
plt.savefig(file_path + '/data/robot_recordings/{}/freq{}/{}/histogram.png'.format(args.env_name, freq, args.approach))

