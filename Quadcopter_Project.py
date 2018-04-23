import csv
import numpy as np
import matplotlib.pyplot as plt
import sys
import pandas as pd
from task import Task
from agents.agent import DDPG

## TODO: Train your agent here.
num_episodes = 1000

# init_pose_height = np.random.uniform(0.5, 1.5)
init_pose = np.array([0., 0., 1., 0., 0., 0.])
init_velocities = np.array([0., 0., 0.])
init_angle_velocities = np.array([0., 0., 0.])
target_pos = np.array([0., 0., 10.])
task = Task(init_pose, init_velocities, init_angle_velocities, target_pos=target_pos)

agent = DDPG(task)

file_output = 'data.csv'

# Setup
done = False
labels = ['i_episode', 'total_reward', 'score', 'time', 'x', 'y', 'z', 'phi', 'theta', 'psi',
          'x_velocity', 'y_velocity', 'z_velocity', 'phi_velocity', 'theta_velocity', 'psi_velocity',
          'rotor_speed1', 'rotor_speed2', 'rotor_speed3', 'rotor_speed4']
results = {x: [] for x in labels}
results_last_episode = {x: [] for x in labels}

# Run the simulation, and save the results.
with open(file_output, 'w') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(labels)

    for i_episode in range(1, num_episodes + 1):
        state = agent.reset_episode()  # start a new episode
        while True:
            action = agent.act(state)
            next_state, reward, done = task.step(action)
            agent.step(action, reward, next_state, done)
            state = next_state
            if i_episode == num_episodes:
                to_write = [i_episode] + [agent.total_reward] + [agent.score] \
                           + [task.sim.time] + list(task.sim.pose) + list(task.sim.v) + list(task.sim.angular_v) \
                           + list(action)
                for ii in range(len(labels)):
                    results_last_episode[labels[ii]].append(to_write[ii])
            if done:
                print("Episode = {:4d}, score = {:7.3f} (best = {:7.3f})".format(i_episode, agent.score, agent.best_score))
                to_write = [i_episode] + [agent.total_reward] + [agent.score] \
                           + [task.sim.time] + list(task.sim.pose) + list(task.sim.v) + list(task.sim.angular_v) \
                           + list(action)
                for ii in range(len(labels)):
                    results[labels[ii]].append(to_write[ii])
                writer.writerow(to_write)
                break

plt.figure(1)
plt.plot(results_last_episode['time'], results_last_episode['x'], label='x')
plt.plot(results_last_episode['time'], results_last_episode['y'], label='y')
plt.plot(results_last_episode['time'], results_last_episode['z'], label='z')
plt.legend()
_ = plt.ylim()
plt.show()

def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / N

plt.figure(2)
smoothed_rews = running_mean(results['total_reward'], 10)
plt.plot(results['i_episode'][-len(smoothed_rews):], smoothed_rews, label='total reward (running mean)')
plt.plot(results['i_episode'], results['total_reward'], color='grey', alpha=0.3, label='total reward')
plt.legend()
_ = plt.ylim()
plt.show()

pass