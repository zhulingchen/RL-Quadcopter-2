import numpy as np
from physics_sim import PhysicsSim

class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None, 
        init_angle_velocities=None, runtime=5., target_pos=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        self.action_repeat = 3

        self.state_size = self.action_repeat * 6
        self.action_low = 100
        self.action_high = 900
        self.action_size = 4

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.]) 

    def get_reward(self):
        """Uses current pose of sim to return reward."""
        # reward = 1. - 0.3*(abs(self.sim.pose[:3] - self.target_pos)).sum()
        # reward = 1 - 0.3 * np.linalg.norm(self.sim.pose[:3] - self.target_pos)
        # reward = 1 - 0.5*np.linalg.norm(self.sim.pose[:3]-self.target_pos) + 0.3*self.sim.pose[2]
        # reward = 0.5*self.sim.pose[2] + self.sim.v[2] - np.linalg.norm(self.sim.pose[:2]) - np.linalg.norm(self.sim.v[:2]) + 0.5

        reward = 1 - min(0.5 * np.linalg.norm(self.sim.pose[:3] - self.target_pos), 2.0)
        reward += self.sim.v[2]
        reward = np.tanh(reward)  # normalize reward to [-1, 1]

        return reward

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward()

            # extra reward processing

            # if self.sim.pose[2] >= self.target_pos[2]:  # agent has crossed the target height
            #     reward += 50.0  # bonus reward
            # elif done and self.sim.time < self.sim.runtime:  # penalize crash
            #     reward -= 20.0  # extra penalty

            if done:
                if self.sim.time < self.sim.runtime:  # penalty for crash
                    # reward -= 1.0  # extra penalty
                    reward = -1.0
                elif np.abs(self.sim.pose[2] - self.target_pos[2]) <= 2:  # bonus for approaching the target
                    # reward += 10.0  # bonus reward
                    reward = 10.0

            pose_all.append(self.sim.pose)
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat) 
        return state