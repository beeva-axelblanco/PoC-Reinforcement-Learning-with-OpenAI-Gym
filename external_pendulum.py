import gym
import random
import time
import numpy as np


class Pendulum():

    def __init__(self):
        self.max_speed=8
        self.max_torque=2.
        self.dt=.05
        self.viewer = None
        self.np_random = np.random.RandomState()

        high = np.array([1., 1., self.max_speed])

    def step(self,u):
        th, thdot = self.state # th := theta

        g = 10.
        m = 1.
        l = 1.
        dt = self.dt

        u = np.clip(u, -self.max_torque, self.max_torque)[0]
        self.last_u = u # for rendering
        costs = angle_normalize(th)**2 + .1*thdot**2 + .001*(u**2)

        newthdot = thdot + (-3*g/(2*l) * np.sin(th + np.pi) + 3./(m*l**2)*u) * dt
        newth = th + newthdot*dt
        newthdot = np.clip(newthdot, -self.max_speed, self.max_speed) #pylint: disable=E1111

        self.state = np.array([newth, newthdot])
        return np.array([np.cos(self.state[0]), np.sin(self.state[0]), self.state[1]]), -costs, False, {}

    def reset(self):
        high = np.array([np.pi, 1])
        self.state = self.np_random.uniform(low=-high, high=high)
        self.last_u = None
        return np.array([np.cos(self.state[0]), np.sin(self.state[0]), self.state[1]])

def angle_normalize(x):
    return (((x+np.pi) % (2*np.pi)) - np.pi)



class QLearn():
    def __init__(self, env):
        self.q_table = {}
        self.gamma = 0.8
        self.alpha = 1
        self.epsilon = 0
        self.env = env
        self.all_actions = np.arange(-env.max_torque, env.max_torque + 0.1, 0.1)

    def get_rand_action(self, state):
        if random.random() < self.epsilon:
            act = self.env.np_random.uniform(low=-self.env.max_torque, high=self.env.max_torque)
            return act, str(np.digitize([act], self.all_actions)[0][0])

        q_value_actions = [self.q_table.get((state, action), 0) for action in range(len(self.all_actions))]
        max_q = max(q_value_actions)
        if q_value_actions.count(max_q) > 1:
            multiple_max = [ind for ind in range(len(self.all_actions)) if q_value_actions[ind] == max_q]
            index = random.choice(multiple_max)
        else:
            index = q_value_actions.index(max_q)

        return [self.all_actions[index]], str(index)

    def learn(self, state, action, new_state, reward):
        max_q = max([self.q_table.get((state, possible_action), 0) for possible_action in self.all_actions])
        self.q_table[(state, action)] = (1 - self.alpha) * self.q_table.get((state, action), 0) + self.alpha * (reward + self.gamma * max_q)


def gen_state(obs):
    return "".join([str(o) for o in obs])


def run(iterations=100, episodes=5):
    start_time = time.time()
    epidode_time = []

    env = Pendulum()
    q_learn = QLearn(env)
    total_reward = 0

    for episode in range(episodes):
        episode_start_time = time.time()
        observation = env.reset()
        action = 0
        done = False

        s1 = np.digitize([observation[0]], np.arange(-1., 1.1, 0.1))[0]
        s2 = np.digitize([observation[0]], np.arange(-1., 1.1, 0.1))[0]
        s3 = np.digitize([observation[0]], np.arange(-8., 8.1, 0.1))[0]
        state = gen_state([s1, s2, s3])

        for iter in range(iterations):
            action, action_index = q_learn.get_rand_action(state)
            observation, reward, done, info = env.step(action)

            s1 = np.digitize([observation[0]], np.arange(-1., 1., 0.1))[0]
            s2 = np.digitize([observation[0]], np.arange(-1., 1., 0.1))[0]
            s3 = np.digitize([observation[0]], np.arange(-8., 8., 0.1))[0]
            new_state = gen_state([s1, s2, s3])

            q_learn.learn(state, action_index, new_state, reward)
            state = new_state
            total_reward += reward

        epidode_time.append(time.time() - episode_start_time)

    elapsed_time = time.time() - start_time
    average_time_episode = np.average(epidode_time)
    print('Total time: {:.2f}s'.format(elapsed_time))
    print('Average of time by episode: {:.2f}s'.format(average_time_episode))

    return (elapsed_time, average_time_episode)

if __name__ == '__main__':
    run()
