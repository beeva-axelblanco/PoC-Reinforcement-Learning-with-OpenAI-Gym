import gym
import random
import time
import numpy as np


class QLearn():
    def __init__(self, env):
        self.q_table = {}
        self.gamma = 0.8
        self.alpha = 1
        self.epsilon = 0
        self.env = env
        self.all_actions = np.arange(env.action_space.low, env.action_space.high + 0.1, 0.1)

    def get_rand_action(self, state):
        if random.random() < self.epsilon:
            act = self.env.action_space.sample()
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

    env = gym.make('Pendulum-v0')
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
