import numpy as np
import random
import math


class ENV:
    def __init__(self, num_recommend, num_category):
        self.num_recommend = num_recommend
        self.num_category = num_category
        self.user_personality = random.random() # np.random.dirichlet(np.ones(self.num_category), size=1)[0]
        self.user_threshold = random.random()
        self.done = 0
        self.satisfy = 1
        self.views = 0
        self.disappoint_factor = 0.9
        self.state = np.array([1]) # np.reshape(np.array([1, 1]), (1, self.num_category))

    def reset(self):
        self.done = 0
        self.satisfy = 1
        self.views = 0
        # self.user_personality = np.random.dirichlet(np.ones(self.num_category), size=1)[0]
        # self.user_threshold = random.random()
        return self.state

    def step(self, action):
        self.state, reward = self.get_state_reward(action)
        # reward = self.get_reward(self.state[0])
        # if reward < self.user_threshold * self.num_recommend:
        #     self.satisfy *= self.disappoint_factor
        # else:
        #     self.satisfy = min(1, self.satisfy * (2 - self.disappoint_factor))
        # if self.satisfy < self.user_threshold:
        #     self.done = 1

        return reward, self.state, self.done

    def get_state_reward(self, action):
        # STATE: (satisfy == dwell_time, view)
        reward = 0
        state = np.array([])
        # action = self.softmax(action[0])
        num_recommend = round(self.num_recommend * self.satisfy)
        seed = np.random.choice(a=list(range(self.num_category)), size=num_recommend, p=action)

        for category in range(self.num_category):
            state = np.append(state, np.count_nonzero(seed == category))
        self.views = sum([math.ceil(min(s, p)) for s, p in zip(state, self.user_personality * num_recommend)])

        if self.views < self.user_threshold * self.num_recommend:
            self.satisfy *= self.disappoint_factor
            # reward = -self.views
        else:
            self.satisfy = min(1, self.satisfy * (2 - self.disappoint_factor))
            reward = self.views
        if self.satisfy < self.user_threshold:
            self.done = 1
            reward = -1000


        self.views /= self.num_recommend
        self.state = np.array([self.satisfy])

        return np.reshape(self.state, (1, self.num_category)), reward


        # state = np.array([])
        # action = self.softmax(action[0])
        # num_recommend = round(self.num_recommend * self.satisfy)
        # seed = np.random.choice(a=list(range(self.num_category)), size=num_recommend, p=action)
        #
        # for category in range(self.num_category):
        #     state = np.append(state, np.count_nonzero(seed == category))
        # # state = [np.round(a * p + 0.3) for a, p in zip(self.user_personality, state)]
        # state = [math.ceil(min(s, p)) for s, p in zip(state, self.user_personality * num_recommend)]
        # reward = sum(state) * self.satisfy
        #
        # return np.reshape(state, (1, self.num_category)), reward

    def get_reward(self, state):
        print(state)
        print(self.num_recommend * self.satisfy)
        return sum(state) # - round(self.num_recommend * self.satisfy)

    def softmax(self, x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)
