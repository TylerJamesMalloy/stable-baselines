import gym
import numpy as np
from gym.spaces import Discrete, MultiDiscrete


class DummyActionMaskEnvDiscrete(gym.Env):
    metadata = {'render.modes': ['human', 'system', 'none']}

    def __init__(self):
        self.action_space = Discrete(3)

        self.observation_shape = (1, 10, 10)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=self.observation_shape, dtype=np.float16)

        self.counter = 0
        self.valid_actions = [1, 1, 1]

    def reset(self):
        self.counter = 0
        self.valid_actions = [1, 1, 1]
        return self.state()

    def step(self, action: int):
        valid_actions = [1, 1, 1]
        if self.valid_actions[action] == 0:
            raise Exception("Invalid action was selected! Valid actions: {}, "
                            "action taken: {}".format(self.valid_actions, action))
        valid_actions[action] = 0

        self.counter += 1
        self.valid_actions = valid_actions
        return self.state(), 1, self.finish(), {'valid_actions': self.valid_actions}

    def render(self, mode='human'):
        pass

    def finish(self):
        return self.counter == 10000

    def state(self):
        tmp = np.reshape(np.array([*range(100)]), self.observation_shape)
        obs = tmp / 100
        return [obs]


class DummyActionMaskEnvMutliDiscrete(gym.Env):
    metadata = {'render.modes': ['human', 'system', 'none']}

    def __init__(self):
        self.action_space = MultiDiscrete([2, 3, 4])

        self.observation_shape = (1, 10, 10)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=self.observation_shape, dtype=np.float16)

        self.counter = 0
        self.valid_actions = [[1, 1],
                              [1, 1, 1],
                              [1, 1, 1, 1]]

    def reset(self):
        self.counter = 0
        self.valid_actions = [[1, 1],
                              [1, 1, 1],
                              [1, 1, 1, 1]]
        return self.state()

    def step(self, action):
        valid_actions = [[1, 1],
                         [1, 1, 1],
                         [1, 1, 1, 1]]
        for i in range(0, len(action)):
            if self.valid_actions[i][action[i]] == 0:
                raise Exception("Invalid action was selected! Valid actions: {}, "
                                "action taken: {}".format(self.valid_actions, action))
            valid_actions[i][action[i]] = 0

        self.valid_actions = valid_actions
        self.counter += 1

        return self.state(), 1, self.finish(), {'valid_actions': self.flat_action_mask()}

    def render(self, mode='human'):
        pass

    def finish(self):
        return self.counter == 10000

    def state(self):
        tmp = np.reshape(np.array([*range(100)]), self.observation_shape)
        obs = tmp / 100
        return [obs]

    def flat_action_mask(self):
        flat = []
        for mask in self.valid_actions:
            flat.extend(mask)
        return flat
