import gym
import numpy as np


class DummyActionMaskEnv(gym.Env):
    metadata = {'render.modes': ['human', 'system', 'none']}

    def __init__(self):
        self.action_space = gym.spaces.Discrete(3)

        self.observation_shape = (1, 10, 10)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=self.observation_shape, dtype=np.float16)

        self.counter = 0
        self.action_state = 2
        self.valid_actions = [1, 1, 1]

    def reset(self):
        self.counter = 0
        self.action_state = -1
        self.valid_actions = [1, 1, 1]

        return self.state()

    def step(self, action):
        if action == 0:
            if self.action_state == 0:
                raise Exception("Invalid action was selected! Valid actions: {}, "
                                "action taken: {}".format(self.valid_actions, action))
            self.action_state = 0
            self.valid_actions = [0, 1, 1]

        if action == 1:
            if self.action_state == 1:
                raise Exception("Invalid action was selected! Valid actions: {}, "
                                "action taken: {}".format(self.valid_actions, action))
            self.action_state = 1
            self.valid_actions = [1, 0, 1]

        if action == 2:
            if self.action_state == 2:
                raise Exception("Invalid action was selected! Valid actions: {}, "
                                "action taken: {}".format(self.valid_actions, action))
            self.action_state = 2
            self.valid_actions = [1, 1, 0]

        self.counter += 1
        print("Env action mask: " + str(self.valid_actions))
        return self.state(), 1, self.finish(), {'valid_actions': self.valid_actions}

    def render(self, mode='human'):
        pass

    def finish(self):
        return self.counter == 10000

    def state(self):
        tmp = np.reshape(np.array([*range(100)]), self.observation_shape)
        obs = tmp / 100
        return [obs]
