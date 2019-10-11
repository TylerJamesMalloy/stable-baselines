import gym
import numpy as np

class DiscreteEnv(gym.Env):
    """
    Action mask  environment for testing purposes.

    action_space = gym.spaces.Discrete(2)
    """
    def __init__(self):
        self.action_space = gym.spaces.Discrete(2)

        self.observation_shape = (1, 10, 10)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=self.observation_shape, dtype=np.float16)

    def reset(self):
        self.counter = 0
        self.action_state = 2
        self.action_mask = [1, 1]
        
        return self.state()

    def step(self, action):
        if action == 0:
            assert self.action_state != 0

            self.action_state = 0
            self.action_mask = [0, 1]

        if action == 1:
            assert self.action_state != 1

            self.action_state = 1
            self.action_mask = [1, 0]

        self.counter += 1
        return self.state(), 0, self.finish(), {'action_mask' : self.action_mask}


    def render(self, mode='human'):
        pass

    def finish(self):
        return self.counter == 500

    def state(self):
        temp = np.reshape(np.array([*range(100)]), self.observation_shape)
        obs = temp/100
        return obs

class MultiDiscreteEnv1(gym.Env):
    """
    Action mask  environment for testing purposes.

    action_space = gym.spaces.MultiDiscreteEnv1([2])
    """
    def __init__(self):
        self.action_space = gym.spaces.MultiDiscrete([2])

        self.observation_shape = (1, 10, 10)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=self.observation_shape, dtype=np.float16)

    def reset(self):
        self.counter = 0
        self.action_state = 2
        self.valid_actions = [[1, 1]]
        
        return self.state()

    def step(self, action):
        if action[0] == 0:
            assert self.action_state != 0

            self.action_state = 0
            self.valid_actions = [[0, 1]]

        if action[0] == 1:
            assert self.action_state != 1

            self.action_state = 1
            self.valid_actions = [[1, 0]]

        self.counter += 1
        return self.state(), 0, self.finish(), {'action_mask' : self.valid_actions}


    def render(self, mode='human'):
        pass

    def finish(self):
        return self.counter == 500

    def state(self):
        temp = np.reshape(np.array([*range(100)]), self.observation_shape)
        obs = temp/100
        return obs

class MultiDiscreteEnv2(gym.Env):
    """
    Action mask  environment for testing purposes.

    action_space = gym.spaces.MultiDiscreteEnv1([2, 4])
    """
    def __init__(self):
        self.action_space = gym.spaces.MultiDiscrete([2, 4])

        self.observation_shape = (1, 10, 10)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=self.observation_shape, dtype=np.float16)

    def reset(self):
        self.counter = 0
        self.action_state = 2
        self.valid_actions = [[1, 1], [1, 1, 1, 1]]
        
        return self.state()

    def step(self, action):
        if action[0] == 0:
            assert self.action_state != 0 or action[1] != 0 or action[1] != 2

            self.action_state = 0
            self.valid_actions = [[0, 1], [0, 1, 0, 1]]

        if action[0] == 1:
            assert self.action_state != 1 or action[1] != 1 or action[1] != 3

            self.action_state = 1
            self.valid_actions = [[1, 0], [1, 0, 1, 0]]

        self.counter += 1
        return self.state(), 0, self.finish(), {'action_mask' : self.valid_actions}


    def render(self, mode='human'):
        pass

    def finish(self):
        return self.counter == 500

    def state(self):
        temp = np.reshape(np.array([*range(100)]), self.observation_shape)
        obs = temp/100
        return obs
        