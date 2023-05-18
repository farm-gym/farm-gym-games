
import numpy as np


class Farmgym_Agent:
    def __init__(self):
        self.farm = None

    def reset(self, farm):
        self.farm = farm

    def init(self, observation):
        pass

    def update(self, obs, reward, terminated, truncated, info):
        pass

    def choose_action(self):
        raise NotImplemented
        # return self.farm.action_space.sample()


class Farmgym_RandomAgent(Farmgym_Agent):
    def __init__(self, mode="POMDP"):
        super(Farmgym_RandomAgent, self).__init__()
        self.x = 1
        self.mode = mode

    def choose_action(self):
        #if self.mode == "POMDP":
            self.x += 0.25
            threshold = 10 / self.x
            if np.random.rand() > threshold:
                return [27] # TODO: choose it be to harvest action !
            return self.farm.action_space.sample()