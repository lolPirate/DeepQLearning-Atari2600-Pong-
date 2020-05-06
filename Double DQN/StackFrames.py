import gym
from collections import deque
import numpy as np


class StackFrames(gym.ObservationWrapper):
    def __init__(self, env, repeat):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            env.observation_space.low.repeat(repeat, axis=0),
            env.observation_space.high.repeat(repeat, axis=0),
            dtype=np.float32,
        )
        self.stack = deque(maxlen=repeat)

    def reset(self):
        self.stack.clear()
        observation = self.env.reset()
        for _ in range(self.stack.maxlen):
            self.stack.append(observation)
        return np.array(self.stack).reshape(self.observation_space.low.shape)

    def observation(self, observation):

        self.stack.append(observation)
        return np.array(self.stack).reshape(self.observation_space.low.shape)
