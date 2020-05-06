import gym
import numpy as np
import cv2


class PreprocessFrame(gym.ObservationWrapper):
    def __init__(self, env, shape):
        super().__init__(env)
        self.shape = (shape[2], shape[0], shape[1])
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=self.shape, dtype=np.float32
        )

    def observation(self, observation):
        new_frame = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
        resized_screen = cv2.resize(
            new_frame, self.shape[1:], interpolation=cv2.INTER_AREA
        )
        new_obs = np.array(resized_screen, dtype=np.uint8).reshape(self.shape)
        new_obs = new_obs / 255.0
        return new_obs
