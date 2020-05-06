import gym
import numpy as np


class RepeatActionAndMaxFrame(gym.Wrapper):
    def __init__(self, env, repeat, clip_rewards, no_ops, fire_first):
        super().__init__(env)
        self.repeat = repeat
        self.shape = env.observation_space.low.shape
        self.frame_buffer = np.zeros_like((2, self.shape))
        self.clip_rewards = clip_rewards
        self.fire_first = fire_first
        self.no_ops = no_ops

    def step(self, action):
        total_reward = 0.0
        done = False
        for i in range(self.repeat):
            observation, reward, done, info = self.env.step(action)
            if self.clip_rewards:
                reward = np.clip(np.array([reward]), -1, 1)[0]
            total_reward += reward
            # insert observation to frame_buffer
            idx = i % 2
            self.frame_buffer[idx] = observation
            if done:
                break
        max_frame = np.maximum(self.frame_buffer[0], self.frame_buffer[1])

        return max_frame, total_reward, done, info

    def reset(self):
        obs = self.env.reset()
        no_ops = np.random.randint(self.no_ops) + 1 if self.no_ops > 0 else 0
        for _ in range(no_ops):
            _, _, done, _ = self.env.step(0)
            if done:
                self.env.reset()
        if self.fire_first:
            assert self.env.unwrapped.get_action_meanings()[1] == "FIRE"
            obs, _, _, _, = self.env.step(1)
        self.frame_buffer = np.zeros_like((2, self.shape))
        self.frame_buffer[0] = obs

        return obs
