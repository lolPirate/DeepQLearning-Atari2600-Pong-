import gym
from gym.wrappers.monitoring.video_recorder import VideoRecorder
import numpy as np
from Agent import Agent
from make_env import make_env
import torch as T


class Test_Game:
    def __init__(
        self, env_name, algo_name, mem_size, batch_size, load_checkpoint=False
    ):

        self.env_name = env_name
        self.algo_name = algo_name
        self.mem_size = mem_size
        self.batch_size = batch_size
        self.load_checkpoint = load_checkpoint
        self.checkpoint_dir = "test_models/"
        self.video_models = 'videos/5_game_test.mp4'
        self.env = make_env(env_name)
        self.recorder = VideoRecorder(self.env, self.video_models)
        self.agent = Agent(
            self.env_name,
            self.algo_name,
            self.mem_size,
            self.batch_size,
            self.env.observation_space.shape,
            self.env.action_space.n,
            epsilon_max= 0.1,
            checkpoint_dir=self.checkpoint_dir,
        )
        self.agent.load_models()

    def start(self, games=1):
        for i in range(games):
            done = False
            observation = self.env.reset()
            while not done:
                self.recorder.capture_frame()
                action = self.agent.choose_action(observation)
                new_observation, reward, done, _ = self.env.step(action)
                observation = new_observation
        
        self.env.close()
        self.recorder.close()


if __name__ == '__main__':
    game = Test_Game("PongNoFrameskip-v4", "DeepQNetwork", 2000, 32)
    game.start(games=5)
                


