import numpy as np
from make_env import make_env
from Agent import Agent
from utils import plot_learning_curve


class Game:
    def __init__(
        self, env_name, algo_name, mem_size, batch_size, load_checkpoint=False
    ):
        self.env_name = env_name
        self.algo_name = algo_name
        self.batch_size = batch_size
        self.env = make_env(env_name)
        self.mem_size = mem_size
        self.agent = Agent(
            self.env_name,
            self.algo_name,
            self.mem_size,
            self.batch_size,
            self.env.observation_space.shape,
            self.env.action_space.n,
            checkpoint_dir="models/",
        )
        self.best_score = -np.inf
        self.load_checkpoint = load_checkpoint
        self.n_games = 500

    def start(self):
        if self.load_checkpoint:
            self.agent.load_models()

        fname = (
            self.agent.algo_name
            + " "
            + self.agent.env_name
            + "_lr"
            + str(self.agent.lr)
            + "_"
            + str(self.n_games)
            + "games"
        )

        figure_file = "plots/" + fname + ".png"
        n_steps = 0

        scores, eps_history, steps_array = list(), list(), list()

        for i in range(self.n_games):
            score = 0
            done = False
            observation = self.env.reset()
            while not done:
                self.env.render()
                action = self.agent.choose_action(observation)
                new_observation, reward, done, _ = self.env.step(action)
                score += reward
                if not self.load_checkpoint:
                    self.agent.stor_transition(
                        observation, action, reward, new_observation, done
                    )
                    self.agent.learn()
                observation = new_observation
                n_steps += 1
            scores.append(score)
            eps_history.append(self.agent.epsilon)
            steps_array.append(n_steps)

            avg_score = np.mean(scores[-100:])
            print(
                f"Episodes : {i} \t Score : {score} \t Average Scores : {round(avg_score,2)} \t Best Score : {round(self.best_score,2)} \t Epsilon : {round(self.agent.epsilon,4)} \t Steps : {n_steps}"
            )

            if avg_score > self.best_score:
                if not self.load_checkpoint:
                    self.agent.save_models()
                self.best_score = avg_score

        plot_learning_curve(steps_array, scores, eps_history, figure_file)


if __name__ == "__main__":
    game = Game("PongNoFrameskip-v4", "DeepQNetwork", 2000, 32)
    game.start()
