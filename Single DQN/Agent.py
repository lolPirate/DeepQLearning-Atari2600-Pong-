import numpy as np
import torch as T
from DeepQNetwork import DeepQNetwork
from ReplayBuffer import ReplayBuffer


class Agent:
    def __init__(
        self,
        env_name,
        algo_name,
        mem_size,
        batch_size,
        input_dims,
        n_actions,
        lr=1e-4,
        gamma=0.99,
        epsilon_max=1,
        epsilon_decay=1e-5,
        epsilon_min=0.1,
        checkpoint_dir="tmp/dqn",
        replace=1000,
    ):
        self.env_name = env_name
        self.algo_name = algo_name
        self.mem_size = mem_size
        self.batch_size = batch_size
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.action_space = [i for i in range(self.n_actions)]
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon_max
        self.epsilon_dec = epsilon_decay
        self.epsilon_min = epsilon_min
        self.replace_target_cnt = replace
        self.checkpoint_dir = checkpoint_dir
        self.learn_step_counter = 0

        self.memory = ReplayBuffer(self.mem_size, input_dims, n_actions)

        self.q_eval = DeepQNetwork(
            self.lr,
            self.n_actions,
            self.input_dims,
            name=self.env_name + " " + self.algo_name + "_q_eval",
            chkpt_dir=self.checkpoint_dir,
        )

        self.q_next = DeepQNetwork(
            self.lr,
            self.n_actions,
            self.input_dims,
            name=self.env_name + " " + self.algo_name + "_q_next",
            chkpt_dir=self.checkpoint_dir,
        )

    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            state = T.tensor([observation]).to(self.q_eval.device)
            actions = self.q_eval.forward(state)
            action = T.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)

        return action

    def stor_transition(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def sample_memory(self):
        state, action, reward, new_state, done = self.memory.sample_buffer(
            self.batch_size
        )

        state = T.tensor(state).to(self.q_eval.device)
        action = T.tensor(action).to(self.q_eval.device)
        reward = T.tensor(reward).to(self.q_eval.device)
        new_state = T.tensor(new_state).to(self.q_eval.device)
        done = T.tensor(done).to(self.q_eval.device)

        return state, action, reward, new_state, done

    def replace_target_network(self):
        if self.learn_step_counter % self.replace_target_cnt == 0:
            self.q_next.load_state_dict(self.q_eval.state_dict())

    def decrement_epsilon(self):

        self.epsilon = (
            self.epsilon - self.epsilon_dec
            if self.epsilon > self.epsilon_min
            else self.epsilon_min
        )

    def save_models(self):
        self.q_eval.save_checkpoint()
        self.q_next.save_checkpoint()

    def load_models(self):
        self.q_eval.load_checkpoint()
        self.q_next.load_checkpoint()

    def learn(self):
        if self.memory.mem_ctr < self.batch_size: return

        self.q_eval.optimizer.zero_grad()

        self.replace_target_network()

        states, actions, rewards, new_states, dones = self.memory.sample_buffer(self.batch_size)

        indices = np.arange(self.batch_size)
        q_pred = self.q_eval.forward(states)[indices, actions]
        
        q_next = self.q_next.forward(new_states).max(dim=1)[0]

        q_next[dones] = 0.0

        rewards = T.tensor(rewards).to(self.q_eval.device)

        q_target = rewards + self.gamma*q_next

        loss = self.q_eval.loss(q_target, q_pred).to(self.q_eval.device)
        loss.backward()

        self.q_eval.optimizer.step()
        self.learn_step_counter += 1
        
        self.decrement_epsilon()



