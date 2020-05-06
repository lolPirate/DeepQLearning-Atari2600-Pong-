import os
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch as T
import numpy as np


class DuelingDeepQNetwork(nn.Module):
    def __init__(self, lr, n_actions, input_dims, name, chkpt_dir):
        super().__init__()

        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)

        # The network consists of 3 Convolutional Layers and 2 Fully Connected Layers
        # channels, outgoing filters, kernel size, stride
        self.conv1 = nn.Conv2d(input_dims[0], 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)

        fc_input_dims = self.calculate_conv_output_dims(input_dims)

        self.fc1 = nn.Linear(fc_input_dims, 512)
        self.V = nn.Linear(512, 1)
        self.A = nn.Linear(512, n_actions)

        self.optimizer = optim.RMSprop(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()

        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        self.to(self.device)

    def calculate_conv_output_dims(self, input_dims):
        state = T.zeros(1, *input_dims)
        dims = self.conv1(state)
        dims = self.conv2(dims)
        dims = self.conv3(dims)
        return int(np.prod(dims.size()))

    def forward(self, state):
        state = T.tensor(state, dtype=T.float).to(self.device)
        layer = F.relu(self.conv1(state))
        layer = F.relu(self.conv2(layer))
        layer = F.relu(self.conv3(layer))
        # conv3 shape = Batch Size X n_filters X H X W
        layer = layer.view(layer.size()[0], -1)
        layer = F.relu(self.fc1(layer))
        A = self.A(layer)
        V = self.V(layer)

        return V, A

    def save_checkpoint(self):
        print('....saving....')
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('....loading checkpoint....')
        self.load_state_dict(T.load(self.checkpoint_file))