import torch
from torch import nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, input_dim, output_dim, detach_lstm_state, init_weights):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(input_dim[0], 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.lstm = nn.LSTMCell(1152, 512)
        self.critic_linear = nn.Linear(512, 1)
        self.actor_linear = nn.Linear(512, output_dim)
        self.detach_lstm_state = detach_lstm_state
        if init_weights:
            self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                # nn.init.kaiming_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LSTMCell):
                nn.init.constant_(module.bias_ih, 0)
                nn.init.constant_(module.bias_hh, 0)

    def forward(self, x, h, c):
        x = F.relu(self.conv1(x/255.))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(x.size(0), -1)
        h, c = self.lstm(x, (h, c))
        if self.detach_lstm_state:
            h = h.detach()
            c = c.detach()
        return self.actor_linear(h), self.critic_linear(h), h, c