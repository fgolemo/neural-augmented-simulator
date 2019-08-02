import torch
import torch.nn.functional as F
from torch import nn


class LstmNetRealv1(nn.Module):

    def __init__(self,
                 n_input_state_sim=12,
                 n_input_state_real=12,
                 n_input_actions=6,
                 nodes=128,
                 layers=3,
                 cuda=True):
        super().__init__()

        self.with_cuda = cuda

        self.nodes = nodes
        self.layers = layers

        self.linear1 = nn.Linear(
            n_input_state_real + n_input_actions + n_input_state_sim, nodes)
        self.lstm1 = nn.LSTM(nodes, nodes, layers)
        self.linear2 = nn.Linear(nodes, n_input_state_real)

        self.hidden = self.init_hidden()

    def zero_hidden(self):
        self.hidden[0].data.zero_()
        self.hidden[1].data.zero_()

    def init_hidden(self):
        # the 1 here in the middle is the minibatch size
        h = torch.randn(self.layers, 1, self.nodes)
        c = torch.randn(self.layers, 1, self.nodes)

        if torch.cuda.is_available() and self.with_cuda:
            h = h.cuda()
            c = c.cuda()

        return h, c

    def forward(self, data_in):
        out = F.leaky_relu(self.linear1(data_in))
        out, self.hidden = self.lstm1(out, self.hidden)
        out = F.leaky_relu(out)
        out = torch.tanh(self.linear2(out)) * 2  # added tanh for normalization
        return out


if __name__ == '__main__':
    net = LstmNetRealv1()

    # sequence of length 998 steps
    test_in = torch.zeros((998, 1, 30))
    test_out = net.forward(test_in)

    print("test_out", test_out.size())
