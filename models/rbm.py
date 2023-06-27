import pandas as pd
import torch

class RBM:
    def __init__(self, n_vis, n_hid):
        """
        Initialize the parameters (weights and biases) we optimize during the training process
        :param n_vis: number of visible units
        :param n_hid: number of hidden units
        """


        self.W = torch.randn(n_hid, n_vis)

        self.v_bias = torch.randn(1, n_vis)

        self.h_bias = torch.randn(1, n_hid)

    def sample_h(self, x):
        """
        Sample the hidden units
        :param x: the dataset
        """

        wx = torch.mm(x, self.W.t())

        activation = wx + self.h_bias.expand_as(wx)

        p_h_given_v = torch.sigmoid(activation)
        return p_h_given_v, torch.bernoulli(p_h_given_v)

    def sample_v(self, y):
        """
        Sample the visible units
        :param y: the dataset
        """

        wy = torch.mm(y, self.W)

        activation = wy + self.v_bias.expand_as(wy)

        p_v_given_h = torch.sigmoid(activation)

        return p_v_given_h, torch.bernoulli(p_v_given_h)

    # def train(self, v0, vk, ph0, phk):
    #     """
    #     Perform contrastive divergence algorithm to optimize the weights that minimize the energy
    #     This maximizes the log-likelihood of the model
    #     """
    #
    #     self.W += (torch.mm(v0.t(), ph0) - torch.mm(vk.t(), phk)).t()
    #
    #     self.v_bias = torch.sum((v0 - vk), 0)
    #     self.h_bias = torch.sum((ph0 - phk), 0)
