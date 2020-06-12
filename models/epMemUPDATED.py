import torch
import torch.nn as nn
from collections import deque
from numpy import random as npr

class episodicRecall(nn.Module):
    """
    Built on:
    03142020
    """

    def __init__(self, hidden_size, memory_size=1000, local_noise=.5, degneration=.001, starting_beta=.35):
        super(episodicRecall, self).__init__()
        self.hidden_size = hidden_size

        #alpha layer operations
        self.cos = nn.CosineSimilarity(dim=-1)
        self.alpha_layer = nn.Linear(hidden_size, 1, bias=False)
        self.alpha_activation = nn.Sigmoid()

        # Layers
        self.out = nn.Linear(hidden_size, hidden_size, bias=False)

        # Non linear activations
        self.softmaxEl = nn.Softmax(dim=-1)
        self.relu = nn.ReLU()

        self.current_drop = nn.Dropout(local_noise)
        self.memory = deque(maxlen=memory_size)
        self.memory += [torch.nn.init.xavier_normal_(torch.zeros(size=(1, hidden_size))) for _ in range(100)]

        self.confidence_boost = degneration
        self.beta = starting_beta
        

    def forward(self, x, mode=1):
        """ This function lives and dies off of having previously
            calculated the projection layer . . . this is run in
            training outside of the forward() call, post creation of
            the encoder outputs. """
        out = None

        if mode==1:
            out = self.single(x)
        else:
            out = self.multi(x)

        if self.training:
            if npr.rand() > self.beta:
                out = x
            self.beta+=self.confidence_boost

        return out

    def single(self, c):
        """ This function lives and dies off of having previously
            calculated the projection layer . . . this is run in
            training outside of the forward() call, post creation of
            the encoder outputs. """

        # (1) Create Alpha layer from Cosine Similarity of current pre_outputs
        #    and global memory.
        noisy_input = self.current_drop(c)
        m = torch.cat(list(self.memory), dim=0)

        alphas = self.cos(noisy_input, m)
        alphas = alphas.unsqueeze(-1) * m
        alphas = self.alpha_activation(self.alpha_layer(alphas))

        # (2)
        shared_context = alphas * m
        context = torch.tanh(shared_context.sum(dim=0).unsqueeze(0))
        #It is random, and not advisable right off, but I've had a lot of
        # success actually with the following . . .
        #(shared_context.mean(dim=0).unsqueeze(0))

        context = self.relu(self.out((context)))

        self.memory.append(c.detach().view(1, -1))

        return context

    def multi(self, st):
        """ This function lives and dies off of having previously
            calculated the projection layer . . . this is run in
            training outside of the forward() call, post creation of
            the encoder outputs. """
        context_i = []

        for c in st:
            # (1) Create Alpha layer from Cosine Similarity of current pre_outputs
            #    and global memory.
            noisy_input = self.current_drop(c)
            m = torch.cat(list(self.memory), dim=0)

            alphas = self.cos(noisy_input, m)
            alphas = alphas.unsqueeze(-1) * m
            alphas = self.alpha_activation(self.alpha_layer(alphas))

            # (2)
            shared_context = alphas * m
            context = torch.tanh(shared_context.sum(dim=0).unsqueeze(0))
            #It is random, and not advisable right off, but I've had a lot of
            # success actually with the following . . .
            #(shared_context.mean(dim=0).unsqueeze(0))

            context = self.relu(self.out((context)))

            self.memory.append(c.detach().view(1, -1))

            context_i.append(context)

        return torch.cat(context_i, dim=0)

