# neuralnet.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 10/29/2019
# Modified by Mahir Morshed for the spring 2021 semester

"""
This is the main entry point for MP3. You should only modify code
within this file and neuralnet_part1.py -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""

import numpy as np
import numpy.random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class1 = 0
class2 = 19

class NeuralNet(nn.Module):
    def __init__(self, lrate, loss_fn, in_size, out_size):
        """
        Initializes the layers of your neural network.

        @param lrate: learning rate for the model
        @param loss_fn: A loss function defined as follows:
            @param yhat - an (N,out_size) Tensor
            @param y - an (N,) Tensor
            @return l(x,y) an () Tensor that is the mean loss
        @param in_size: input dimension
        @param out_size: output dimension
        """
        super(NeuralNet, self).__init__()
        self.loss_fn = loss_fn

        self.layers = [0, 3, 6]

        pool_size = 2
        
        self.model = nn.Sequential(
            nn.Conv2d(3, 6, 5), # 0
            nn.MaxPool2d(pool_size, pool_size),
            nn.ReLU(),

            nn.Conv2d(6, 16, 5), # 3
            nn.MaxPool2d(pool_size, pool_size),
            #nn.ReLU(),

            nn.Flatten(),

            nn.Linear(16*5*5,2), # 6
        )

        self.optimizer = optim.SGD(self.model.parameters(), lr=lrate)


    def forward(self, x):
        """Performs a forward pass through your neural net (evaluates f(x)).

        @param x: an (N, in_size) Tensor
        @return y: an (N, out_size) Tensor of output from the network
        """
        # Normalize
        mean = torch.mean(x)
        sd = torch.std(x)
        x = (x-mean)/sd

        # Reshape
        x = torch.reshape(x, (-1, 3, 32, 32))
        
        return self.model(x)

    def step(self, x,y):
        """
        Performs one gradient step through a batch of data x with labels y.

        @param x: an (N, in_size) Tensor
        @param y: an (N,) Tensor
        @return L: total empirical risk (mean of losses) at this timestep as a float
        """
        y_pred = self.forward(x)
        loss = self.loss_fn(y_pred, y)

        L = loss.item()

        loss.backward()

        self.optimizer.step()
        self.optimizer.zero_grad()

        return L

    def get_total_params(self):
        count = 0

        for layer in self.layers:
            count += self.model[layer].weight.numel() + self.model[layer].bias.numel()

        return count

def fit(train_set,train_labels,dev_set,n_iter,batch_size=100):
    """ Make NeuralNet object 'net' and use net.step() to train a neural net
    and net(x) to evaluate the neural net.

    @param train_set: an (N, in_size) Tensor
    @param train_labels: an (N,) Tensor
    @param dev_set: an (M,) Tensor
    @param n_iter: an int, the number of iterations of training
    @param batch_size: size of each batch to train on. (default 100)

    This method _must_ work for arbitrary M and N.

    The model's performance could be sensitive to the choice of learning rate.
    We recommend trying different values in case your first choice does not seem to work well.

    @return losses: array of total loss at the beginning and after each iteration.
            Ensure that len(losses) == n_iter.
    @return yhats: an (M,) NumPy array of binary labels for dev_set
    @return net: a NeuralNet object
    """

    net = NeuralNet(.01, nn.CrossEntropyLoss(), 3072, 2)

    print("Total Parameters: ", net.get_total_params())

    losses = np.zeros(n_iter)
    yhats = []

    # Train
    tile_count = (n_iter * batch_size) // len(train_labels) + 1
    tiled_set = train_set.tile((tile_count, 1))
    tiled_labels = train_labels.tile(tile_count)

    indices = np.arange(n_iter)
    np.random.shuffle(indices)

    for i in indices:
        losses[i] = net.step(
            tiled_set[i * batch_size : (i + 1) * batch_size, :],
            tiled_labels[i * batch_size : (i + 1) * batch_size]
            )

    # Classify DevSet
    for item in dev_set:
        output = net.forward(item)
        yhats.append(torch.argmax(output).item())

    return list(losses), yhats, net
