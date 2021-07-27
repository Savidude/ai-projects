# AI for Breakout

# Importing the librairies
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Initializing and setting the variance of a tensor of weights
There are 2 fully connected layers for the actor and for the critic. These 2 FC layers will have weights. 
The function sets a standard deviation for each of these groups of weights.
"""
def normalized_columns_initializer(weights, std=1.0):
    out = torch.randn(weights.size()) # Initialize a torch tensor with random weights that follow a normal distribution
    out *= std / torch.sqrt(out.pow(2).sum(1, keepdim=True).expand_as(out)) # Normalize the weights
    return out

"""
Initializing the weights of the neural network in an optimal way for the learning

m: object representing the neural network
"""
def weights_init(m):
    classname = m.__class__.__name__ # get the type of neural network
    if classname.find('Conv') != -1:
        weight_shape = list(m.weight.data.size()) # list containing the shape of the weights in the NN
        fan_in = np.prod(weight_shape[1:4]) # product of the dimension 1, 2, and 3 of the weight shape (dim1 * dim2 * dim 3)
        fan_out = np.prod(weight_shape[2:4]) * weight_shape[0] # product of the dimension 0, 2, and 3 of the weight shape  (dim0 * dim2 * dim 3)
        w_bound = np.sqrt(6. / (fan_in + fan_out)) # represents the size of the tensor of weights. This  is to generate random weights inversely propotional to the size of the tensor of weights
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0) # set all bias values to 0
    elif classname.find('Linear') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1] # there are less dimensions for a full connection than a convolution connection
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)

# Making the A3C brain

class ActorCritic(torch.nn.Module):

    """
    num_inputs: the dimensions of the input images
    action_space: the space that contains all the actions
    """
    def __init__(self, num_inputs, action_space):
        super(ActorCritic, self).__init__()
        self.conv1 = nn.Conv2d(num_inputs, 32, 3, stride=2, padding=1) # 32 feature detectors of size 3x3, having a stride of 2, and padding 1
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.lstm = nn.LSTMCell(32 * 3 * 3, 256) # memory: (size of the output after the convolution (32 3x3 feature detectors), number of outputs of the LSTM)
        num_outputs = action_space.n
        self.critic_linear = nn.Linear(256, 1) # linear full connection for the critic (number of inputs from the LSTM, number of outputs) output = V(S)
        self.actor_linear = nn.Linear(256, num_outputs) # linear full connection for the actor (number of inputs from the LSTM, number of outputs) output = Q(S,A)
        self.apply(weights_init) # Initialize the weights and biases of the NN by applying the weights_init function to the object
        # A small standard deviation should be given to the actor and a large one to the critic in order to have a good balance between exploration and exploitation
        self.actor_linear.weight.data = normalized_columns_initializer(self.actor_linear.weight.data, 0.01) # normalize actor weights with a small standard deviation
        self.actor_linear.bias.data.fill_(0)
        self.critic_linear.weight.data = normalized_columns_initializer(self.critic_linear.weight.data, 1.0) # normalize actor weights with a large standard deviation
        self.critic_linear.bias.data.fill_(0)
        # initialize the 2 biases in the LSTM to 0
        self.lstm.bias_ih.data.fill_(0)
        self.lstm.bias_hh.data.fill_(0)
        self.train()

    """
    Forward propagation function

    inputs: input images, hidden and cell nodes of the LSTM
    """
    def forward(self, inputs):
        inputs, (hx, cx) = inputs
        x = F.elu(self.conv1(inputs)) # apply the elu activation function
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))
        x = F.elu(self.conv4(x))
        x = x.view(-1, 32 * 3 * 3) # flatten the convolutions (-1: 1 dimensional vector, 32 * 3 * 3: size of the output of the convolution)
        hx, cx = self.lstm(x, (hx, cx))
        x = hx
        return self.critic_linear(x), self.actor_linear(x), (hx, cx)
