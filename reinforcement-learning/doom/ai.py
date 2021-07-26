# AI for Doom

# Importing the libraries
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

# Importing the packages for OpenAI and Doom
import gym
from gym.wrappers import SkipWrapper
# Actions that can be performed in Doom
from ppaquette_gym_doom.wrappers.action_space import ToDiscrete

# Importing the other Python files
import experience_replay, image_preprocessing

# Part 1 - Building the AI

# Making the brain
class CNN(nn.Module):
    
    """
    Initializes the CNN. The CNN has 3 convolutional layers, and 1 hidden layer 

    number_actions: the number of actions that can be performed in the Doom environment
    """
    def __init__(self, number_actions):
        super(CNN, self).__init__()
        self.convolution1 = nn.Conv2d(in_channels = 1, out_channels = 32, kernel_size = 5)      # Applies convolutions to the input image 
        self.convolution2 = nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 3)   # Takes the first convolution layer as input and apply convolutions on top of that
        self.convolution3 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 2)   # Takes the second convolution layer as input and apply convolutions on top of that
        self.fc1 = nn.Linear(in_features = self.count_neurons((1, 80, 80)), out_features = 40)    # Flatten all pixels obtained by the different series of convolutions
        self.fc2 = nn.Linear(in_features = 40, out_features = number_actions)

    """
    Count the number of neurons in the vector after convolutions are applied

    image_dim:  dimensions of the original input image (1, 80, 80) = (working with black and white 80x80 images)
    """
    def count_neurons(self, image_dim):
        x = Variable(torch.rand(1, *image_dim)) # create a fake image with the given dimensions
        # Propagate signals into the NN from the convolutional layers until the flattening layer (until convolution3)
        x = F.relu(F.max_pool2d(self.convolution1(x), 3, 2)) # Apply convolution 1 to the fake input image with 3x3 Max Pooling with a stride of 2 and activate with Relu
        x = F.relu(F.max_pool2d(self.convolution2(x), 3, 2)) # Apply convolution 2 to the fake input image with 3x3 Max Pooling with a stride of 2 and activate with Relu
        x = F.relu(F.max_pool2d(self.convolution3(x), 3, 2)) # Apply convolution 3 to the fake input image with 3x3 Max Pooling with a stride of 2 and activate with Relu
        return x.data.view(1, -1).size(1) # Return the size of the flattened layer

    """
    Forward propagation function

    x: input image
    """
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.convolution1(x), 3, 2))
        x = F.relu(F.max_pool2d(self.convolution2(x), 3, 2))
        x = F.relu(F.max_pool2d(self.convolution3(x), 3, 2))
        x = x.view(x.size(0), -1) # Flattern convolutional layer
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Making the body
class SoftmaxBody(nn.Module):
    
    """
    T: temperature
    """
    def __init__(self, T):
        super(SoftmaxBody, self).__init__()
        self.T = T

    """
    Propagate the output signal of the brain to the the body of the AI

    outputs: Output signals of the brain
    """
    def forward(self, outputs):
        probs = F.softmax(outputs * self.T)   
        actions = probs.multinomial() # get a random draw corresponding to the probability to decide the action that must be taken
        return actions

# Making the AI
class AI:

    """
    Initialize the AI

    brain: CNN brain
    body: body of the agent
    """
    def __init__(self, brain, body):
        self.brain = brain
        self.body = body

    """
    Python function: similar to init, but performs action when the class is used (https://www.geeksforgeeks.org/__call__-in-python/)

    inputs: input images
    """
    def __call__(self, inputs):
        input = Variable(torch.from_numpy(np.array(inputs, dtype = np.float32))) # receive the input images
        output = self.brain(input) # get an output from the NN
        actions = self.body(output) # interpret an action from the NN output
        return actions.data.numpy()

# Part 2 - Training the AI with Deep Convolutional Q-Learning

# Getting the Doom environment
# obtain the "DoomCorridor-v0" Doom environment, provide the dimensions in black and white
doom_env = image_preprocessing.PreprocessImage(SkipWrapper(4)(ToDiscrete("minimal")(gym.make("ppaquette/DoomCorridor-v0"))), width = 80, height = 80, grayscale = True)
# doom_env = image_preprocessing.PreprocessImage(gym.make('VizdoomCorridor-v0'), width = 256, height = 256, grayscale = True)
# import the whole game with videos
doom_env = gym.wrappers.Monitor(doom_env, "videos", force = True)
number_actions = doom_env.action_space.n # get the number of actions of the doom environment

# Building an AI
cnn = CNN(number_actions)
softmax_body = SoftmaxBody(T = 1.0)
ai = AI(brain = cnn, body = softmax_body)

# Setting up Experience Replay
n_steps = experience_replay.NStepProgress(env = doom_env, ai = ai, n_step = 10)
memory = experience_replay.ReplayMemory(n_steps = n_steps, capacity = 10000)
    
# Implementing Eligibility Trace
"""
This algorithm is implemented by following the steps indicated in a paper

batch: a set of inputs and targets to train the AI
"""
def eligibility_trace(batch):
    gamma = 0.99
    inputs = []
    targets = []
    for series in batch: # series: sequence of n (10) transitions
        # get the state of the first (series[0].state) and last series[-1].state transition of the series
        input = Variable(torch.from_numpy(np.array([series[0].state, series[-1].state], dtype = np.float32))) 
        output = cnn(input) # get the prediction of the output
        # set a cumulative reward of 0 if the last stae of the series was reached, or the max Q-value of the last state (output[1]) of the series was not reached
        cumul_reward = 0.0 if series[-1].done else output[1].data.max()
        for step in reversed(series[:-1]): # iterate from the one before last state of the sequence to the first state
            cumul_reward = step.reward + gamma * cumul_reward
        state = series[0].state # get input state
        target = output[0].data # get target associated with the first input state
        target[series[0].action] = cumul_reward # assign cumilative reward to the action of the first state
        inputs.append(state)
        targets.append(target)
    return torch.from_numpy(np.array(inputs, dtype = np.float32)), torch.stack(targets)

# Making the moving average on 100 steps
class MA:

    def __init__(self, size):
        self.list_of_rewards = []
        self.size = size

    """
    rewards: in some cases rewards can be in lists, but in other cases, it can be a single element
    """
    def add(self, rewards):
        if isinstance(rewards, list):
            self.list_of_rewards += rewards
        else:
            self.list_of_rewards.append(rewards)
        while len(self.list_of_rewards) > self.size:
            del self.list_of_rewards[0]
            
    def average(self):
        return np.mean(self.list_of_rewards)
ma = MA(100)

# Training the AI
loss = nn.MSELoss()
optimizer = optim.Adam(cnn.parameters(), lr = 0.001)
nb_epochs = 100
for epoch in range(1, nb_epochs + 1):
    memory.run_steps(200) # each epoch will be 200 runs of 10 steps
    # sample some batches from these runs
    # take 128 batches which are randomly selected from the 200 runs
    for batch in memory.sample_batch(128):
        inputs, targets = eligibility_trace(batch)
        inputs, targets = Variable(inputs), Variable(targets)
        predictions = cnn(inputs)
        loss_error = loss(predictions, targets)
        optimizer.zero_grad()
        loss_error.backward()
        optimizer.step()
    rewards_steps = n_steps.rewards_steps()
    ma.add(rewards_steps)
    avg_reward = ma.average()
    print("Epoch: %s, Average Reward: %s" % (str(epoch), str(avg_reward)))
