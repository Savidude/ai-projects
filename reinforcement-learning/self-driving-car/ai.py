import numpy as np
import random
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable


# Creating the architecture of the Neural Network
class Network(nn.Module):
    HIDDEN_LAYER_SIZE = 30

    """
    input_size: number of inputs to the neural network
    nb_action:  number of actions that can be taken
    fc1:        connection between the input layer and hidden layer
    fc2:        connection between the hidden layer and output layer
    """

    def __init__(self, input_size, nb_action):
        super(Network, self).__init__()
        self.input_size = input_size
        self.nb_action = nb_action
        self.fc1 = nn.Linear(in_features=self.input_size, out_features=self.HIDDEN_LAYER_SIZE)
        self.fc2 = nn.Linear(in_features=self.HIDDEN_LAYER_SIZE, out_features=self.nb_action)

    """
    Perform forward propagation
    state:      input of the neural network
    :returns   Q-values for each possible action
    """

    def forward(self, state):
        # Activate the hidden layer's neurons
        x = F.relu(self.fc1(state))
        # Get output from the values of the hidden layer
        q_values = self.fc2(x)
        return q_values


# Implementing experience replay
class ReplayMemory(object):
    """
    capacity: number of events kept in memory
    memory:   the last 100 (if capacity = 100) events
    """

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    """
    Appends a new event into the memory while ensuring that the memory always has 100 (if capacity = 100) events
    event: the event being added to the memory
    """

    def push(self, event):
        self.memory.append(event)
        if len(self.memory) > self.capacity:
            del self.memory[0]

    """
    Get random samples from the memory
    batch_size: size of the sample being returned
    :returns    randomly selected samples
    """

    def sample(self, batch_size):
        # if list = ((state1, action1, reward1), (state2, action2, reward2))
        # then zip(*list) = ((state1, state2), (action1, action2), (reward1, reward2))
        samples = zip(*random.sample(self.memory, batch_size))

        # Maps samples to a torch variable that contains tensors and gradients
        # For each batch in the sample, we have to concatenate it with respect to the state (first dimension (index 0))
        return map(lambda x: Variable(torch.cat(x, 0)), samples)


# Implementing Deep Q-Learning
class Dqn:
    TEMPERATURE = 100
    REPLAY_MEMORY_CAPACITY = 100000
    REPLAY_MEMORY_RANDOM_SAMPLE_SIZE = 100
    REWARD_WINDOW_SIZE = 1000

    """
    gamma:          delay coefficient
    reward_window:  sliding window of the mean of the last 100 rewards. Used to evaluate the evolution of the AI performance
    model:          the neural network
    memory:         object of the experience memory
    optimizer:      tools required to perform gradient descent (uses Adam optimizer)
    
    Variables composing the transition events
    last_state:     details of the previous state of the agent (contains the 3 signals, orientation, -orientation)
    last_action:    index of the rotation angle of the car when the last action is performed [0, 20, -20] (index 0 = 0, index 1 = 20, index 2 = -20)
    last_reward:    previous reward
    """

    def __init__(self, input_size, nb_action, gamma):
        self.gamma = gamma
        self.reward_window = []
        self.model = Network(input_size=input_size, nb_action=nb_action)
        self.memory = ReplayMemory(self.REPLAY_MEMORY_CAPACITY)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.last_state = torch.Tensor(input_size).unsqueeze(0)
        self.last_action = 0
        self.last_reward = 0

    """
    Function that will select the right action each time (left, right, forward)
    state: input state of the neural network (torch tensor)
    :returns action that must be taken
    """

    def select_action(self, state):
        q_values = self.model(Variable(state, volatile=True))

        # Get probabilities of each action using a softmax function from the output of the neural network
        # softmax([1,2,3]) = [0.04, 0.11, 0.85] => softmax([1,2,3] * 3) = [0, 0.02, 0.98]
        # using a temperature parameter of 3 increases the probability output of the softmax. We can be more sure of the Q-value corresponding to the action to be taken
        probs = F.softmax(input=(q_values * self.TEMPERATURE))

        # get a random draw corresponding to the probability to decide the action that must be taken
        action = probs.multinomial()
        return action.data[0, 0]

    """
    Train the neural network
    batch_state:        current state
    batch_next_state:   next state
    batch_reward:       reward for the batch
    batch_action:       action taken for the batch
    """

    def learn(self, batch_state, batch_next_state, batch_reward, batch_action):
        # Get actions chosen by the network instead of all of them (0, 1, 2)
        # This gets the best actions to perform for each of the input states of batch_state
        outputs = self.model(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1)

        # Get the maximum of the Q-values of the next state
        next_outputs = self.model(batch_next_state).detach().max(1)[0]

        # target = reward + (gamma * next_outputs)
        target = batch_reward + (self.gamma * next_outputs)

        # Obtain the loss and run backpropagation to update the weights
        td_loss = F.smooth_l1_loss(outputs, target)
        self.optimizer.zero_grad()
        td_loss.backward(retain_variables=True)
        self.optimizer.step()

    """
    Update details of the agent when it reaches a new state and return the action that needs to be taken
    reward:     reward from the previous action
    new_signal: signal obtained after taking the previous action
    :returns    action that was played
    """

    def update(self, reward, new_signal):
        new_state = torch.Tensor(new_signal).float().unsqueeze(0)
        # A new event of transition is observed. This event is added to the memory
        # This transition contains (last state, new state, last action performed, last reward obtained)
        self.memory.push((self.last_state, new_state, torch.LongTensor([int(self.last_action)]), torch.Tensor([self.last_reward])))

        # Play a new action
        action = self.select_action(new_state)

        # Learn from the previous actions
        if len(self.memory.memory) > self.REPLAY_MEMORY_RANDOM_SAMPLE_SIZE:
            batch_state, batch_next_state, batch_action, batch_reward = self.memory.sample(
                self.REPLAY_MEMORY_RANDOM_SAMPLE_SIZE)
            self.learn(batch_state, batch_next_state, batch_reward, batch_action)

        self.last_action = action
        self.last_state = new_state
        self.last_reward = reward
        self.reward_window.append(reward)

        # The reward window needs to be a fixed size to show the evolution of the reward
        if len(self.reward_window) > self.REWARD_WINDOW_SIZE:
            del self.reward_window[0]

        return action

    """
    Compute the score of the sliding window of rewards
    :returns mean score of all records in sliding window of rewards
    """

    def score(self):
        return sum(self.reward_window) / (len(self.reward_window) + 1)

    """
    Save the brain of the car such that it can be reused whenever the application is quit
    """

    def save(self):
        torch.save({'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict()}, 'last_brain.pth')

    """
    Loads the saved brain of the car
    """

    def load(self):
        if os.path.isfile('last_brain.pth'):
            print("=> loading checkpoint...")
            checkpoint = torch.load('last_brain.pth')
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("done")
        else:
            print("no checkpoint found")
