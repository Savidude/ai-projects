import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

# if gpu is to be used (https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# test simple linear function
W = 2
b = 0.3
## unsqueeze(1): converts a tensor of size 100 into a 100x1 tensor 
x = (torch.arange(100, dtype=torch.float).unsqueeze(1)).to(device = device)
y = W * x + b

###### PARAMS ######
learning_rate = 0.01
num_episodes = 1000

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.linear1 = nn.Linear(1,1)
        
    def forward(self, x):
        output = self.linear1(x)
        return output
    
mynn = NeuralNetwork()

loss_func = nn.MSELoss()
optimizer = optim.Adam(params=mynn.parameters(), lr=learning_rate)

for i_episode in range(num_episodes):
    
    predicted_value = mynn(x)
    
    loss = loss_func(predicted_value, y)
    
    optimizer.zero_grad() # Zero the gradients of the model parameters before doing backpropagation
    loss.backward() # compute the gradient of the loss
    optimizer.step() # update the model's parameters
    
    if i_episode % 50 == 0:
        print("Episode %i, loss %.4f " % (i_episode, loss.item()))

plt.figure(figsize=(12,5))
plt.plot(x.data.numpy(), y.data.numpy(), alpha=0.6, color='green')
plt.plot(x.data.numpy(), predicted_value.data.numpy(), alpha=0.6, color='red')

plt.show()
