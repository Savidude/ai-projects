from IPython.core.display import clear_output
import numpy as np
import gym
import random
import time
from IPython.display import HTML

import torch

# create environment
env = gym.make("FrozenLake-v0")

# Get Frozen Lake  environment that is deterministic
# from gym.envs.registration import register
# register(
#     id='FrozenLakeNotSlippery-v0',
#     entry_point='gym.envs.toy_text:FrozenLakeEnv',
#     kwargs={'map_name' : '4x4', 'is_slippery': False},
# )
# env = gym.make('FrozenLakeNotSlippery-v0')

# create a Q-table
action_size = env.action_space.n # number of actions
state_size = env.observation_space.n # number of states
qtable = torch.zeros([state_size, action_size])
# print(qtable)
"""
The Q-table shows how much rows (states) and columns (actions) we need

tensor([[0., 0., 0., 0.],
        [0., 0., 0., 0.],
        [0., 0., 0., 0.],
        [0., 0., 0., 0.],
        [0., 0., 0., 0.],
        [0., 0., 0., 0.],
        [0., 0., 0., 0.],
        [0., 0., 0., 0.],
        [0., 0., 0., 0.],
        [0., 0., 0., 0.],
        [0., 0., 0., 0.],
        [0., 0., 0., 0.],
        [0., 0., 0., 0.],
        [0., 0., 0., 0.],
        [0., 0., 0., 0.],
        [0., 0., 0., 0.]])
"""

# Create the hyperparameters
total_episodes = 10000        # Total episodes
learning_rate = 0.1           # Learning rate
max_steps = 100                # Max steps per episode
gamma = 0.99                  # Discounting rate

# Exploration parameters
epsilon = 1.0                 # Exploration rate
max_epsilon = 1.0             # Exploration probability at start
min_epsilon = 0.01            # Minimum exploration probability 
decay_rate = 0.001            # Exponential decay rate for exploration prob

# The Q-learning algorithm
rewards_all_episodes = []

for episode in range(total_episodes):
    # Reset the environment
    state = env.reset()
    step = 0
    done = False
    total_rewards = 0

    for step in range(max_steps):
        # Choose an action a in the current world state (s)
        exp_exp_tradeoff = random.uniform(0, 1) # randomize a number

        # If this number > greater than epsilon --> exploitation (taking the biggest Q value for this state)
        if exp_exp_tradeoff > epsilon:
            ## qtable[state]: get all values of the `state` row of the table
            random_values = qtable[state] + torch.rand(1, action_size)/1000 # add a slightly random value to all actions in the state if all values are 0
            ### eg: qtable[state] = tensor([[0., 0., 0., 0.]])
            ### eg: random_values = tensor([[0.0006, 0.0003, 0.0001, 0.0010]])

            ## torch.max(random_values, 1): gets the maximum value and index of the state
            ## torch.max(random_values, 1)[1]: gets the maximum index of the state
            action = torch.max(random_values, 1)[1].item()
        else: # Else doing a random choice --> exploration
            action = env.action_space.sample()

        # Take the action (a) and observe the outcome state(s') and reward (r)
        new_state, reward, done, info = env.step(action)

        # Update Q(s,a):=  R(s,a) + gamma * max Q(s',a') 
        ## torch.max(qtable, 1): gets the maximum values and indexes of all rows of the qtable
        ## torch.max(qtable, 1)[0]: gets the maximum values of all rows of the qtable
        ## torch.max(qtable, 1)[0][new_state]: gets the maximum value of the 'new_state' row of the qtable
        Q = reward + gamma * torch.max(qtable, 1)[0][new_state].item()
        # TD = R(s,a) + gamma * max Q(s',a') - Q(s,a)
        TD = Q - qtable[state, action]
        # Update Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]
        qtable[state, action] = qtable[state, action] + learning_rate * TD

        total_rewards += reward
        state = new_state
        if done == True: 
            break

    # Reduce epsilon (because we need less and less exploration)
    epsilon = min_epsilon + (max_epsilon - min_epsilon)*np.exp(-decay_rate*episode) 
    rewards_all_episodes.append(total_rewards)

# Calculate and print the average reward per 1000 episodes
rewards_per_thousand_episodes = np.split(np.array(rewards_all_episodes), total_episodes/1000)
count = 1000
print("--------- Avergae Reward Per 1000 Episodes ---------\n")
for r in rewards_per_thousand_episodes:
    print(count, ": ", str(sum(r)/1000))
    count += 1000

# Print the updated Q-table
print("--------- Q-table ---------\n")
print(qtable)

# Use the Q-table to play Frozen Lake
env.reset()

for episode in range(3):
    state = env.reset()
    step = 0
    done = False
    print("****************************************************")
    print("EPISODE ", episode)
    time.sleep(1)

    for step in range(max_steps):
        clear_output(wait = True)
        env.render()
        time.sleep(0.3)

        # Take the action (index) that have the maximum expected future reward given that state
        action = np.argmax(qtable[state,:])
        new_state, reward, done, info = env.step(action)

        if done:
            clear_output(wait = True)
            # Here, we decide to only print the last state (to see if our agent is on the goal or fall into an hole)
            env.render()
            if reward == 1:
                print("You reached the goal!")
                time.sleep(3)
            else:
                print("You fell through a hole!")
                time.sleep(3)
            
            clear_output(wait = True)
            # We print the number of step it took.
            print("Number of steps", step)
            break
        state = new_state
env.close()
