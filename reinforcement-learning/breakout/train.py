# Training the AI

import torch
import torch.nn.functional as F
from envs import create_atari_env
from model import ActorCritic
from torch.autograd import Variable

"""
Function that makes sure everything works correctly if the model used by the agent does not have a shared gradient
"""
def ensure_shared_grads(model, shared_model):
    for param, shared_param in zip(model.parameters(), shared_model.parameters()):
        if shared_param.grad is not None:
            return
        shared_param._grad = param.grad

"""
rank: used to desynchronize the agents. That is, to shift the seed of the agent by an offset of 1
params: parameters of the environment
shared_model: shared NN model
optimizer: optimized used by the NN (Shared Adam)
"""
def train(rank, params, shared_model, optimizer):
    torch.manual_seed(params.seed + rank) # shift the seed of the agent with the rank
    env = create_atari_env(params.env_name) # get the environment
    env.seed(params.seed + rank) # align the seed of the environment with respect to each of the agents
    model = ActorCritic(env.observation_space.shape[0], env.action_space)
    # Prepare the input states 
    state = env.reset() # initialize the state as a numby array with dimensions 1 x 42 x42 (black and white image of size 42 x 42)
    state = torch.from_numpy(state) # convert the image into a torch tensor

    done = True # indicate if the game is over. This will be initialized to 'True' when the game is over
    episode_length = 0 # initialize the length of one episode to 0. This will be incremented later
    while True:
        episode_length += 1
        model.load_state_dict(shared_model.state_dict()) # load the state of the shred model as a dict
        if done: # if the game is done, reinitialize the hidden and cell states in the LSTM to 0
            cx = Variable(torch.zeros(1, 256))
            hx = Variable(torch.zeros(1, 256))
        else: # if the game is not done, keep the old hidden and cell states
            cx = Variable(cx.data)
            hx = Variable(hx.data)
        values = [] # output of the critic
        log_probs = [] # log of the probabilities
        rewards = []
        entropies = []
        for step in range(params.num_steps): # for loop over the exploration steps
            value, action_values, (hx, cx) = model((Variable(state.unsqueeze(0)), (hx, cx))) # uses the ActorCritic.forward() function: get the predictions of the model
            prob = F.softmax(action_values) # get a set of probabilities of the possible Q-values
            log_prob = F.log_softmax(action_values) # get distribution of the log probabilities with log_softmax for the possible Q-values
            entropy = -(log_prob * prob).sum(1)
            entropies.append(entropy) # store the computed entropy in the list of entropies

            #  Play the action by taking a random draw from the distribution of probabilities of the softmax
            action = prob.multinomial().data # use the multinomial() function to take arandom draw from the distribution of probabilities
            log_prob = log_prob.gather(1, Variable(action)) # get the log probability associated with the action that was played
            # append the value and log_prov obtained to the respective lists
            values.append(value)
            log_probs.append(log_prob)

            # play the action
            state, reward, done, _ = env.step(action.numpy()) # get the new state, reward, and if the game is over after playing an action
            done = (done or episode_length >= params.max_episode_length) # update the done variable to make sure that an agent is not stuck in a state
            reward = max(min(reward, 1), -1) # make sure the reward is between -1 and +1
            if done: # restart the enviornmnet if the game is done
                episode_length = 0
                state = env.reset()
            state = torch.from_numpy(state) # update the state
            rewards.append(reward) # add the reward to the list
            if done: # break out of loop if done
                break
        
        # After for-loop
        # Update the shared network
        R = torch.zeros(1, 1) # Initialize the cumulative reward
        if not done:
            value, _, _ = model((Variable(state.unsqueeze(0)), (hx, cx))) # get the value of the last state reached by the network
            R = value.data
        values.append(Variable(R)) # append the new value to the values list
        # Initialize the policy loss and the value loss
        policy_loss = 0
        value_loss = 0

        R = Variable(R)
        gae = torch.zeros(1, 1) # intialize the generalized advantage estimation A(a,s) = Q(a,s) - V(s)
        for i in reversed(range(len(rewards))):
            # Calculate the value loss
            R = params.gamma * R + rewards[i] # R = r_0 + (gamma * r_1) + (gamma^2 * r_2) + (gamma^3 * r_3) + ... + (gamma^(n-1) * r_{n-1}) + (gamma^n * V(last_state))
            advantage = R - values[i]
            value_loss = value_loss + 0.5 * advantage.pow(2) # when the optimal action is played: Q*(a*,s) = V*(s)

            # Calculate the policy loss
            TD = rewards[i] + params.gamma * values[i + 1].data - values[i].data # calculate temporal difference
            gae = gae * params.gamma * params.tau + TD
            policy_loss = policy_loss - log_probs[i] * Variable(gae) - 0.01 * entropies[i]

        # Train the weights of the NN
        optimizer.zero_grad()
        (policy_loss + 0.5 * value_loss).backward() # perform backpropagation while giving the policy loss twice as much importance to the policy loss than the value loss
        torch.nn.utils.clip_grad_norm(model.parameters(), 40) # prevent the gradient from getting extremely large values
        ensure_shared_grads(model, shared_model)
        optimizer.step() # perform optimization
