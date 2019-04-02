import gym
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.optim as optim
import pandas as pd
from torch.autograd import Variable
from tqdm import tqdm, trange

from NN import NN
from ReplayMemory import ReplayMemory



def optimize(Q_0):
    # get action-value function for state + 1
    Q_1 = p_network(Variable(torch.from_numpy(state_1).type(torch.FloatTensor)))
    # take action with max Value of Q_1
    maxQ_1, _ = torch.max(Q_1, -1)

    # Q-Target as copy of Q_O (Q_0 is the action value function of state 0)
    target_Q = Variable(Q_0.clone())

    #change max value action to satisfy Bellman Equation
    # -> Muliply the highest Action Value of Q_1 with Gamma and add received reward for current state
    target_Q[action] = reward + torch.mul(maxQ_1.detach(), GAMMA)

    # Calculate loss between Q_0 and Q-Target (Mean Squarred Error)
    loss = loss_function(Q_0, target_Q)

    # train model (backpropagation with loss)
    # so that Q_0 approximates Q-Target
    p_network.zero_grad()
    loss.backward()
    optimizer.step()


def optimize_with_ER():
    if len(R_MEMORY) > BATCH_SIZE:
        transitions = R_MEMORY.sample(BATCH_SIZE)
        batch = R_MEMORY.Transition(*zip(*transitions))

        states_0 = torch.stack(batch.state)
        actions_0 = torch.tensor(batch.action).view(BATCH_SIZE, 1)
        rewards = torch.tensor(batch.reward)
        states_1 = torch.tensor(batch.next_state)

        #print("states one", states_1.shape, "states new",  torch.tensor(batch.next_state).shape)
        # action-values for the states_0
        # was passiert bei q_0genau?? -> wegen random action nicht immer maximum
        max_Qs_0 = p_network(states_0.float()).gather(1, actions_0)
        max_Qs_1 = target_network(states_1.float()).max(1)[0]

        # Compute the expected Q values
        target_Qs = rewards + (max_Qs_1 * GAMMA)
        # loss = loss_function(max_Qs_0, target_Qs.unsqueeze(1))
        loss = loss_function(max_Qs_0, target_Qs.unsqueeze(1))

        p_network.zero_grad()
        loss.backward()
        optimizer.step()



env = gym.make('MountainCar-v0').env
# for better reproducing, should work without
env.seed(1)
np.random.seed(1)
torch.manual_seed(1)


UPDATE_TARGET_N = 10
BATCH_SIZE = 128
RUNS = 1000
STEPS = 500
R_MEMORY = ReplayMemory(10000)
MIN_EPSILON = 0.04
successes = 0
GAMMA = 0.99
epsilon = 0.4
l_r = 0.001

state_0 = env.reset

position = []


p_network = NN()
target_network = NN()
target_network.load_state_dict(p_network.state_dict())
target_network.eval()
loss_function = nn.MSELoss() # mean squared error
optimizer = optim.SGD(p_network.parameters(), lr=l_r)

# Scheduler will adjust learning rate after every run with the factor gamma
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

steps_history = []
last_nrsteps = 0
run_durations = []


# turn Experience Replay here on or off
#######################################
Experience_Replay = False ##############
#######################################

for run in trange(RUNS):
    state_0 = env.reset()

    for step in range(STEPS):
        # get action-value function of state_0
        Q_0 = p_network(Variable(torch.from_numpy(state_0).type(torch.FloatTensor)))

        # epsilon probability of choosing a random action
        if np.random.rand(1) < epsilon:
            action = np.random.randint(0, 3)
        else:
            # choose max value action
            _, action = torch.max(Q_0, -1)  # returns values, indices
            action = action.item()
        # make next step and receive next state and reward, done true when successfull
        state_1, _, done, _ = env.step(action)

        # Rewardfunction:
        # get reward based on car position
        reward = state_1[0] + 0.5
        # increase reward for task completion
        if state_1[0] >= 0.5:
            reward += 1

        if Experience_Replay:
            # store transition in replay memory
            R_MEMORY.push(torch.tensor(state_0), action, reward, state_1)
            optimize_with_ER()

        else: optimize(Q_0)

        if done or step + 1 == STEPS:
            # if successful
            if state_1[0] >= 0.5:
                successes += 1
                # Adjust epsilon
                if epsilon >= MIN_EPSILON:
                    epsilon *= .99
                # Adjust learning rate
                scheduler.step()
            # gather history
            position.append(state_1[0])
            steps_history.append(step + 1)
            break
        else:
            state_0 = state_1


    if run % UPDATE_TARGET_N == 0:
        target_network.load_state_dict(p_network.state_dict())

        print('successful runs: {:d} - {:.4f}%'.format(successes, successes / RUNS * 100))
        #print("steps", steps_history[-1)








print("average steps for last 200 runs",np.sum(steps_history[-200:])/200)
print("average steps for last 400 runs",np.sum(steps_history[-400:])/400)
env.close()

# plt.figure(2, figsize=[10,5])
# p = pd.Series(position)
# ma = p.rolling(10).mean()
# plt.plot(p, alpha=0.8)
# plt.plot(ma)
# plt.xlabel('Run')
# plt.ylabel('Position')
# plt.title('Car Final Position')
# plt.savefig('Final Position - Modified.png')
# plt.show()



plt.figure(2, figsize=[10,5])
p = pd.Series(steps_history)
ma = p.rolling(100).mean()
plt.plot(p, alpha=0.8)
plt.plot(ma)
plt.xlabel('Runs')
plt.ylabel('steps taken')
plt.title('Car Final Position')
plt.savefig('Final Position - Modified.png')
plt.show()

#legende zu den Bildern für z.B durchschnit steps usw


#225

