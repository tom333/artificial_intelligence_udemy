
# Importing the libraries
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

# Importing the packages for OpenAI and Doom
import gym
from gym import wrappers
import vizdoomgym

# Importing the other Python files
import experience_replay, image_preprocessing

class CNN(nn.Module):
    def __init__(self, number_actions):
        super(CNN, self).__init__()
        self.convolution1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5)
        self.convolution2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3)
        self.convolution3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=2)
        self.fc1 = nn.Linear(in_features=self.count_neurons((1,80,80)), out_features=40)
        self.fc2 = nn.Linear(in_features=40, out_features=number_actions)
        
    def count_neurons(self, image_dim):
        x = Variable(torch.rand(1, *image_dim))
        x = F.relu(F.max_pool2d(self.convolution1(x), kernel_size=3, stride=2))
        x = F.relu(F.max_pool2d(self.convolution2(x), kernel_size=3, stride=2))
        x = F.relu(F.max_pool2d(self.convolution3(x), kernel_size=3, stride=2))
        return x.data.view(1, -1).size(1)
    
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.convolution1(x), kernel_size=3, stride=2))
        x = F.relu(F.max_pool2d(self.convolution2(x), kernel_size=3, stride=2))
        x = F.relu(F.max_pool2d(self.convolution3(x), kernel_size=3, stride=2))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
class SoftmaxBody(nn.Module):
    def __init__(self, T):
        super(SoftmaxBody, self).__init__()
        self.T = T
        
    def forward(self, outputs):
        probs = F.softmax(outputs * self.T, dim=0)
        action = probs.multinomial(num_samples=1)
        return action
    
class AI:
    def __init__(self, brain, body):
        self.brain = brain
        self.body = body
        
    def __call__(self, inputs):
        inputs = Variable(torch.from_numpy(np.array(inputs, dtype=np.float32)))
        outputs = self.brain(inputs)
        actions = self.body(outputs)
        return actions.data.numpy()


doom_env = image_preprocessing.PreprocessImage(gym.make("VizdoomCorridor-v0"), width=80, height=80, grayscale=True)
doom_env = wrappers.Monitor(doom_env, "videos", force = True)

number_actions = doom_env.action_space.n

cnn = CNN(number_actions)
softmax_body = SoftmaxBody(T = 1.)

ai = AI(brain=cnn, body=softmax_body)

# XP replay
n_steps = experience_replay.NStepProgress(doom_env, ai, n_step=10)
memory = experience_replay.ReplayMemory(n_steps = n_steps, capacity = 10000)

#Eligibility trace
def eligibility_trace(batch):
    gamma = 0.99
    inputs = []
    targets = []
    for series in batch:
        state = series[0].state
        inputs.append(state)
        input = Variable(torch.from_numpy(np.array([series[0].state, series[-1].state], dtype=np.float32)))
        output = cnn(input)
        cumul_reward = 0.0 if series[-1].done else output[1].data.max()
        for step in reversed(series[:-1]):
            cumul_reward = step.reward + gamma * cumul_reward
        target = output[0].data
        target[series[0].action] = cumul_reward
        targets.append(target)
    return torch.from_numpy(np.array(input, dtype = np.float32)), torch.stack(targets)

class MA:
    def __init__(self, size=100):
        self.size =size
        self.list_of_rewards = []
    
    def add(self, rewards):
        self.list_of_rewards += rewards
        while len(self.list_of_rewards) > self.size:
            del self.list_of_rewards[0]
    
    def average(self):
        return np.mean(self.list_of_rewards)

ma = MA(100)

# training
loss = nn.MSELoss()
optimizer = optim.Adam(cnn.parameters(), lr = 0.001)
nb_epochs = 100

for epoch in range(1, nb_epochs+1):
    memory.run_steps(200)
    for batch in memory.sample_batch(128):
        inputs, targets = eligibility_trace(batch)
        inputs = Variable(inputs)
        targets = Variable(targets)
        predictions = cnn(inputs)
        loss_error = loss(predictions, targets)
        optimizer.zero_grad()
        loss_error.backward()
        optimizer.step()
    reward_steps = n_steps.reward_steps()
    ma.add(reward_steps)
    avg_reward = ma.average()
    print("epoch: %s, Arerage reward: %s" % (str(epoch), str(avg_reward)))
    if avg_reward >= 1000:
        print("gagné !!!")
        break


















