import torch
import torch.nn as nn



class NN (nn.Module):
    def __init__(self):
        super(NN, self).__init__()
        self.actions_size = 3
        self.states_size = 2
        self.hidden1 = 100
        # fully connected layer
        self.fc1 = nn.Linear(self.states_size, self.hidden1, bias=False)
        self.fc2 = nn.Linear(self.hidden1, self.actions_size, bias=False)

        #xavier weight initialisation: worse performance with optimal reward function
        # torch.nn.init.xavier_uniform(self.fc1.weight)
        # torch.nn.init.xavier_uniform(self.fc2.weight)


    def forward (self, input):
        model = torch.nn.Sequential (
            self.fc1,
            self.fc2
        )
        return model(input)

