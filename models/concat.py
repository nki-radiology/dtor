import torch
import torch.nn as nn
import torch.nn.Functional as F


class Concat(nn.Module):
    def __init__(self, inmodel, n_extra, n_output):
        super(Concat, self).__init__()
        self.cnn = inmodel
        self.cnn.fc = nn.Linear(
            self.cnn.fc.in_features, 20)
        
        self.fc1 = nn.Linear(20 + n_extra, (20 + n_extra ) * 2)
        self.fc2 = nn.Linear((20 + n_extra ) * 2, n_output)
        
    def forward(self, image, data):
        x1 = self.cnn(image)
        x2 = data
        
        x = torch.cat((x1, x2), dim=1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x