import torch
import torch.nn as nn


class ModelB(nn.Module):
    def __init__(self, inmodel, base_output_shape = 3172, output_classes=2, fix_inmodel=True):
        super(ModelB, self).__init__()
        self.modelA = inmodel

        if fix_inmodel:
            for param in self.modelA.parameters():
                param.requires_grad = False
        
        self.block1 = nn.Linear(base_output_shape, 50)
        self.block2 = nn.Linear(50, 8)
        self.classifier = nn.Linear(8, output_classes)
        self.head_softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        x = self.modelA(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.classifier(x)
        
        return x, self.head_softmax(x)
