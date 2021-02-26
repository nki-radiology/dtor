import torch
import torch.nn as nn


class ModelBSingle(nn.Module):
    def __init__(self, inmodel, base_output_shape=6425228, output_classes=2, fix_inmodel=True):
        super(ModelBSingle, self).__init__()
        self.modelA = inmodel

        if fix_inmodel:
            for param in self.modelA.parameters():
                param.requires_grad = False
        
        self.block1 = nn.Linear(base_output_shape, 50)
        self.block2 = nn.Linear(50, 8)
        self.classifier = nn.Linear(8, output_classes)
        self.head_softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        processed = []
        for i in range(x.size(1)):
            processed.append(self.modelA(x[:, i, :, :, :]))
        x = torch.cat(processed, dim=1)
        x = self.block1(x)
        x = self.block2(x)
        x = self.classifier(x)
        
        return x, self.head_softmax(x)
