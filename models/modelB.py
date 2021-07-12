import torch
import torch.nn as nn


class ModelB(nn.Module):
    def __init__(self, inmodel, base_output_shape=3172, output_classes=2, fix_inmodel=14):
        super(ModelB, self).__init__()
        self.modelA = inmodel

        if fix_inmodel:
            assert isinstance(fix_inmodel,int), "Tell me how many layers to fix"
            for n, l in enumerate(self.modelA.children()):
                if n<fix_inmodel:
                    for param in l.parameters():
                        param.requires_grad = False
                else:
                    continue

        self.classifier = nn.Linear(base_output_shape, output_classes)
        self.head_softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        x = self.modelA(x)
        conv_flat = x.view(
            x.size(0),
            -1,
        )
        x = self.classifier(conv_flat)
        
        return x, self.head_softmax(x)
