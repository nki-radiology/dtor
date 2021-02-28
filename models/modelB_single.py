import torch
import torch.nn as nn


class ModelBSingle(nn.Module):
    def __init__(self, inmodel, base_output_shape=1806336, output_classes=2, fix_inmodel=True):
        super(ModelBSingle, self).__init__()
        self.modelA = inmodel

        if fix_inmodel:
            for param in self.modelA.parameters():
                param.requires_grad = False
        
        self.block1 = nn.Linear(base_output_shape, 50)
        self.classifier = nn.Linear(50, output_classes)
        self.head_softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        processed = []
        for i in range(x.size(1)):
            processed.append(self.modelA(x[:, i, :, :, :].unsqueeze(dim=1)))
        x = torch.cat(processed, dim=1)

        conv_flat = x.view(
            x.size(0),
            -1,
        )
        #print(conv_flat.shape)
        x = self.block1(conv_flat)
        x = self.classifier(x)
        
        return x, self.head_softmax(x)
