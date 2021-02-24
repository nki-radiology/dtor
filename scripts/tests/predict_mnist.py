import numpy as np
import torch
from models.ltp import Model
from dtor.datasets.dataset_mnist import MNIST3DDataset

model = Model(prelim=64)
model.load_state_dict(torch.load("results/model-mnist-fold0.pth", map_location=torch.device('cpu')))
model.eval()

data = MNIST3DDataset(tr_test="test")

for n in range(5):
    f, truth, _ = data[n]
    x = torch.from_numpy(f).unsqueeze(0)
    l, p = model(x)
    pred = torch.argmax(p)
    print(f"True label: {truth}, Model prediction: {pred}")
