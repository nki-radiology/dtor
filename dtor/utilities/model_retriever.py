# -*- coding: utf-8 -*-
"""Process the model choice"""

__author__ = "Sean Benson"
__copyright__ = "MIT"

from models.gen_model import generate_model
from models.ltp import Model
from models.modelB import ModelB
from dtor.opts import ResNetOptions
from torchvision.models.video import r3d_18


def model_choice(m_name="nominal"):
    assert m_name in ["nominal", "resnet18", "resnet18+dense", "nominal_mnist"]

    if m_name == "nominal":
        return Model()
    elif m_name == "nominal_mnist":
        return Model(prelim=64)
    elif m_name == "resnet18":
        opt = ResNetOptions("settings/resnet10.json")
        model, _ = generate_model(opt)
        return model
    elif m_name == "resnet18+dense":
        modela = r3d_18(pretrained=True, progress=True)
        model = ModelB(modela, base_output_shape=400)
        return model
    else:
        raise NotImplementedError
