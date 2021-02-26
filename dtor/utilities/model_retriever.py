# -*- coding: utf-8 -*-
"""Process the model choice"""

__author__ = "Sean Benson"
__copyright__ = "MIT"

from models.gen_model import generate_model
from models.ltp import Model
from models.modelB import ModelB
from models.modelB_single import ModelBSingle
from models.Unet import UNet, RecombinationBlock
from dtor.opts import ResNetOptions
from torchvision.models.video import r3d_18
from dtor.utilities.utils import safe_restore


def model_choice(m_name="nominal", pretrain_loc=None, resume=None):
    assert m_name in ["nominal", "resnet18", "resnet18+dense", "nominal_mnist", "unet"]

    if m_name == "nominal":
        model = Model()
    elif m_name == "nominal_mnist":
        model = Model(prelim=64)
    elif m_name == "resnet18":
        opt = ResNetOptions("settings/resnet10.json")
        model, _ = generate_model(opt)
    elif m_name == "resnet18+dense":
        modela = r3d_18(pretrained=True, progress=True)
        model = ModelB(modela, base_output_shape=400)
    elif m_name == "unet":
        modela = UNet(1, [32, 48, 64, 96, 128], 3, net_mode='3d', conv_block=RecombinationBlock)
        if pretrain_loc:
            modela = safe_restore(modela, pretrain_loc)
        model = ModelBSingle(modela)
    else:
        raise NotImplementedError

    if resume:
        model = safe_restore(model, resume)

    return model


def load_model(prefix, fold, model_type="nominal", full_name=None):
    """

    Args:
        prefix: model prefix name
        fold: which fold
        model_type: model structure
        full_name: Use this to load the model if given

    Returns:

    """
    model_name = f"results/model-{prefix}-fold{fold}.pth"
    if full_name:
        model_name = full_name
    print(f"Loading model {model_name}")
    _model = model_choice(model_type)
    _model = safe_restore(_model, model_name)
    return _model
