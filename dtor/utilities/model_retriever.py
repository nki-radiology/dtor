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
from cnn_finetune import make_model
import torch.nn as nn


def model_choice(m_name="nominal", pretrain_loc=None, resume=None, sample=None, pretrained_2d_name=None):
    assert m_name in ["pretrained_2d", "nominal", "resnet18", "resnet18+dense", "unet"]

    if m_name == "nominal":
        model_dry = Model(dry=True)
        prelim = classifier_shape(model_dry, sample)
        model = Model(prelim=prelim)
    elif m_name == "resnet18":
        opt = ResNetOptions("settings/resnet10.json")
        model, _ = generate_model(opt)
    elif m_name == "resnet18+dense":
        modela = r3d_18(pretrained=True, progress=True)
        model = ModelB(modela, base_output_shape=classifier_shape(modela, sample))
    elif m_name == 'pretrained_2d':
        model = make_model(pretrained_2d_name, num_classes=2, pretrained=True, input_size=(214, 214),
                           classifier_factory=make_classifier)
    elif m_name == "unet":
        modela = UNet(1, [32, 48, 64, 96, 128], 3, net_mode='3d', conv_block=RecombinationBlock)
        if pretrain_loc:
            modela = safe_restore(modela, pretrain_loc)
        model = ModelBSingle(modela,
                             base_output_shape=classifier_shape(modela, sample[:, 0, :, :, :].unsqueeze(dim=1)))
    else:
        raise NotImplementedError

    if resume:
        model = safe_restore(model, resume)

    return model


def load_model(prefix, fold, model_type="nominal", full_name=None, sample=None):
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
    _model = model_choice(model_type, sample=sample)
    _model = safe_restore(_model, model_name)
    return _model


def classifier_shape(_model, _sample):
    x = _model(_sample)
    conv_flat = x.view(
        x.size(0),
        -1,
    )
    return conv_flat.shape[1]


def make_classifier(in_features, num_classes):
    return nn.Sequential(
        nn.Linear(in_features, 4096),
        nn.ReLU(inplace=True),
        nn.Linear(4096, num_classes),
    )
