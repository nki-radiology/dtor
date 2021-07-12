# -*- coding: utf-8 -*-
"""Process the model choice"""

__author__ = "Sean Benson"
__copyright__ = "MIT"

from models.resnet import generate_model
from models.ltp import Model
from models.modelB import ModelB
from models.modelB_single import ModelBSingle
from models.Unet import UNet, RecombinationBlock
from torchvision.models.video import r3d_18
from dtor.utilities.utils import safe_restore
from cnn_finetune import make_model
import torch.nn as nn
import torch


def model_choice(m_name="nominal", pretrain_loc=None, resume=None, sample=None, pretrained_2d_name=None, 
        depth=101, n_classes=700, fix_inmodel=0):
    assert m_name in ["pretrained_2d", "nominal", "resnet+dense", "unet"]

    if m_name == "nominal":
        model_dry = Model(dry=True)
        prelim = classifier_shape(model_dry, sample)
        model = Model(prelim=prelim)
    elif m_name == "resnet+dense":
        if depth == 18:
            modela = r3d_18(pretrained=True, progress=True)
        else:
            modela = generate_model(depth, n_classes=n_classes)
        if pretrain_loc:
            modela = safe_restore(modela, pretrain_loc)
        modela = nn.Sequential(*list(modela.children())[:-2])
        shape = classifier_shape(modela, sample)
        model = ModelB(modela, base_output_shape=shape, fix_inmodel=fix_inmodel)
    elif m_name == 'pretrained_2d':
        model = make_model(pretrained_2d_name, num_classes=2, pretrained=True, input_size=(214, 214),
                           classifier_factory=make_classifier)
        if fix_inmodel:
            assert isinstance(fix_inmodel, int), "Tell me how many layers to fix"
            for n, l in enumerate(model.children()):
                if n < fix_inmodel:
                    continue
                for param in l.parameters():
                    param.requires_grad = False
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
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    _model.to(device)
    _sample = _sample.to(device)
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
