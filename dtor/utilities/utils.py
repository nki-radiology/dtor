# -*- coding: utf-8 -*-
"""Utilities"""

__author__ = "Sean Benson"
__copyright__ = "MIT"

import numpy as np
from numpy.lib import stride_tricks
from skimage.util.shape import view_as_windows
import torch
from torchvision import transforms
import matplotlib.pylab as pylab
import PIL


def image_search(inlist, filename, debug=False):
    """

    Args:
        inlist: List of filepaths
        filename: Suffix to match to
        debug: Extra info required?

    Returns: Single filepath

    """
    if debug:
        print(inlist)
        print(filename)
    im = [i for i in inlist if i.split("/")[-1] == filename]
    assert len(im) == 1, f"Match is not unique: {im}"
    return im[0]


def bbox3d(img, _min=0, _buffer=0):
    """

    Args:
        img: Input numpy array
        _min: Minimum threshold

    Returns: Indices of bounding box

    """

    r = np.any(img, axis=(1, 2))
    c = np.any(img, axis=(0, 2))
    z = np.any(img, axis=(0, 1))

    rmin, rmax = np.where(r > _min)[0][[0, -1]]
    cmin, cmax = np.where(c > _min)[0][[0, -1]]
    zmin, zmax = np.where(z > _min)[0][[0, -1]]

    _rm = img.shape[0]-_buffer-1
    _cm = img.shape[1]-_buffer-1
    _zm = img.shape[2]-_buffer-1
    print(f"Image shape: {img.shape}")
    
    return max(rmin, _buffer+1)-_buffer, min(rmax, _rm)+_buffer, max(cmin, _buffer+1)-_buffer, min(cmax, _cm)+_buffer, max(zmin, _buffer+1)-_buffer, max(zmax, _zm)+_buffer


def crop3d(in3d, box):
    """

    Args:
        in3d: 3D numpy array
        box: Box boundary indices

    Returns:

    """
    assert len(box) == 6, "Require box=[rmin, rmax, cmin, cmax, zmin, zmax]"
    cropped = in3d[box[0]:box[1], box[2]:box[3], box[4]:box[5]]
    return cropped


# Taken from https://github.com/MIC-DKFZ/medicaldetectiontoolkit/blob/master/utils/dataloader_utils.py
def pad_nd_image(
    image,
    new_shape=None,
    mode="edge",
    kwargs=None,
    return_slicer=False,
    shape_must_be_divisible_by=None,
):
    """
    one padder to pad them all. Documentation? Well okay. A little bit. by Fabian Isensee

    :param image: nd image. can be anything
    :param new_shape: what shape do you want? new_shape does not have to have the same dimensionality as image. If
    len(new_shape) < len(image.shape) then the last axes of image will be padded. If new_shape < image.shape in any of
    the axes then we will not pad that axis, but also not crop! (interpret new_shape as new_min_shape)
    Example:
    image.shape = (10, 1, 512, 512); new_shape = (768, 768) -> result: (10, 1, 768, 768). Cool, huh?
    image.shape = (10, 1, 512, 512); new_shape = (364, 768) -> result: (10, 1, 512, 768).

    :param mode: see np.pad for documentation
    :param return_slicer: if True then this function will also return what coords you will need to use when cropping back
    to original shape
    :param shape_must_be_divisible_by: for network prediction. After applying new_shape, make sure the new shape is
    divisibly by that number (can also be a list with an entry for each axis). Whatever is missing to match that will
    be padded (so the result may be larger than new_shape if shape_must_be_divisible_by is not None)
    :param kwargs: see np.pad for documentation
    """
    if kwargs is None:
        kwargs = {}

    if new_shape is not None:
        old_shape = np.array(image.shape[-len(new_shape) :])
    else:
        assert shape_must_be_divisible_by is not None
        assert isinstance(shape_must_be_divisible_by, (list, tuple, np.ndarray))
        new_shape = image.shape[-len(shape_must_be_divisible_by) :]
        old_shape = new_shape

    num_axes_nopad = len(image.shape) - len(new_shape)

    new_shape = [max(new_shape[i], old_shape[i]) for i in range(len(new_shape))]

    if not isinstance(new_shape, np.ndarray):
        new_shape = np.array(new_shape)

    if shape_must_be_divisible_by is not None:
        if not isinstance(shape_must_be_divisible_by, (list, tuple, np.ndarray)):
            shape_must_be_divisible_by = [shape_must_be_divisible_by] * len(new_shape)
        else:
            assert len(shape_must_be_divisible_by) == len(new_shape)

        for i in range(len(new_shape)):
            if new_shape[i] % shape_must_be_divisible_by[i] == 0:
                new_shape[i] -= shape_must_be_divisible_by[i]

        new_shape = np.array(
            [
                new_shape[i]
                + shape_must_be_divisible_by[i]
                - new_shape[i] % shape_must_be_divisible_by[i]
                for i in range(len(new_shape))
            ]
        )

    difference = new_shape - old_shape
    pad_below = difference // 2
    pad_above = difference // 2 + difference % 2
    pad_list = [[0, 0]] * num_axes_nopad + list(
        [list(i) for i in zip(pad_below, pad_above)]
    )
    res = np.pad(image, pad_list, mode, **kwargs)
    if not return_slicer:
        return res
    else:
        pad_list = np.array(pad_list)
        pad_list[:, 1] = np.array(res.shape) - pad_list[:, 1]
        slicer = list(slice(*i) for i in pad_list)
        return res, slicer


def cutup(data, blck, strd):
    """

    Args:
        data: numpy array
        blck: size of the blocks required
        strd: step size

    Returns: array of blocks

    """
    sh = np.array(data.shape)
    blck = np.asanyarray(blck)
    strd = np.asanyarray(strd)
    nbl = (sh - blck) // strd + 1
    strides = np.r_[data.strides * strd, data.strides]
    dims = np.r_[nbl, blck]
    data6 = stride_tricks.as_strided(data, strides=strides, shape=dims)
    return data6


def expand_image(_img, block, stride, deform=True):
    """

    Args:
        _img: numpy array
        block: size of the blocks required
        stride: step size

    Returns: array of blocks

    """
    if deform:
        
        ims_Z=np.zeros([_img.shape[0],block[1],block[0]])
        f_img=np.zeros([block[2],block[1],block[0]])
    
        for z in range(0,_img.shape[0]):
    
            ims_Z[z,:,:]=cv2.resize(_img[z,:,:], (block[0],block[1]))

        for x in range(0,ims_Z.shape[2]):
    
            f_img[:,:,x]=cv2.resize(ims_Z[:,:,x], (block[1],block[2]))
    else:
        
        to_pad = []
        pad = False
        for i in range(len(_img.shape)):
            if _img.shape[i] < block[i]:
                pad = True
                to_pad.append(block[i])
            else:
                to_pad.append(_img.shape[i])
        if pad:
            print(f"Enttire image must be padded: {_img.shape}, must be padded")
            _img = pad_nd_image(_img, new_shape=to_pad)
        a_img = view_as_windows(_img, block, step=stride)
        f_img = a_img.reshape(-1, *a_img.shape[-3:])
    # Make sure blocks are padded
        for s in f_img:
            if s.shape != block:
                print(f"Shape: {s.shape}, must be padded to match: {block}")
                s = pad_nd_image(s, new_shape=block)
                assert s.shape == block, "Padding failed"
    return f_img



def find_folds(_df):
    cols = list(_df.columns.values)
    folds = [f for f in cols if "fold" in f]
    return len(folds)


def set_plt_config():
    params = {'legend.fontsize': 'x-large',
              'figure.figsize': (15, 5),
              'axes.labelsize': 'x-large',
              'axes.titlesize': 'x-large',
              'xtick.labelsize': 'x-large',
              'ytick.labelsize': 'x-large'}
    pylab.rcParams.update(params)


def get_class_distribution_loaders(dataloader_obj, labels):
    count_dict = {k: 0 for k in labels}
    for _, j, _ in dataloader_obj:
        y_idx = j.item()
        count_dict[str(y_idx)] += 1
    return count_dict


def get_class_weights(dset, sample_frac=0.2, labels=['0', '1']):
    """

    Args:
        dset: torch dataset object
        sample_frac: how much of the dataset do you want to use for the weight calculation
        labels: what are the outputs

    Returns: torch 1D tensor with class weights in order

    """
    dataset_size = len(dset)
    dataset_indices = list(range(dataset_size))
    np.random.shuffle(dataset_indices)
    split_index = int(np.floor(sample_frac * dataset_size))
    idx = dataset_indices[:split_index]
    sampler = torch.utils.data.SubsetRandomSampler(idx)
    loader = torch.utils.data.DataLoader(dataset=dset, shuffle=False, batch_size=1, sampler=sampler)

    c_dict = get_class_distribution_loaders(loader, labels)
    f_dict = dict()
    for k, c in c_dict.items():
        f_dict[k] = c/len(idx)
    out_weight = [1.0 for k in c_dict.keys()]
    for k, f in f_dict.items():
        out_weight[int(k)] = 1.0/f
    print(f"Calculated class weights as {out_weight}")
    return torch.FloatTensor(out_weight)


def safe_restore(_model, state_loc):
    """

    Args:
        _model: Input model object
        state_loc: Location of the state dict

    Returns: updated model

    """

    try:
        _model.load_state_dict(torch.load(state_loc,
                                          map_location=torch.device('cuda' if torch.cuda.is_available() else "cpu")))
    except RuntimeError:
        _d = torch.load(state_loc,  map_location=torch.device('cuda' if torch.cuda.is_available() else "cpu"))
        if 'net' in list(_d.keys()):
            _d = _d['net']
        for k in list(_d.keys()):
            newkey = '.'.join(k.split('.')[1:])
            print(f"{k} becomes {newkey}")
            _d[newkey] = _d[k]
            _d.pop(k, None)
        _model.load_state_dict(_d)
    return _model


class Stats(PIL.ImageStat.Stat):
    def __add__(self, other):
        return Stats(list(map(np.add, self.h, other.h)))


def find_mean_std(dl):
    statistics = None
    to_pil = transforms.ToPILImage()

    for data, _ in dl:
        for b in range(data.shape[0]):
            for _channel in range(data.shape[1]):
                for _slice in range(data.shape[2]):
                    _image = data[b][_channel][_slice]
                    if statistics is None:
                        statistics = Stats(to_pil(_image))
                    else:
                        statistics += Stats(to_pil(_image))
    print(f'mean:{statistics.mean}, std:{statistics.stddev}')
