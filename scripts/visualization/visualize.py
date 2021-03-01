import matplotlib.pyplot as plt
import pathlib
import os
import numpy as np
from dtor.viewer import multi_slice_viewer

p_ltp = 1013
abl_ltp = 19
p_nltp = 1002
abl_nltp = 4

o_liver_post = lambda p, a: os.path.join(pathlib.Path(os.environ["DTORROOT"]),
                            f"data/processed/{p}_{a}_liver_post.npy")
o_liver_pre = lambda p, a: os.path.join(pathlib.Path(os.environ["DTORROOT"]),
                           f"data/processed/{p}_{a}_liver_pre.npy")
o_tumor_post = lambda p, a: os.path.join(pathlib.Path(os.environ["DTORROOT"]),
                            f"data/processed/{p}_{a}_tumor_post.npy")

liver_post_pos = np.load(o_liver_post(p_ltp, abl_ltp))
liver_pre_pos = np.load(o_liver_pre(p_ltp, abl_ltp))
tumor_post_pos = np.load(o_tumor_post(p_ltp, abl_ltp))
#
liver_post_neg = np.load(o_liver_post(p_nltp, abl_nltp))
liver_pre_neg = np.load(o_liver_pre(p_nltp, abl_nltp))
tumor_post_neg = np.load(o_tumor_post(p_nltp, abl_nltp))
