# -*- coding: utf-8 -*-
"""CT dataset"""

__author__ = "Sean Benson"
__copyright__ = "MIT"

import os
from torch.utils.data import Dataset
import pathlib
import numpy as np
from dtor.utilities.utils import cutup, pad_nd_image, expand_image
from dtor.utilities.utils import bbox3d, crop3d
from tqdm import tqdm
from tqdm.contrib import tzip
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

npy_name = lambda x, p, t: os.path.join(pathlib.Path(os.environ["DTORROOT"]),
                                f"data/processed/{p}_{t}_{x}.npy")
sub_name = lambda n, p, t: os.path.join(pathlib.Path(os.environ["DTORROOT"]),
                                f"data/cache/{p}_{t}_{n}.npy")


class CTImageDataset(Dataset):
    """Dataset from CT images."""

    def __init__(self, label_file=None, # create
                            root_dir=os.environ["DTORROOT"], # create
                            shape=(64,64,32), # create
                            stride=(48, 48, 24), # create
                            buffer=10, # create
                            tot_folds=3, # create
                            label="L_LTP_date", # create
                            chunked_csv=None, # use
                            fold=None, # use
                            tr_test=None, # use
                            transform=None # use
                            ):
        """
        Initialization
        Args:
            label_file: File with the image locations
            root_dir: Package location
            shape: How big do we want our 3d chunks to be
            stride: Overlap
            buffer: Buffer around the ablation zone
            tot_folds: Number of folds in our K-fold
            label: Output training label
            chunked_csv: Where are our chunked images
            fold: Which fold to return
            tr_test: Test or train
            transform: Any augmentations needed or preprocessing steps
        """

        self.root_dir = root_dir
        self.transform = transform
        self.buffer = buffer
        if chunked_csv:
            self.chunked_images = pd.read_csv(chunked_csv, sep="\t")
            # Restrict by fold + train/test
            self.chunked_images = self.chunked_images[self.chunked_images[f"fold_{fold}"]==tr_test]
            self.chunked_images.index = range(len(self.chunked_images))
        else:
            # Make the chunked images and add the folds info
            try:
                self.images_frame = pd.read_excel(label_file, engine='openpyxl')
            except:
                self.images_frame = pd.read_csv(label_file)
            #
            cols = list(self.images_frame.columns.values)
            cols = [c for c in cols if "Unnamed" not in c]
            self.images_frame = self.images_frame[cols].dropna()
            self.images_frame.drop_duplicates(subset=["StudyID", "Abl_no"], inplace=True)
            self.images_frame.index = range(len(self.images_frame))
            #
            self.chunked_images = self.expand_points(shape, stride, label, tot_folds)

    def __len__(self):
        return len(self.chunked_images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        fname = self.chunked_images.loc[idx, 'filename']
        label = self.chunked_images.loc[idx, 'label']
        weight = self.chunked_images.loc[idx, 'weight']
        image = np.load(fname)
        image = torch.from_numpy(image).to(torch.float32)

        if self.transform:
            image = self.transform(image)

        sample = [image, label, (weight, fname)]

        return sample

    def expand_points(self, shape=(64, 64, 32), stride=(48, 48, 24), label="LTP", tot_folds=1):
        d_data = dict()
        d_data["patient"] = []
        d_data["abl"] = []
        d_data["label"] = []
        d_data["weight"] = []
        d_data["filename"] = []
        d_data["data_point_id"] = []
        for n in range(tot_folds):
            d_data[f"fold_{n}"] = []

        # K-fold
        print(f"Generating {tot_folds} folds, from dataset:")
        print(self.images_frame)

        splits_frame = self.images_frame.drop_duplicates(subset=["StudyID"])
        _kf = StratifiedKFold(n_splits=tot_folds, random_state=42, shuffle=True)
        for f, (train, test) in enumerate(_kf.split(splits_frame, splits_frame[label])):
            colname = f"fold_{f}"
            for n in range(len(self.images_frame)):
                self.images_frame[colname] = self.images_frame.apply(lambda x: "train" if int(x.name) in train else "test",
                                                                 axis=1)

        # Create blocks for network input
        point_counter = 0
        print(f"Generating data for {len(self.images_frame)} ablations")
        for n1 in tqdm(range(len(self.images_frame))):
            try:
                patient = self.images_frame.loc[n1, 'StudyID'].split("-")[-1]
            except ValueError:
                # Catches when we reach the end of the excel file but there are still empty rows after
                continue
            abl = str(int(self.images_frame.loc[n1, 'Abl_no']))
            _label = str(int(self.images_frame.loc[n1, label]))
            print(f"Processing patient/abl ID: {patient}/{abl}")

            # Load in cropped arrays
            o_liver_post = npy_name("liver_post", patient, abl)
            o_liver_pre = npy_name("liver_pre", patient, abl)
            o_tumor_post = npy_name("tumor_post", patient, abl)
            #
            if not os.path.exists(o_liver_post):
                print("Files not found, skipping")
                continue

            liver_post = np.load(o_liver_post)
            liver_pre = np.load(o_liver_pre)
            tumor_post = np.load(o_tumor_post)

            # Crop around each ablation zone
            a = int(self.images_frame.loc[n1, 'L_Label'])

            c_tumor_post = np.where(tumor_post == a, 1, 0)
            a_box = bbox3d(c_tumor_post, _buffer=self.buffer)
            cc_tumor_post = crop3d(c_tumor_post, a_box)
            cc_liver_post = crop3d(liver_post, a_box)
            cc_liver_pre = crop3d(liver_pre, a_box)

            print(f"Final cropped shape is {cc_liver_post.shape}")
            # Generate our (padded if necessary) chunks
            try:
                l_liver_post = expand_image(cc_liver_post, shape, stride)
                l_liver_pre = expand_image(cc_liver_pre, shape, stride)
                l_tumor_post = expand_image(cc_tumor_post, shape, stride)
            except ValueError:
                print("Expand failed, likely cropped too small")
                continue

            # Fill in the target dataframe
            d_tmp = dict()
            for f in range(tot_folds):
                d_tmp[f"fold_{f}"] = self.images_frame.loc[n1, f"fold_{f}"]

            for _l in tzip(l_liver_post, l_liver_pre, l_tumor_post):
                f_post, f_pre, f_tumor = _l
                if any([np.sum(f_post) < 0.1, np.sum(f_pre) < 0.1, np.sum(f_tumor) < 1.0]):
                    continue
                d_data["patient"].append(patient)
                d_data["abl"].append(abl)
                d_data["label"].append(_label)
                #
                d_data["weight"].append(1.0+(np.sum(f_tumor)/f_tumor.size))
                #
                d_data["data_point_id"].append(point_counter)
                #
                for f in range(tot_folds):
                    d_data[f"fold_{f}"].append(d_tmp[f"fold_{f}"])

                fname = sub_name(point_counter, patient, abl)
                data = np.stack((f_post, f_pre, f_tumor), axis=-1)
                data = np.moveaxis(data, -1, 0)
                np.save(fname, data)
                d_data["filename"].append(fname)
                point_counter += 1

        # Make dataframe from dictionary
        df = pd.DataFrame.from_dict(d_data)
        df.set_index("data_point_id")
        df_chunked_fname = os.path.join(pathlib.Path(os.environ["DTORROOT"]), f"data/chunked.csv")
        df.to_csv(df_chunked_fname, sep="\t")
        return df
