import numpy as np
import os
import pandas as pd
import pathlib
import SimpleITK as sitk
import argparse
from imutils import paths
from dtor.utilities.utils import image_search, bbox3d, crop3d

parser = argparse.ArgumentParser()
parser.add_argument("--labels", type=str, help="Location of labels", default="data/external/test_ltp/test_ltp.xlsx")
args = parser.parse_args()

labels = os.path.join(pathlib.Path(os.environ["DTORROOT"]), args.labels)
dirs = pd.read_csv(labels)
missed = []

for n in range(len(dirs)):
    patient = dirs.loc[n, 'StudyID'].split("-")[-1]
    abl = str(int(dirs.loc[n, 'Abl_no']))
    folder = os.path.join(pathlib.Path(os.environ["DTORROOT"]),
                          f"data/external/{patient}/{abl}")
    print(f"Processing patient/tumor ID: {patient}/{abl}")
    image_paths = list(paths.list_files(folder, validExts='.gz'))
    if len(image_paths)==0:
        print("Nothing found, maybe investigate...")
        missed.append((patient,abl))
        continue

    # Cropped image locations
    o_liver_post = os.path.join(pathlib.Path(os.environ["DTORROOT"]),
                          f"data/processed/{patient}_{abl}_liver_post.npy")
    o_liver_pre = os.path.join(pathlib.Path(os.environ["DTORROOT"]),
                           f"data/processed/{patient}_{abl}_liver_pre.npy")
    o_tumor_post = os.path.join(pathlib.Path(os.environ["DTORROOT"]),
                           f"data/processed/{patient}_{abl}_tumor_post.npy")
    
    if all([os.path.exists(o_liver_post),
        os.path.exists(o_liver_pre),
        os.path.exists(o_tumor_post)]):
        print("Already processed, continuing...")
        continue
    

    # Find the right image locations
    liver_post = image_search(image_paths, "Post.nii.gz")
    tumor_post = image_search(image_paths, "Post_label.nii.gz")
    liver_pre = image_search(image_paths, "PRE_to_POST_affine.nii.gz")

    # Make liver mask
    a_liver_post = sitk.GetArrayFromImage(sitk.ReadImage(liver_post))
    box = bbox3d(a_liver_post, 0)

    # Crop images
    c_liver_post = crop3d(a_liver_post, box)
    c_tumor_post = crop3d(sitk.GetArrayFromImage(sitk.ReadImage(tumor_post)), box)
    # For the tumor we say that any non-zero point is treated the same
    c_tumor_post = np.where(c_tumor_post>0, 1, 0)
    c_liver_pre = crop3d(sitk.GetArrayFromImage(sitk.ReadImage(liver_pre)), box)


    # Save cropped images
    np.save(o_liver_post, c_liver_post)
    np.save(o_liver_pre, c_liver_pre)
    np.save(o_tumor_post, c_tumor_post)

print("Missed the following:")
print(missed)
