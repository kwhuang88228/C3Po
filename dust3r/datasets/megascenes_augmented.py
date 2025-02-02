import csv
import os
import sys

import numpy as np
import PIL.Image
import torch
from tqdm import tqdm

sys.path.append(os.path.dirname("/share/phoenix/nfs06/S9/kh775/code/dust3r/dust3r"))
from dust3r.image_pairs import make_pairs
from dust3r.utils.image import load_megascenes_augmented_images
from torch.utils.data import DataLoader, Dataset


class MegaScenesAugmented(Dataset):
    def __init__(self, image_pairs_path, data_dir, npx_dir):
        self.image_pairs = []
        self.load_image_pairs(image_pairs_path)
        self.data_dir = data_dir
        self.npx_dir = npx_dir
        self.max_xys_len = 0
        self.get_max_xys_len()
        
    def load_image_pairs(self, image_pairs_path):
        print("Loading image pairs...")
        with open(image_pairs_path, "r") as f:
            reader = csv.reader(f)
            self.image_pairs = [row for row in reader]
        print(f"{len(self.image_pairs)} image pairs loaded")

    def get_max_xys_len(self):
        print("Calculating max_xys_len...")
        self.max_xys_len = 0
        for file_name in os.listdir(self.npx_dir):
            npx_path = os.path.join(self.npx_dir, file_name)
            xys = np.load(npx_path)
            xys_len = xys[0].shape[0]
            if xys_len > self.max_xys_len:
                self.max_xys_len = xys_len
        print(f"max_xys_len: {self.max_xys_len}")

    def __len__(self):
        return (len(self.image_pairs))

    def __getitem__(self, idx):
        i, landmark, comp, plan_name, image_name = self.image_pairs[idx]
        plan_path = os.path.join(self.data_dir, landmark, "plans", plan_name)
        image_path = os.path.join(self.data_dir,  landmark, "images", image_name)
        xys_path = os.path.join(self.npx_dir, f"{int(i):08}.npy")
        xys = np.load(xys_path)
        images = load_megascenes_augmented_images(
            [plan_path, image_path], 
            size=224, 
            plan_xys=xys[0],
            image_xys=xys[1], 
            max_xys_len=self.max_xys_len,
            verbose=False
        )
        view1, view2 = images
        # pairs = make_pairs(images, scene_graph='complete', prefilter=None, symmetrize=True)
        return view1, view2


if __name__ == "__main__":
    image_pairs_path = "/share/phoenix/nfs06/S9/kh775/code/wsfm/scripts/data/keypoint_localization/data_train/image_pairs.csv"
    data_dir = "/share/phoenix/nfs06/S9/kh775/dataset/megascenes_augmented_exhaustive"
    coords_dir = "/share/phoenix/nfs06/S9/kh775/code/wsfm/scripts/data/keypoint_localization/data_train/coords"
    dataset = MegaScenesAugmented(image_pairs_path, data_dir, coords_dir)

    dataloader = DataLoader(dataset)
    max_s = 0
    for view1, view2 in tqdm(dataloader):
        # print(view1, view2)
        # print(view1["img"].size(), view2["img"].size())
        _, s, _ = view1["plan_xys"].size()
        
        if s > max_s:
            max_s = s
    
    print(max_s)
        