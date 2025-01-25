import json
import os
import sys

import numpy as np
import PIL.Image
import torch

sys.path.append(os.path.dirname("/share/phoenix/nfs06/S9/kh775/code/dust3r/dust3r"))
from dust3r.image_pairs import make_pairs
from dust3r.utils.image import load_megascenes_augmented_images
from torch.utils.data import DataLoader, Dataset


class MegaScenesAugmented(Dataset):
    def __init__(self, json_dir, data_dir):
        self.json_dir = json_dir
        self.data_dir = data_dir
        self.mappings = self.load_mappings()

        self.plan_paths = []
        self.image_paths = []
        self.plan_xys_list = []
        self.image_xys_list = []
        self.load_paths()
        
    def load_mappings(self):
        mappings = dict()

        for json_file in sorted(os.listdir(self.json_dir)):
            if json_file.endswith(".json"):
                json_file_path = os.path.join(self.json_dir, json_file)

                with open(json_file_path, "r") as f:
                    data = json.load(f)

                    for landmark in data:
                        if landmark not in mappings:
                            mappings[landmark] = data[landmark]
                        else:
                            mappings[landmark].update(data[landmark])

        return mappings

    def load_paths(self):
        for landmark in sorted(self.mappings):
            for comp in sorted(self.mappings[landmark]):
                for plan in sorted(self.mappings[landmark][comp]):
                    for image_id in sorted(self.mappings[landmark][comp][plan]):
                        image_info = self.mappings[landmark][comp][plan][image_id]
                        plan_name = image_info["plan_name"]
                        image_name = image_info["image_name"]
                        plan_path = os.path.join(self.data_dir, landmark, "plans", plan_name)
                        image_path = os.path.join(self.data_dir, landmark, "images", image_name)

                        plan_xys = image_info["plan_xy"]
                        image_xys = image_info["image_xy"]
                        assert len(plan_xys) == len(image_xys)
                        
                        for i in range(len(plan_xys)):
                            plan_xy = plan_xys[i]
                            image_xy = image_xys[i]
                        
                            self.plan_paths.append(plan_path)
                            self.image_paths.append(image_path)
                            self.plan_xys_list.append(plan_xy)
                            self.image_xys_list.append(image_xy)

    def __len__(self):
        assert len(self.plan_paths) == len(self.image_paths)
        assert len(self.plan_xys_list) == len(self.image_xys_list)

        return (len(self.plan_paths))

    def __getitem__(self, idx):
        plan_xy = self.plan_xys_list[idx]
        image_xy = self.image_xys_list[idx]
        images = load_megascenes_augmented_images(
            [self.plan_paths[idx], self.image_paths[idx]], 
            size=224, 
            plan_xy=plan_xy,
            image_xy=image_xy
        )
        view1, view2 = images
        # pairs = make_pairs(images, scene_graph='complete', prefilter=None, symmetrize=True)
        return view1, view2


if __name__ == "__main__":
    data_dir = "/share/phoenix/nfs06/S9/kh775/dataset/megascenes_augmented_exhaustive"
    json_dir = "/share/phoenix/nfs06/S9/kh775/code/wsfm/scripts/data/keypoint_localization/data_test"
    dataset = MegaScenesAugmented(json_dir, data_dir)
    print(dataset[0][0]["img"].size())
    print(dataset[0][0]["plan_xy"])

    dataloader = DataLoader(dataset)
    for view1, view2 in dataloader:
        # print(view1, view2)
        print(view1["img"].size())
        print(view1["plan_xy"].size())
        break
