import csv
import os
import os.path as osp

import numpy as np
from dust3r.datasets.base.base_stereo_view_dataset import BaseStereoViewDataset
# sys.path.append(os.path.dirname("/share/phoenix/nfs06/S9/kh775/code/dust3r/dust3r"))
from dust3r.utils.image import load_megascenes_augmented_images


class MegaScenesAugmented(BaseStereoViewDataset):
    def __init__(self, *args, data_dir, image_dir, **kwargs):
        self.data_dir = data_dir
        self.image_dir = image_dir
        super().__init__(*args, **kwargs)
        # assert self.split == 'train'
        self.loaded_data = self._load_data()
        
    def _load_data(self):
        bad_landmark_comp_pairs = [
            ("San_Lorenzo_(Genoa)", "1"), 
            ("Temple_Church,_London", "0")
        ]
        print("Loading image pairs...")
        with open(osp.join(self.data_dir, f"data_{self.split}", "image_pairs.csv"), "r") as f:
            self.image_pairs = []            
            reader = csv.reader(f)
            bad_pairs = self._get_bad_pairs()
            for row in reader:
                i, landmark, comp, plan_name, image_name = row
                if (plan_name, image_name) not in bad_pairs and (landmark, comp) not in bad_landmark_comp_pairs:
                    self.image_pairs.append(row)
        print(f"{len(self.image_pairs)} image pairs loaded")

    def _get_bad_pairs(self):
        bad_pairs = set()
        for pair in list(open(os.path.join(self.data_dir, "bad_pairs.txt")).readlines()):
            plan_path, image_path = pair.replace("\n", "").split(" /")
            plan_name = plan_path.split("/")[-1]
            image_name = ("/"+image_path).replace(self.image_dir, "")
            image_name = "/".join(image_name.split("/")[2:])
            bad_pairs.add((plan_name, image_name))    
        return bad_pairs

    def __len__(self):
        return (len(self.image_pairs))

    def __getitem__(self, idx):
        idx = idx[0] if self.split == "train" else idx
        i, landmark, comp, plan_name, image_name = self.image_pairs[idx]
        plan_path = os.path.join(self.image_dir, landmark, "plans", plan_name)
        image_path = os.path.join(self.image_dir,  landmark, "images", image_name)
        xys_path = os.path.join(self.data_dir, f"data_{self.split}", "coords", f"{int(i):06}.npy")
        xys = np.load(xys_path)
        size = self._resolutions[0][0]
        images = load_megascenes_augmented_images(
            [plan_path, image_path], 
            size=size, 
            plan_xys=xys[0],
            image_xys=xys[1], 
            verbose=False
        )
        view1, view2 = images
        return view1, view2


if __name__ == "__main__":
    image_pairs_path = "/share/phoenix/nfs06/S9/kh775/code/wsfm/scripts/data/keypoint_localization/data_test/one_plan_2/image_pairs.csv"
    data_dir = "/share/phoenix/nfs06/S9/kh775/dataset/megascenes_augmented_exhaustive"
    coords_dir = "/share/phoenix/nfs06/S9/kh775/code/wsfm/scripts/data/keypoint_localization/data_test/one_plan_2/coords"
    dataset = MegaScenesAugmented(image_pairs_path, data_dir, coords_dir)
    print(dataset[0][0].keys())  # dict_keys(['img', 'true_shape', 'idx', 'instance', 'plan_xys', 'image_xys'])
    print(dataset[0][0]["img"].shape, dataset[0][0]["plan_xys"].shape)

    # dataloader = DataLoader(dataset)
    # max_s = 0
    # for view1, view2 in tqdm(dataloader):
    #     # print(view1, view2)
    #     # print(view1["img"].size(), view2["img"].size())
    #     _, s, _ = view1["plan_xys"].size()
        
    #     if s > max_s:
    #         max_s = s
    
    # print(max_s)
        