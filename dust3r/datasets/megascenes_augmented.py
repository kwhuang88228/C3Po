import csv
import os
import os.path as osp

import dust3r.datasets.utils.cropping as cropping
import numpy as np
import PIL
import torch
import torchvision.transforms as tvf
from dust3r.datasets.base.base_stereo_view_dataset import BaseStereoViewDataset
from dust3r.datasets.utils.transforms import ImgNorm
from dust3r.utils.image import load_megascenes_augmented_images
from scipy import stats


class MegaScenesAugmented(BaseStereoViewDataset):
    def __init__(self, *args, data_dir, image_dir, split, **kwargs):
        self.data_dir = data_dir
        self.image_dir = image_dir
        
        super().__init__(*args, **kwargs)
        self.split = split
        # assert self.split == 'train'
        self.loaded_data = self._load_data()
        
    def _load_data(self):
        bad_landmark_comp_pairs = [
            ("San_Lorenzo_(Genoa)", "1"), 
            ("Temple_Church,_London", "0")
        ]
        print("Loading image pairs...")
        print(f"self.split = {self.split}")
        with open(osp.join(self.data_dir, f"data_{self.split}", "image_pairs.csv"), "r") as f:
            self.image_pairs = []            
            reader = csv.reader(f)
            bad_pairs = self._get_bad_pairs()
            for row in reader:
                i, landmark, comp, plan_name, image_name = row
                if (plan_name, image_name) not in bad_pairs and (landmark, comp) not in bad_landmark_comp_pairs:
                    self.image_pairs.append(row)
                # if int(i) > 5:
                #     break
                # break
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
    
    def transpose_to_landscape(self, view):
        height, width = view['true_shape']

        if width < height:
            # rectify portrait to landscape
            assert view['img'].shape == (3, height, width)
            view['img'] = view['img'].flip(2).transpose(1, 2)
            view['xys'] = np.column_stack((view['xys'][:, 1], -view['xys'][:, 0])) + np.array([0, width])

    def is_good_type(self, key, v):
        """ returns (is_good, err_msg) 
        """
        if isinstance(v, (str, int, tuple)):
            return True, None
        if v.dtype not in (np.float64, np.float32, torch.float32, bool, np.int32, np.int64, np.uint8):
            return False, f"bad {v.dtype=}"
        return True, None

    def _crop_resize_if_necessary(self, image, xys, resolution, is_plan, rng=None, info=None):
        if not isinstance(image, PIL.Image.Image):
            image = PIL.Image.fromarray(image)
        
        H, W = image.size
        if H > 1.1 * W:
            # image is portrait mode
            resolution = resolution[::-1]
        elif 0.9 < H / W < 1.1 and resolution[0] != resolution[1]:
            # image is square, so we chose (portrait, landscape) randomly
            if rng.integers(2):
                resolution = resolution[::-1]

        if is_plan:
            max_h = 500
            # im.size = (w, h)
            image_scale = max_h / image.size[1]
            image = image.resize((round(image.size[0] * image_scale), max_h))
        
        resolution = np.array(resolution[::-1])
        image_size = np.array(image.size)
        ratio = max(resolution/image_size)
        if ratio < 1:
            interp = PIL.Image.LANCZOS
        else:
            interp = PIL.Image.BICUBIC
        new_size = tuple(int(round(x*ratio)) for x in image.size)
        image = image.resize(new_size, interp)
        W, H = image.size
        cx, cy = W//2, H//2 
        # xys_center = self.get_xys_center(xys)
        xys_center = (cx, cy)
        halfw, halfh = resolution//2
        
        l, t, r, b = (xys_center[0]-halfw, xys_center[1]-halfh, xys_center[0]+halfw, xys_center[1]+halfh)
        if l >= 0 and t >= 0 and r < resolution[0] and b < resolution[1]:
            image = image.crop((l, t, r, b))

            offset = np.array([-l, -t])
            xys = xys * ratio + offset
        else:
            image = image.crop((l, t, r, b))

            offset = np.array([-l, -t])
            xys = xys * ratio + offset

        # xys = xys * ratio
        return image, xys

    def _get_views(self, pair_idx, resolution, rng):
        i, landmark, comp, plan_name, image_name = self.image_pairs[pair_idx]
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
        