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

try:
    lanczos = PIL.Image.Resampling.LANCZOS
    bicubic = PIL.Image.Resampling.BICUBIC
except AttributeError:
    lanczos = PIL.Image.LANCZOS
    bicubic = PIL.Image.BICUBIC

# class MegaScenesAugmented(BaseStereoViewDataset):
#     def __init__(self, *args, data_dir, image_dir, **kwargs):
#         self.data_dir = data_dir
#         self.image_dir = image_dir
#         super().__init__(*args, **kwargs)
#         # assert self.split == 'train'
#         self.loaded_data = self._load_data()
        
#     def _load_data(self):
#         print("Loading image pairs...")
#         with open(osp.join(self.data_dir, f"data_{self.split}", "one_plan_2", "image_pairs.csv"), "r") as f:
#             self.image_pairs = []            
#             reader = csv.reader(f)
#             bad_pairs = [] #self._get_bad_pairs()
#             for row in reader:
#                 _, _, _, plan_name, image_name = row
#                 if len(bad_pairs) > 0:
#                     if (plan_name, image_name) not in bad_pairs:
#                         self.image_pairs.append(row)
#                 else:
#                     self.image_pairs.append(row)
#         print(f"{len(self.image_pairs)} image pairs loaded")

#     def _get_bad_pairs(self):
#         bad_pairs = set()
#         for pair in list(open(os.path.join(self.data_dir, "bad_pairs.txt")).readlines()):
#             plan_path, image_path = pair.replace("\n", "").split(" /")
#             plan_name = plan_path.split("/")[-1]
#             image_name = ("/"+image_path).replace(self.image_dir, "")
#             image_name = "/".join(image_name.split("/")[2:])
#             bad_pairs.add((plan_name, image_name))    
#         return bad_pairs

#     def __len__(self):
#         return (len(self.image_pairs))

#     def __getitem__(self, idx):
#         idx = idx[0] if self.split == "train" else idx
#         i, landmark, comp, plan_name, image_name = self.image_pairs[idx]
#         plan_path = os.path.join(self.image_dir, landmark, "plans", plan_name)
#         image_path = os.path.join(self.image_dir,  landmark, "images", image_name)
#         xys_path = os.path.join(self.data_dir, f"data_{self.split}", "one_plan_2", "coords", f"{int(i):08}.npy")
#         xys = np.load(xys_path)
#         images = load_megascenes_augmented_images(
#             [plan_path, image_path], 
#             size=224, 
#             plan_xys=xys[0],
#             image_xys=xys[1], 
#             verbose=False
#         )
#         view1, view2 = images
#         return view1, view2


class MegaScenesAugmented(BaseStereoViewDataset):
    def __init__(self, *args, data_dir, image_dir, split, **kwargs):
        self.data_dir = data_dir
        self.image_dir = image_dir
        
        super().__init__(*args, **kwargs)
        self.split = split
        # assert self.split == 'train'
        self.loaded_data = self._load_data()
        
    def _load_data(self):
        print("Loading image pairs...")
        print(f"self.split = {self.split}")
        with open(osp.join(self.data_dir, f"data_{self.split}", "image_pairs.csv"), "r") as f:
            self.image_pairs = []            
            reader = csv.reader(f)
            bad_pairs = self._get_bad_pairs()
            for row in reader:
                i, landmark, comp, plan_name, image_name = row
                if (plan_name, image_name) not in bad_pairs and (landmark, comp) not in [("San_Lorenzo_(Genoa)","1"), ("Temple_Church,_London","0")]:
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
    
    def transpose_to_landscape(self, view):
        height, width = view['true_shape']

        if width < height:
            # rectify portrait to landscape
            assert view['img'].shape == (3, height, width)
            view['img'] = view['img'].swapaxes(1, 2)
            view['xys'] = view['xys'][:,[1, 0]]

            # assert view['valid_mask'].shape == (height, width)
            # view['valid_mask'] = view['valid_mask'].swapaxes(0, 1)

            # assert view['depthmap'].shape == (height, width)
            # view['depthmap'] = view['depthmap'].swapaxes(0, 1)

            # assert view['pts3d'].shape == (height, width, 3)
            # view['pts3d'] = view['pts3d'].swapaxes(0, 1)

            # # transpose x and y pixels
            # view['camera_intrinsics'] = view['camera_intrinsics'][[1, 0, 2]]

    def is_good_type(self, key, v):
        """ returns (is_good, err_msg) 
        """
        if isinstance(v, (str, int, tuple)):
            return True, None
        if v.dtype not in (np.float64, np.float32, torch.float32, bool, np.int32, np.int64, np.uint8):
            return False, f"bad {v.dtype=}"
        return True, None

    def _crop_resize_if_necessary(self, image, xys, intrinsics, resolution, is_plan, rng=None, info=None):
        if not isinstance(image, PIL.Image.Image):
            image = PIL.Image.fromarray(image)
        
        H, W = image.size
        if H > 1.1 * W:
            # image is portrait mode
            # print("flipping resolution1")
            resolution = resolution[::-1]
        elif 0.9 < H / W < 1.1 and resolution[0] != resolution[1]:
            # image is square, so we chose (portrait, landscape) randomly
            if rng.integers(2):
                # print("flipping resolution2")
                resolution = resolution[::-1]

        if is_plan:
            max_h = 500
            # im.size = (w, h)
            image_scale = max_h / image.size[1]
            image = image.resize((round(image.size[0] * image_scale), max_h))
        
        resolution = np.array(resolution)
        image_size = np.array(image.size)
        ratio = max(resolution/image_size)
        if ratio < 1:
            interp = PIL.Image.LANCZOS
        else:
            interp = PIL.Image.BICUBIC
        new_size = tuple(int(round(x*ratio)) for x in image.size)
        image = image.resize(new_size, interp)
        W, H = image.size
        cx, cy = W//2, H//2 #ok
        halfh, halfw = resolution//2
        
        image = image.crop((cx-halfw, cy-halfh, cx+halfw, cy+halfh)) # (l, t, r, b)

        offset = np.array([-(cx-halfw), -(cy-halfh)])
        # np.array([0, (size - H_target) // 2]) if W_target > H_target else np.array([(size - W_target) // 2, 0])
        xys = xys * ratio + offset


        intrinsics = (image.size[0] / 2, image.size[1] / 2)
    

        return image, xys, intrinsics

    def _get_views(self, pair_idx, resolution, rng):
        i, landmark, comp, plan_name, image_name = self.image_pairs[pair_idx]
        plan_path = os.path.join(self.image_dir, landmark, "plans", plan_name)
        image_path = os.path.join(self.image_dir,  landmark, "images", image_name)
        xys_path = os.path.join(self.data_dir, f"data_{self.split}", "coords", f"{int(i):06}.npy")
        xys_pair = np.load(xys_path)
        paths = [plan_path, image_path]

        views = []

        for i, (path, xys) in enumerate(zip(paths, xys_pair)):
            if path.endswith(".gif"):
                image = PIL.Image.open(path)
                image.seek(image.n_frames // 2)
                image = image.convert('RGB')
            else:
                image = PIL.Image.open(path).convert('RGB')

            intrinsics = np.array(image.size) / 2
            image, xys, intrinsics = self._crop_resize_if_necessary(image, xys, intrinsics, resolution, is_plan=not (i), rng=rng)

            views.append(dict(
                img=image,
                xys=xys,
                dataset="MegaScenesAugmented",
                resolution=resolution,
                landmark=landmark,
                comp=comp
            ))            

        return views
    
    def __getitem__(self, idx):
        if isinstance(idx, tuple):
            # the idx is specifying the aspect-ratio
            idx, ar_idx = idx
        else:
            assert len(self._resolutions) == 1
            ar_idx = 0

        # set-up the rng
        if self.seed:  # reseed for each __getitem__
            self._rng = np.random.default_rng(seed=self.seed + idx)
        elif not hasattr(self, '_rng'):
            seed = torch.initial_seed()  # this is different for each dataloader process
            self._rng = np.random.default_rng(seed=seed)

        # over-loaded code
        resolution = self._resolutions[ar_idx]  # DO NOT CHANGE THIS (compatible with BatchedRandomSampler)
        views = self._get_views(idx, resolution, self._rng)
        assert len(views) == self.num_views

        # check data-types
        for v, view in enumerate(views):
            view['idx'] = (idx, ar_idx, v)

            # encode the image
            width, height = view['img'].size
            view['true_shape'] = np.int32((height, width))
            view['img'] = self.transform(view['img'])

            # check all datatypes
            for key, val in view.items():
                res, err_msg = self.is_good_type(key, val)
                assert res, f"{err_msg} with {key}={val} for view"

        for view in views:
            # print(view['xys'][:5, ])
            # transpose to make sure all views are the same size
            self.transpose_to_landscape(view)
            # this allows to check whether the RNG is is the same state each time
            view['rng'] = int.from_bytes(self._rng.bytes(4), 'big')
        return views


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
        