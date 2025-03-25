import csv
import os
import os.path as osp

import numpy as np
from dust3r.datasets.base.base_stereo_view_dataset import BaseStereoViewDataset
# sys.path.append(os.path.dirname("/share/phoenix/nfs06/S9/kh775/code/dust3r/dust3r"))
from dust3r.utils.image import load_megascenes_augmented_images

# class MegaScenesAugmented(BaseStereoViewDataset):
#     def __init__(self, *args, data_dir, image_dir, **kwargs):
#         self.data_dir = data_dir
#         self.image_dir = image_dir
#         super().__init__(*args, **kwargs)
#         # assert self.split == 'train'
#         self.loaded_data = self._load_data()
        
#     def _load_data(self):
#         bad_landmark_comp_pairs = [
#             ("San_Lorenzo_(Genoa)", "1"), 
#             ("Temple_Church,_London", "0")
#         ]
#         print("Loading image pairs...")
#         # /share/phoenix/nfs06/S9/kh775/code/wsfm/scripts/data/keypoint_localization/old/data_train/one_plan_2/image_pairs.csv
#         with open(osp.join(self.data_dir, f"data_{self.split}", "one_plan_2", "image_pairs.csv"), "r") as f:
#             self.image_pairs = []            
#             reader = csv.reader(f)
#             # bad_pairs = self._get_bad_pairs()
#             for row in reader:
#                 i, landmark, comp, plan_name, image_name = row
#                 # if (plan_name, image_name) not in bad_pairs and (landmark, comp) not in bad_landmark_comp_pairs:
#                 self.image_pairs.append(row)
#         print(f"{len(self.image_pairs)} image pairs loaded")

#     def _get_bad_pairs(self):
#         bad_pairs = set()
#         for file_name in ["corrupted_pairs.txt", "oob_pairs_test.txt", "oob_pairs_train.txt"]:
#             file_path = os.path.join(self.data_dir, file_name)
#             for pair in list(open(file_path).readlines()):
#                 plan_path, image_path = pair.replace("\n", "").split(" /")
#                 plan_name = plan_path.split("/")[-1]
#                 image_name = ("/"+image_path).replace(self.image_dir, "")
#                 image_name = "/".join(image_name.split("/")[2:])
#                 bad_pairs.add((plan_name, image_name))    
#             print(f"Loaded {len(list(open(file_path).readlines()))} bad pairs from {file_name}")
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
#         size = self._resolutions[0][0]
#         images = load_megascenes_augmented_images(
#             [plan_path, image_path], 
#             size=size, 
#             plan_xys=xys[0],
#             image_xys=xys[1], 
#             verbose=False
#         )
#         view1, view2 = images
#         return view1, view2
# CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node 1 train.py \
#     --train_dataset="MegaScenesAugmented(data_dir='/share/phoenix/nfs06/S9/kh775/code/wsfm/scripts/data/keypoint_localization/old/', image_dir='/share/phoenix/nfs06/S9/kh775/dataset/megascenes_augmented_exhaustive/', split='train', resolution=[(512, 512)])" \
#     --test_dataset="MegaScenesAugmented(data_dir ='/share/phoenix/nfs06/S9/kh775/code/wsfm/scripts/data/keypoint_localization/old/', image_dir='/share/phoenix/nfs06/S9/kh775/dataset/megascenes_augmented_exhaustive/', split='test', resolution=[(512, 512)])" \
#     --train_criterion="ConfLoss(PointLoss(L21), alpha=0.2)" \
#     --test_criterion="PointLoss(L21)" \
#     --model "AsymmetricCroCo3DStereo(pos_embed='RoPE100', patch_embed_cls='ManyAR_PatchEmbed', img_size=(512, 512), head_type='dpt', output_mode='pts3d', depth_mode=('exp', -inf, inf), conf_mode=('exp', 1, inf), enc_embed_dim=1024, enc_depth=24, enc_num_heads=16, dec_embed_dim=768, dec_depth=12, dec_num_heads=12)" \
#     --pretrained="checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth" \
#     --lr=1e-04 --min_lr=1e-06 --warmup_epochs=3 --epochs=500 --train_batch_size=2 --test_batch_size=1 --accum_iter=1 \
#     --save_freq=1 --keep_freq=10 --eval_freq=1 --num_workers=4\
#     --output_dir="checkpoints/dust3r_dpt512_allplans_xz_DELETE"


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
            bad_plans = self._get_bad_plans()
            for row in reader:
                i, landmark, comp, plan_name, image_name = row
                if (plan_name, image_name) not in bad_pairs and (landmark, comp) not in bad_landmark_comp_pairs and plan_name not in bad_plans:
                    self.image_pairs.append(row)
        print(f"{len(self.image_pairs)} image pairs loaded")

    def _get_bad_pairs(self):
        bad_pairs = set()
        for file_name in ["corrupted_pairs.txt", "oob_pairs_test.txt", "oob_pairs_train.txt"]:
            file_path = os.path.join(self.data_dir, file_name)
            for pair in list(open(file_path).readlines()):
                plan_path, image_path = pair.replace("\n", "").split(" /")
                plan_name = plan_path.split("/")[-1]
                image_name = ("/"+image_path).replace(self.image_dir, "")
                image_name = "/".join(image_name.split("/")[2:])
                bad_pairs.add((plan_name, image_name))    
            print(f"Loaded {len(list(open(file_path).readlines()))} bad pairs from {file_name}")
        return bad_pairs

    def _get_bad_plans(self):
        bad_plans = set()
        plan_path = os.path.join(self.data_dir, "bad_plans.txt")
        for plan_name in list(open(plan_path).readlines()):
            bad_plans.add(plan_name.replace("\n", "").split("plans/")[-1])
        print(f"Loaded {len(list(open(plan_path).readlines()))} bad plans from {plan_path}")
        return bad_plans

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
        