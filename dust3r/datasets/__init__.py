# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
import numpy as np
import torch

from .arkitscenes import ARKitScenes  # noqa
from .base.batched_sampler import BatchedRandomSampler  # noqa
from .blendedmvs import BlendedMVS  # noqa
from .co3d import Co3d  # noqa
from .habitat import Habitat  # noqa
from .megadepth import MegaDepth  # noqa
from .megascenes_augmented import MegaScenesAugmented
from .scannetpp import ScanNetpp  # noqa
from .staticthings3d import StaticThings3D  # noqa
from .utils.transforms import *
from .waymo import Waymo  # noqa
from .wildrgbd import WildRGBD  # noqa


def collate_fn(batch):  # batch:[(view1, view2) * batch_size]
    # print(view1["img"].size(), view1["plan_xys"].shape)  # Should be torch.Size([8, 3, 224, 224]) torch.Size([8, 733, 2])   
    max_xys_len = max(item[0]["plan_xys"].shape[0] for item in batch)
    # print(f"max_xys_len: {max_xys_len}")
    
    view1_img_batched = []
    view2_img_batched = [] 
    plan_xys_batched = []
    image_xys_batched = []
    view1_instances = []
    view2_instances = []
    for view1, view2 in batch:  #(['img', 'plan_xys', 'image_xys'])
        view1_img_batched.append(torch.squeeze(torch.Tensor(view1["img"]), 0))
        view2_img_batched.append(torch.squeeze(torch.Tensor(view2["img"]), 0))
        view1_instances.append(view1["instance"])
        view2_instances.append(view2["instance"])

        plan_xys_batched.append(torch.from_numpy(np.pad(view1["plan_xys"], ((0, max_xys_len - view1["plan_xys"].shape[0]), (0, 0)), mode="constant", constant_values=0)))
        image_xys_batched.append(torch.from_numpy(np.pad(view1["image_xys"], ((0, max_xys_len - view1["image_xys"].shape[0]), (0, 0)), mode="constant", constant_values=0)))

    view1_img_batched = torch.stack(view1_img_batched)
    view2_img_batched = torch.stack(view2_img_batched)
    plan_xys_batched = torch.stack(plan_xys_batched)
    image_xys_batched = torch.stack(image_xys_batched)
    final_view1 = dict(
        img=view1_img_batched, 
        plan_xys=plan_xys_batched,
        image_xys=image_xys_batched,
        instance=view1_instances
    )
    final_view2 = dict(       
        img=view2_img_batched,
        plan_xys=plan_xys_batched,
        image_xys=image_xys_batched,
        instance=view2_instances
    )
    return final_view1, final_view2

def get_data_loader(dataset, batch_size, num_workers=8, shuffle=True, drop_last=True, pin_mem=True):
    import torch
    from croco.utils.misc import get_rank, get_world_size

    # pytorch dataset
    if isinstance(dataset, str):
        dataset = eval(dataset)

    world_size = get_world_size()
    rank = get_rank()

    try:
        sampler = dataset.make_sampler(batch_size, shuffle=shuffle, world_size=world_size,
                                       rank=rank, drop_last=drop_last)
    except (AttributeError, NotImplementedError):
        # not avail for this dataset
        if torch.distributed.is_initialized():
            sampler = torch.utils.data.DistributedSampler(
                dataset, num_replicas=world_size, rank=rank, shuffle=shuffle, drop_last=drop_last
            )
        elif shuffle:
            sampler = torch.utils.data.RandomSampler(dataset)
        else:
            sampler = torch.utils.data.SequentialSampler(dataset)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_mem,
        drop_last=drop_last,
    )

    return data_loader
