from os.path import join

import numpy as np
from dust3r.inference import loss_of_one_batch
from dust3r.losses import *
from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.training import build_dataset
from dust3r.utils.image import load_megascenes_augmented_images
from dust3r.utils.viz import get_viz
import matplotlib.pyplot as plt

def get_inference_viz(plan_path, img_path, xys_path, model, criterion, device):
    plan_xys, image_xys = np.load(xys_path)
    pair = load_megascenes_augmented_images((plan_path, img_path), size=224, plan_xys=plan_xys, image_xys=image_xys)  
    batches = build_dataset([pair], batch_size, num_workers=4, test=True)

    for batch in batches:
        output = loss_of_one_batch(batch, model=model, criterion=criterion, device=device, symmetrize_batch=False, use_amp=False, ret=None)
        viz = get_viz(output["view1"], output["view2"], output["pred1"], output["pred2"])
        return viz

if __name__ == '__main__':
    device = 'cuda'
    batch_size = 1
    criterion = eval("PointLoss()").to(device)
    schedule = 'cosine'
    lr = 0.01
    niter = 300

    model_name = "/share/phoenix/nfs06/S9/kh775/code/dust3r/checkpoints/dust3r_224_500_allplans_dist/checkpoint-best.pth"
    model = AsymmetricCroCo3DStereo.from_pretrained(model_name).to(device)

    image_dir = "/share/phoenix/nfs06/S9/kh775/dataset/megascenes_augmented_exhaustive"
    landmark = "Abbaye_de_Moyenmoutier"
    plan_path = join(image_dir, landmark, "plans", "File:Abbaye de Moyenmoutier-Plan de l'abbaye.jpg")
    img_path = join(image_dir, landmark, "images", "commons/Abbaye_de_Moyenmoutier/0/pictures/Abbaye de Moyenmoutier en août 2016 (1).jpg")
    npx_path = "/share/phoenix/nfs06/S9/kh775/code/wsfm/scripts/data/keypoint_localization/data/data_test/coords/097144.npy"
    viz = get_inference_viz(plan_path, img_path, npx_path, model, criterion, device)
    viz.savefig("inference_.png")

    # device = 'cuda'
    # batch_size = 1
    # criterion = eval("PointLoss()").to(device)
    # schedule = 'cosine'
    # lr = 0.01
    # niter = 300

    # model_name = "/share/phoenix/nfs06/S9/kh775/code/dust3r/checkpoints/dust3r_224_500_allplans_dist/checkpoint-best.pth"
    # model = AsymmetricCroCo3DStereo.from_pretrained(model_name).to(device)

    # image_dir = "/share/phoenix/nfs06/S9/kh775/dataset/megascenes_augmented_exhaustive"
    # landmark = "Abbaye_de_Moyenmoutier"
    # plan_path = join(image_dir, landmark, "plans", "File:Abbaye de Moyenmoutier-Plan de l'abbaye.jpg")
    # img_path = join(image_dir, landmark, "images", "commons/Abbaye_de_Moyenmoutier/0/pictures/Abbaye de Moyenmoutier en août 2016 (1).jpg")
    # plan_xys, image_xys = np.load("/share/phoenix/nfs06/S9/kh775/code/wsfm/scripts/data/keypoint_localization/data/data_test/coords/097144.npy")
    # pair = load_megascenes_augmented_images((plan_path, img_path), size=224, plan_xys=plan_xys, image_xys=image_xys)  
    # batches = build_dataset([pair], batch_size, num_workers=4, test=True)

    # for i, batch in enumerate(batches):
    #     output = loss_of_one_batch(batch, model=model, criterion=criterion, device=device, symmetrize_batch=False, use_amp=False, ret=None)
    #     viz = get_viz(output["view1"], output["view2"], output["pred1"], output["pred2"])
    #     viz.savefig(f"inferece_{i}.png")
        
