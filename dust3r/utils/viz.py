import io
import matplotlib.pyplot as plt
import PIL.Image
from torchvision.transforms import ToTensor
import numpy as np
import torch


def reverse_ImgNorm(np_array):
    np_array = np_array * 0.5 + 0.5
    np_array *= 255.0
    np_array = np_array.clip(0, 255).astype(np.uint8)
    return np_array

def reverse_CoordNorm(np_array, size):
    return (np_array + 1) * (size - 1) / 2


def get_rgbs(pred, image_size):
    rgbs = []
    for pred_x, pred_y in pred:
        r = max(0.0, min(pred_x / image_size, 1.0)) 
        b = max(0.0, min(pred_y / image_size, 1.0)) 
        rgbs.append((r, 0, b))
    return rgbs

def get_nonzero_xys(xys):  #xys:(N,2)
    mask = torch.all(xys == 0, dim=1)
    M = torch.sum(mask.flip(0)).item()
    if M != 0:
        return xys[:-M]
    else:
        return xys

def get_viz(view1, view2, pred1, pred2):
    def gen_plot(view1, view2, pred1, pred2):
        view1_img = view1["img"].permute(0, 2, 3, 1).cpu().numpy()
        view2_img = view2["img"].permute(0, 2, 3, 1).cpu().numpy()

        B, image_size, _,  _ = view1_img.shape
        # titles = ["gt", "pred", "conf_plan", "conf_image", "image"]
        titles = ["gt", "pred2", "conf_pred2", "image+correspondences", "image"]
        N = len(titles)
        fig, axes = plt.subplots(B, N, figsize=(24, B*N))
        training_with_xy = False #training_with_xz if False
        
        for b in range(B):
            if B == 1:
                view1_img_scaled = reverse_ImgNorm(view1_img[b])
                axes[0].imshow(view1_img_scaled)
                gt = get_nonzero_xys(view1["xys"][b].cpu()).numpy()
                axes[0].scatter(gt[:,0], gt[:,1], s=5)
                axes[0].set_title(titles[0])
                
                axes[1].imshow(view1_img_scaled)    
                pred = pred2["pts3d_in_other_view"][b].detach().cpu().numpy()
                view2_xys = get_nonzero_xys(view2["xys"][b].cpu())
                x_coords = view2_xys[:,0].numpy()
                y_coords = view2_xys[:,1].numpy()
                if training_with_xy:
                    pred = pred[y_coords, x_coords, :2]
                else:
                    pred = np.stack((pred[y_coords, x_coords, 0], pred[y_coords, x_coords, 2]), axis=1)
                pred = reverse_CoordNorm(pred, image_size)
                rgbs = get_rgbs(pred, image_size)
                axes[1].scatter(pred[:,0], pred[:,1], s=5, c=rgbs)
                axes[1].set_title(titles[1])  

                conf2 = pred2["conf"][b].detach().cpu().numpy()
                axes[2].imshow(conf2)   
                axes[2].set_title(titles[2]) 

                view2_img_scaled = reverse_ImgNorm(view2_img[b])
                axes[3].imshow(view2_img_scaled)   
                image_xys = get_nonzero_xys(view2["xys"][b].cpu()).numpy()   
                axes[3].scatter(image_xys[:,0], image_xys[:,1], s=1, c=rgbs) 
                axes[3].set_title(titles[3])      

                axes[4].imshow(view2_img_scaled)   
                axes[4].set_title(titles[4])   
            else:
                view1_img_scaled = reverse_ImgNorm(view1_img[b])
                axes[b, 0].imshow(view1_img_scaled)
                gt = get_nonzero_xys(view1["xys"][b].cpu()).numpy()
                axes[b, 0].scatter(gt[:,0], gt[:,1], s=5)
                axes[b, 0].set_title(titles[0])
                
                axes[b, 1].imshow(view1_img_scaled)    
                pred = pred2["pts3d_in_other_view"][b].detach().cpu().numpy()
                view2_xys = get_nonzero_xys(view2["xys"][b].cpu())
                x_coords = view2_xys[:,0].numpy()
                y_coords = view2_xys[:,1].numpy()
                if training_with_xy:
                    pred = pred[y_coords, x_coords, :2]
                else:
                    pred = np.stack((pred[y_coords, x_coords, 0], pred[y_coords, x_coords, 2]), axis=1)
                pred = reverse_CoordNorm(pred, image_size)
                rgbs = get_rgbs(pred, image_size)
                axes[b, 1].scatter(pred[:,0], pred[:,1], s=5, c=rgbs)
                axes[b, 1].set_title(titles[1])   

                conf2 = pred2["conf"][b].detach().cpu().numpy()
                axes[b, 2].imshow(conf2)   
                axes[b, 2].set_title(titles[2]) 

                view2_img_scaled = reverse_ImgNorm(view2_img[b])
                axes[b, 3].imshow(view2_img_scaled)   
                image_xys = get_nonzero_xys(view2["xys"][b].cpu()).numpy()   
                axes[b, 3].scatter(image_xys[:,0], image_xys[:,1], s=1, c=rgbs) 
                axes[b, 3].set_title(titles[3])      

                axes[b, 4].imshow(view2_img_scaled)   
                axes[b, 4].set_title(titles[4])   
        plt.subplots_adjust(hspace=0.0, wspace=0.0)  # Set both to 0 to remove space
        plt.tight_layout()
        return fig
    viz = gen_plot(view1, view2, pred1, pred2)
    return viz