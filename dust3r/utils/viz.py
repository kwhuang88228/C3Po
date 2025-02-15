import io
import matplotlib.pyplot as plt
import PIL.Image
from torchvision.transforms import ToTensor
import numpy as np


def reverse_ImgNorm(np_array):
    np_array = np_array * 0.5 + 0.5
    np_array *= 255.0
    np_array = np_array.clip(0, 255).astype(np.uint8)
    return np_array

def reverse_CoordNorm(np_array, size):
    return (np_array + 1) * (size - 1) / 2


def get_rgbs(pred):
    rgbs = []
    for pred_x, pred_y in pred:
        r = pred_x / 255.0
        b = pred_y / 255.0
        rgbs.append((r, 0, b))
    return rgbs

def get_viz(view1, view2, pred1, pred2):
    def gen_plot(view1, view2, pred1, pred2):
        view1_img = view1["img"].permute(0, 2, 3, 1).cpu().numpy()
        view2_img = view2["img"].permute(0, 2, 3, 1).cpu().numpy()

        # conf = pred2['conf'][0].detach().cpu().numpy()
        # print(f"conf shape: {conf.shape}; {np.max(conf)}; {np.min(conf)}")

        B, image_size, _,  _ = view1_img.shape
        fig, axes = plt.subplots(B, 4, figsize=(10, B*4))
        titles = ["gt", "pred", "confidence", "image"]
        for b in range(B):
            if B == 1:
                view1_img_scaled = reverse_ImgNorm(view1_img[b])
                axes[0].imshow(view1_img_scaled)
                gt = view1["plan_xys"][b].cpu().numpy()
                axes[0].scatter(gt[:,0], gt[:,1], s=5)
                axes[0].set_title(titles[0])
                
                axes[1].imshow(view1_img_scaled)    
                pred = pred2["pts3d_in_other_view"][b].detach().cpu().numpy()
                x_coords = view2["image_xys"][b][:,0].cpu().numpy()
                y_coords = view2["image_xys"][b][:,1].cpu().numpy()
                pred = pred[y_coords, x_coords, :2]
                pred = reverse_CoordNorm(pred, image_size)
                axes[1].scatter(pred[:,0], pred[:,1], s=5)
                axes[1].set_title(titles[1])  

                conf = pred2["conf"][b].detach().cpu().numpy()
                axes[2].imshow(conf, cmap="jet", interpolation="nearest")   
                axes[2].set_title(titles[2]) 

                view2_img_scaled = reverse_ImgNorm(view2_img[b])
                axes[3].imshow(view2_img_scaled)   
                image_xys = view1["image_xys"][b].cpu().numpy()   
                rgbs = get_rgbs(pred)
                axes[3].scatter(image_xys[:,0], image_xys[:,1], s=1, c=rgbs) 
                axes[3].set_title(titles[3])      
            else:
                view1_img_scaled = reverse_ImgNorm(view1_img[b])
                axes[b, 0].imshow(view1_img_scaled)
                gt = view1["plan_xys"][b].cpu().numpy()
                axes[b, 0].scatter(gt[:,0], gt[:,1], s=5)
                axes[b, 0].set_title(titles[0])
                
                axes[b, 1].imshow(view1_img_scaled)    
                pred = pred2["pts3d_in_other_view"][b].detach().cpu().numpy()
                x_coords = view2["image_xys"][b][:,0].cpu().numpy()
                y_coords = view2["image_xys"][b][:,1].cpu().numpy()
                pred = pred[y_coords, x_coords, :2]
                pred = reverse_CoordNorm(pred, image_size)
                axes[b, 1].scatter(pred[:,0], pred[:,1], s=5)
                axes[b, 1].set_title(titles[1])  

                conf = pred2["conf"][b].detach().cpu().numpy()
                axes[b, 2].imshow(conf, cmap="jet", interpolation="nearest")   
                axes[b, 2].set_title(titles[2]) 

                view2_img_scaled = reverse_ImgNorm(view2_img[b])
                axes[b, 3].imshow(view2_img_scaled)   
                image_xys = view1["image_xys"][b].cpu().numpy()   
                rgbs = get_rgbs(pred)
                axes[b, 3].scatter(image_xys[:,0], image_xys[:,1], s=1, c=rgbs) 
                axes[b, 3].set_title(titles[3])  
        plt.tight_layout()
        return fig
    viz = gen_plot(view1, view2, pred1, pred2)
    return viz