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

def get_viz(view1, view2, pred1, pred2):
    def gen_plot(view1, view2, pred1, pred2):
        view1_img = view1["img"].permute(0, 2, 3, 1).cpu().numpy()
        view2_img = view2["img"].permute(0, 2, 3, 1).cpu().numpy()

        B, image_size, _,  _ = view1_img.shape
        fig, axes = plt.subplots(B, 3, figsize=(10, B*3))
        titles = ["gt", "pred", "image"]
        for b in range(B):
            if B == 1:
                view1_img_scaled = reverse_ImgNorm(view1_img[b])
                axes[0].imshow(view1_img_scaled)
                view1_points = view1["plan_xys"][b].cpu().numpy()
                axes[0].scatter(view1_points[:,0], view1_points[:,1], s=5)
                axes[0].set_title(titles[0])
                
                axes[1].imshow(view1_img_scaled)    
                pred2_points = pred2["pts3d_in_other_view"][b].detach().cpu().numpy()
                x_coords = view2["image_xys"][b][:,0].cpu().numpy()
                y_coords = view2["image_xys"][b][:,1].cpu().numpy()
                pred2_points = pred2_points[y_coords, x_coords, :2]
                pred2_points = reverse_CoordNorm(pred2_points, image_size)
                axes[1].scatter(pred2_points[:,0], pred2_points[:,1], s=5)
                axes[1].set_title(titles[1])  

                view2_img_scaled = reverse_ImgNorm(view2_img[b])
                axes[2].imshow(view2_img_scaled)   
                view2_points = view1["image_xys"][b].cpu().numpy()   
                axes[2].scatter(view2_points[:,0], view2_points[:,1], s=5) 
                axes[2].set_title(titles[2])  
            else:
                view1_img_scaled = reverse_ImgNorm(view1_img[b])
                axes[b, 0].imshow(view1_img_scaled)
                view1_points = view1["plan_xys"][b].cpu().numpy()
                axes[b, 0].scatter(view1_points[:,0], view1_points[:,1], s=5)
                axes[b, 0].set_title(titles[0])
                
                axes[b, 1].imshow(view1_img_scaled)    
                pred2_points = pred2["pts3d_in_other_view"][b].detach().cpu().numpy()
                x_coords = view2["image_xys"][b][:,0].cpu().numpy()
                y_coords = view2["image_xys"][b][:,1].cpu().numpy()
                pred2_points = pred2_points[y_coords, x_coords, :2]
                pred2_points = reverse_CoordNorm(pred2_points, image_size)
                axes[b, 1].scatter(pred2_points[:,0], pred2_points[:,1], s=5)
                axes[b, 1].set_title(titles[1])  

                view2_img_scaled = reverse_ImgNorm(view2_img[b])
                axes[b, 2].imshow(view2_img_scaled)   
                view2_points = view1["image_xys"][b].cpu().numpy()   
                axes[b, 2].scatter(view2_points[:,0], view2_points[:,1], s=5) 
                axes[b, 2].set_title(titles[2])   
        plt.tight_layout()
        return fig
    viz = gen_plot(view1, view2, pred1, pred2)
    return viz