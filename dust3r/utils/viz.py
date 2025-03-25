import base64
import os
from io import BytesIO
import PIL.Image

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.transforms import ToTensor


def ReverseImgNorm(x):
    x = x * 0.5 + 0.5
    x *= 255.0
    if isinstance(x, np.ndarray):
        x = x.clip(0, 255).astype(np.uint8)
    else:
        x = torch.clamp(x, 0, 255).to(torch.uint8)
    return x

def ReverseCoordNorm(np_array, size):
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

def get_viz(view1, view2, pred1, pred2, losses=None):
    def gen_plot(view1, view2, pred1, pred2, losses=None):
        view1_img = view1["img"].permute(0, 2, 3, 1).cpu().numpy()
        view2_img = view2["img"].permute(0, 2, 3, 1).cpu().numpy()

        B, image_size, _,  _ = view1_img.shape
        # titles = ["gt", "pred", "conf_plan", "conf_image", "image"]
        titles = ["gt", "pred2", "conf_pred2", "image+correspondences", "image"]
        N = len(titles)
        fig, axes = plt.subplots(B, N, figsize=(30, B*N))
        training_with_xy = False #training_with_xz if False
        
        bs = range(B) if losses is None else np.argsort(losses)
        idx = 0

        for b in bs:
            if B == 1:
                view1_img_scaled = ReverseImgNorm(view1_img[b])
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
                pred = ReverseCoordNorm(pred, image_size)
                rgbs = get_rgbs(pred, image_size)
                axes[1].scatter(pred[:,0], pred[:,1], s=5, c=rgbs)
                axes[1].set_title(titles[1])   

                conf2 = pred2["conf"][b].detach().cpu().numpy()
                axes[2].imshow(conf2)   
                axes[2].set_title(titles[2]) 

                view2_img_scaled = ReverseImgNorm(view2_img[b])
                axes[3].imshow(view2_img_scaled)   
                image_xys = get_nonzero_xys(view2["xys"][b].cpu()).numpy()   
                axes[3].scatter(image_xys[:,0], image_xys[:,1], s=1, c=rgbs) 
                axes[3].set_title(titles[3])      

                axes[4].imshow(view2_img_scaled)   
                axes[4].set_title(titles[4])   

                if losses is not None:
                    axes[0].set_ylabel(f"loss: {losses[b]:.6f}")
            else:
                view1_img_scaled = ReverseImgNorm(view1_img[b])
                axes[idx, 0].imshow(view1_img_scaled)
                gt = get_nonzero_xys(view1["xys"][b].cpu()).numpy()
                axes[idx, 0].scatter(gt[:,0], gt[:,1], s=5)
                axes[idx, 0].set_title(titles[0])
                
                axes[idx, 1].imshow(view1_img_scaled)    
                pred = pred2["pts3d_in_other_view"][b].detach().cpu().numpy()
                view2_xys = get_nonzero_xys(view2["xys"][b].cpu())
                x_coords = view2_xys[:,0].numpy()
                y_coords = view2_xys[:,1].numpy()
                if training_with_xy:
                    pred = pred[y_coords, x_coords, :2]
                else:
                    pred = np.stack((pred[y_coords, x_coords, 0], pred[y_coords, x_coords, 2]), axis=1)
                pred = ReverseCoordNorm(pred, image_size)
                rgbs = get_rgbs(pred, image_size)
                axes[idx, 1].scatter(pred[:,0], pred[:,1], s=5, c=rgbs)
                axes[idx, 1].set_title(titles[1])   

                conf2 = pred2["conf"][b].detach().cpu().numpy()
                axes[idx, 2].imshow(conf2)   
                axes[idx, 2].set_title(titles[2]) 

                view2_img_scaled = ReverseImgNorm(view2_img[b])
                axes[idx, 3].imshow(view2_img_scaled)   
                image_xys = get_nonzero_xys(view2["xys"][b].cpu()).numpy()   
                axes[idx, 3].scatter(image_xys[:,0], image_xys[:,1], s=1, c=rgbs) 
                axes[idx, 3].set_title(titles[3])      

                axes[idx, 4].imshow(view2_img_scaled)   
                axes[idx, 4].set_title(titles[4])   

                # axes[idx, 5].imshow(view1_img_scaled)    
                # pred = pred1["pts3d"][idx].detach().cpu().numpy()
                # # pred = np.stack((pred[:, :, 0], pred[:, :, 1]), axis=1)
                # pred = np.stack((pred[:, :, 0], pred[:, :, 2]), axis=1)
                # pred = reverse_CoordNorm(pred, image_size)
                # axes[idx, 5].scatter(pred[:,0], pred[:,1], s=5)
                # axes[idx, 5].set_title(titles[5]) 

                # conf1 = pred1["conf"][b].detach().cpu().numpy()
                # axes[idx, 6].imshow(conf1)   
                # axes[idx, 6].set_title(titles[6]) 
                
                if losses is not None:
                    axes[idx, 0].set_ylabel(f"{b}. loss: {losses[b]:.6f}")
                else:
                    axes[idx, 0].set_ylabel(f"{b}")
            
            idx += 1

        plt.subplots_adjust(hspace=0.0, wspace=0.0)  # Set both to 0 to remove space
        plt.tight_layout()
        return fig
    viz = gen_plot(view1, view2, pred1, pred2, losses=losses)
    return viz


def get_cdf(values, epoch):
    """
    Plot a graph where the y-axis represents the percentage of values in the list
    that are less than the x value.
    
    Args:
        values: List of floating point numbers
    """
    # Sort the values
    sorted_values = np.sort(values)
    
    # Calculate the cumulative percentages (0 to 100)
    percentages = np.arange(1, len(sorted_values) + 1) / len(sorted_values) * 100
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    plt.plot(sorted_values, percentages, '-o', markersize=4)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlabel('Loss')
    plt.ylabel('Percentage of values < x')
    plt.title(f'Epoch {epoch}')
    
    # Add a horizontal line at 50% for the median
    plt.axhline(y=50, color='r', linestyle='--', alpha=0.5, label='50% (median)')
    
    # Add a few percentage annotations
    plt.axhline(y=25, color='gray', linestyle='--', alpha=0.5, label='25%')
    plt.axhline(y=75, color='gray', linestyle='--', alpha=0.5, label='75%')
    
    plt.legend()
    plt.tight_layout()
    
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)

    image = PIL.Image.open(buf)
    image_tensor = torch.tensor(np.array(image).transpose(2, 0, 1))  # Convert to CHW format
    return image_tensor


def get_viz_html(fig, save_path):
    buf = BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)

    img_str = base64.b64encode(buf.read()).decode('utf-8')

    html_str = f'<img src="data:image/png;base64,{img_str}"/>'
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(html_str)
        print(f"HTML saved to {os.path.abspath(save_path)}")

# def get_viz_html(view1s, view2s, pred1s, pred2s, save_path=None, point_color='blue', point_size=1):
#     """
#     Display two PyTorch tensors side by side in HTML format, with optional point overlays.
    
#     Parameters:
#     tensor1 (torch.Tensor): First tensor with shape (B, C, H, W)
#     tensor2 (torch.Tensor): Second tensor with shape (B, C, H, W)
#     titles (list, optional): List of tuples with titles for each pair of images
#     save_path (str, optional): Path to save the HTML file. If None, returns the HTML object
#     points1 (torch.Tensor, optional): Points to draw on first tensor, shape (B, N, 2) where N is number of points
#                                      and values are (x, y) coordinates in pixel space
#     points2 (torch.Tensor, optional): Points to draw on second tensor, shape (B, N, 2)
#     point_color (str, optional): Color of the points to draw
#     point_size (int, optional): Size of the points to draw
    
#     Returns:
#     IPython.display.HTML or None: HTML output displaying the images side by side if save_path is None
#     """
#     B, C, H, W = view1s["img"].size()
#     titles = [[f"{title} {i}" for title in ["gt", "pred2", "conf_pred2", "image+correspondences", "image"]] for i in range(B)]

#     # Function to convert a tensor to a base64 encoded image with optional points
#     def tensor_to_base64(tensor, points=None, to_pil=None):        
#         # Convert to numpy and then to PIL Image
#         images_base64 = []
        
#         for i in range(tensor.size(0)):
#             tensor_i = tensor[i]
#             if tensor_i.size(0) == tensor_i.size(1):
#                 tensor_i = torch.unsqueeze(tensor_i, 0)
#                 print(max(tensor_i), min(tensor_i))
#             print(tensor_i.size())
#             img = ReverseImgNorm(tensor_i)
#             img = to_pil(img)
            
#             # If points are provided, draw them on the image
#             if points is not None and i < len(points):
#                 draw = ImageDraw.Draw(img)
#                 batch_points = points[i]
                
#                 for point in get_nonzero_xys(batch_points):
#                     x, y = point
#                     # Draw a small circle at each point
#                     draw.ellipse(
#                         [(x - point_size, y - point_size), 
#                          (x + point_size, y + point_size)], 
#                         fill=point_color
#                     )
            
#             buffer = BytesIO()
#             img.save(buffer, format="PNG")
#             img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
#             images_base64.append(img_str)
        
#         return images_base64
    
#     # Convert both tensors to base64 with points if provided
#     images0_base64 = tensor_to_base64(view1s["img"], points=view1s["xys"], to_pil=T.ToPILImage())
#     images1_base64 = tensor_to_base64(view1s["img"], to_pil=T.ToPILImage())
#     images2_base64 = tensor_to_base64(pred2s["conf"], to_pil=T.ToPILImage())
#     images3_base64 = tensor_to_base64(view2s["img"], points=view2s["xys"], to_pil=T.ToPILImage())
#     images4_base64 = tensor_to_base64(view2s["img"], to_pil=T.ToPILImage())
    
    
#     # Create HTML
#     html = """
#     <style>
#         .image-container {
#             display: flex;
#             margin-bottom: 20px;
#         }
#         .image-pair {
#             margin-right: 20px;
#             text-align: center;
#         }
#         .image-pair img {
#             max-width: 100%;
#             height: auto;
#         }
#         .image-title {
#             margin-top: 5px;
#             font-weight: bold;
#         }
#     </style>
#     """
    
#     for i in range(B):
#         html += f"""
#         <div class="image-container">
#             <div class="image-pair">
#                 <div class="image-title">{titles[i][0]}</div>
#                 <img src="data:image/png;base64,{images0_base64[i]}" />
#             </div>
#             <div class="image-pair">
#                 <div class="image-title">{titles[i][1]}</div>
#                 <img src="data:image/png;base64,{images1_base64[i]}" />
#             </div>
#             <div class="image-pair">
#                 <div class="image-title">{titles[i][2]}</div>
#                 <img src="data:image/png;base64,{images2_base64[i]}" />
#             </div>
#             <div class="image-pair">
#                 <div class="image-title">{titles[i][3]}</div>
#                 <img src="data:image/png;base64,{images3_base64[i]}" />
#             </div>
#             <div class="image-pair">
#                 <div class="image-title">{titles[i][4]}</div>
#                 <img src="data:image/png;base64,{images4_base64[i]}" />
#             </div>
#         </div>
#         """
    
#     if save_path:
#         with open(save_path, 'w', encoding='utf-8') as f:
#             f.write(html)
#         print(f"HTML saved to {os.path.abspath(save_path)}")
#         return None
#     else:
#         return HTML(html)

# # Example usage:
# # Assuming you have two tensors of shape (B, C, H, W)
# # tensor1 = torch.randn(3, 3, 224, 224)  # 3 RGB images
# # tensor2 = torch.randn(3, 3, 224, 224)  # 3 RGB images
# # custom_titles = [("Original 1", "Processed 1"), ("Original 2", "Processed 2"), ("Original 3", "Processed 3")]
# # 
# # # Create some random points - shape (B, N, 2) where N is number of points per image
# # # points1 = torch.randint(0, 224, (3, 10, 2))  # 3 images, 10 points each, x and y coordinates
# # # points2 = torch.randint(0, 224, (3, 10, 2))
# # 
# # # To display in notebook:
# # display_tensors_side_by_side(tensor1, tensor2, titles=custom_titles)
# # 
# # # To display with points:
# # # display_tensors_side_by_side(tensor1, tensor2, titles=custom_titles, 
# # #                             points1=points1, points2=points2, point_color='yellow', point_size=4)
# # 
# # # To save as HTML file:
# # # display_tensors_side_by_side(tensor1, tensor2, points1=points1, points2=points2, titles=custom_titles, save_path="comparison.html")
