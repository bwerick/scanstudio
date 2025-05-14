from segment_anything import SamPredictor, sam_model_registry, SamAutomaticMaskGenerator
import cv2 as cv
import numpy as np 
import pandas as pd 
import os
from matplotlib import pyplot as plt
import torch



imagepath = "/Users/erickduarte/git/segmentation/test_frames/The Sun Also Rises Ernest Hemingway.mp4/0000000495.jpg"
image = cv.cvtColor(cv.imread(imagepath), cv.COLOR_BGR2RGB)




y, x = image.shape[:2]
print(image.shape)
x_center = x // 2
y_center = y // 2

input_point = np.array([[x_center * 0.7, y_center * 0.5], [x_center * 0.5, y_center * 1.5], [x_center*1.3, y_center* 0.5], [x_center*1.5, y_center* 1.5]])
input_label = np.array([1, 1, 0, 0])

#input_point = np.array([[1200, 1250]])
#input_label = np.array([1])

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


sam = sam_model_registry["vit_h"](checkpoint="/Users/erickduarte/git/segment-anything/sam_vit_h_4b8939.pth")
predictor = SamPredictor(sam)
predictor.set_image(image)



#mask_generator = SamAutomaticMaskGenerator(sam, points_per_batch=16)
#sam.to(device="mps")



# plt.imshow(image)
# show_points(input_point, input_label, plt.gca())
# plt.axis('on')
# plt.show()  
# plt.waitforbuttonpress()



masks, scores, logits = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    multimask_output=False,
)
print(masks.shape)



#color = np.array([255/255, 255/255, 255/255, 1]) 
h, w = masks[0].shape[:2] 
mask = masks[0].reshape(h, w)
if len(image.shape) == 3:
    masknew = np.stack([mask, mask, mask], axis=-1)

else:
    masknew = np.stack(mask, axis=-1)
print(masknew.shape)
# mask_image = (mask * 255).astype(np.uint8)  # Convert to uint8 format
# cv2.imwrite('mask.png', mask_image)
page_segment = np.zeros_like(image)
page_segment[masknew] = image[masknew]

cv.imwrite("pagesegmentnewest.png", page_segment)

""" for i, (mask, score) in enumerate(zip(masks, scores)):
#     plt.figure(figsize=(10,10))
    print(mask, score)
    plt.imshow(image)
    show_mask(mask, plt.gca())
    show_points(input_point, input_label, plt.gca())
    plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
    plt.show()
    plt.waitforbuttonpress() """