import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os

# color_pal = [int(x * 255.0) for x in color_palette]
GIBSON_OBJECT_COLORS = [
    (0.9400000000000001, 0.7818, 0.66),
    (0.9400000000000001, 0.8868, 0.66),
    (0.8882000000000001, 0.9400000000000001, 0.66),
    (0.7832000000000001, 0.9400000000000001, 0.66),
    (0.6782000000000001, 0.9400000000000001, 0.66),
    (0.66, 0.9400000000000001, 0.7468000000000001),
    (0.66, 0.9400000000000001, 0.8518000000000001),
    (0.66, 0.9232, 0.9400000000000001),
    (0.66, 0.8182, 0.9400000000000001),
    (0.66, 0.7132, 0.9400000000000001),
    (0.7117999999999999, 0.66, 0.9400000000000001),
    (0.8168, 0.66, 0.9400000000000001),
    (0.9218, 0.66, 0.9400000000000001),
    (0.9400000000000001, 0.66, 0.8531999999999998),
    (0.9400000000000001, 0.66, 0.748199999999999),

    (0.66, 0.7132, 0.9400000000000001),
    (0.7117999999999999, 0.66, 0.9400000000000001),
    (0.8168, 0.66, 0.9400000000000001),
    (0.9218, 0.66, 0.9400000000000001),
    (0.9400000000000001, 0.66, 0.8531999999999998),
    (0.9400000000000001, 0.66, 0.748199999999999),
]

OBJECT_COLORS = GIBSON_OBJECT_COLORS

COLOR_PALETTE = [
    1.0,
    1.0,
    1.0,  # Out-of-bounds
    0.7,
    0.7,
    0.7,  # Floor
    0.3,
    0.3,
    0.3,  # Wall
    *[oci for oc in OBJECT_COLORS for oci in oc],
]

color_pal = [int(x * 255.0) for x in COLOR_PALETTE]

def visualize_semmap(semmap, save_dir):
    vis_semmap = np.zeros((semmap.shape[1],semmap.shape[2]))
    for i in range(semmap.shape[0]):
        vis_semmap[semmap[i].astype(np.bool_)] = i+1
    vis_map(vis_semmap, save_dir)

def vis_map(vis_map, save_dir):
    img = Image.new("P", (vis_map.shape[1], vis_map.shape[0]))
    img.putpalette(color_pal)
    img.putdata((vis_map.flatten() % 40).astype(np.uint8))
    img = img.convert("RGB")
    img = np.array(img)
    cv2.imwrite(save_dir, img)
    print('debug visual semmap')

def visualize_distmap(distmap, save_dir):
    distmap_np = distmap.numpy()*10
    for i in range(distmap.shape[0]):
        distmap_vis = cv2.applyColorMap(distmap_np[i].astype(np.uint8), cv2.COLORMAP_SUMMER)
        os.makedirs(save_dir, exist_ok=True)
        cv2.imwrite(os.path.join(save_dir, 'map_{}.png'.format(i)),np.flipud(distmap_vis), [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    print('debug distmap')
    pass
        
def boolmap2nummap(boolmap):
    nummap = np.zeros((boolmap.shape[0],boolmap.shape[2],boolmap.shape[3]))
    for b in range(boolmap.shape[0]):
        semmap = boolmap[b]
        vis_semmap = np.zeros((semmap.shape[1],semmap.shape[2]))
        for i in range(semmap.shape[0]):
            if i==0:
                vis_semmap[semmap[i].astype(np.bool)] = 2
            elif i==1:
                vis_semmap[semmap[i].astype(np.bool)] = 1
            else:
                vis_semmap[semmap[i].astype(np.bool)] = i+3
        nummap[b] = vis_semmap
    return nummap