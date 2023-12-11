from semexp.constants import color_palette
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
]

OBJECT_COLORS = GIBSON_OBJECT_COLORS

COLOR_PALETTE = [
    1.0,
    1.0,
    1.0,  # Out-of-bounds
    0.9,
    0.9,
    0.9,  # Floor
    0.3,
    0.3,
    0.3,  # Wall
    *[oci for oc in OBJECT_COLORS for oci in oc],
]

color_palette_for_sxz = [
    1.0,
    1.0,
    1.0,  # Out of bounds
    0.6,
    0.6,
    0.6,  # Obstacle
    0.95,
    0.95,
    0.95,  # Free space
    0.96,
    0.36,
    0.26,  # Visible mask
    0.12156862745098039,
    0.47058823529411764,
    0.7058823529411765,  # Goal mask
    0.9400000000000001,
    0.7818,
    0.66,
    0.9400000000000001,
    0.8868,
    0.66,
    0.8882000000000001,
    0.9400000000000001,
    0.66,
    0.7832000000000001,
    0.9400000000000001,
    0.66,
    0.6782000000000001,
    0.9400000000000001,
    0.66,
    0.66,
    0.9400000000000001,
    0.7468000000000001,
    0.66,
    0.9400000000000001,
    0.8518000000000001,
    0.66,
    0.9232,
    0.9400000000000001,
    0.66,
    0.8182,
    0.9400000000000001,
    0.66,
    0.7132,
    0.9400000000000001,
    0.7117999999999999,
    0.66,
    0.9400000000000001,
    0.8168,
    0.66,
    0.9400000000000001,
    0.9218,
    0.66,
    0.9400000000000001,
    0.9400000000000001,
    0.66,
    0.8531999999999998,
    0.9400000000000001,
    0.66,
    0.748199999999999,
    0.94117647, # frontier
    0.50196078,
    0.50196078,
    1,
    0,
    0,
]

color_pal = [int(x * 255.0) for x in color_palette_for_sxz]

def visualize_semmap(semmap, save_dir):
    # for i in range(pred_map.shape[0]):
    #     if (i==0):
    #         pred_map_img[(pred_map[i]>0.5).astype(np.bool)] = 2
    #     elif (i==1):
    #         pred_map_img[(pred_map[i]>0.7).astype(np.bool)] = 1
    #     else:
    #         pred_map_img[(pred_map[i]>0.2).astype(np.bool)] = i+3
    vis_semmap = np.zeros((semmap.shape[1],semmap.shape[2]))
    for i in range(semmap.shape[0]):
        # vis_semmap[semmap[i].astype(np.bool)] = i+1
        if (i==0):
            vis_semmap[(semmap[i]>0.5).astype(np.bool)] = 2
        elif (i==1):
            vis_semmap[(semmap[i]>0.8).astype(np.bool)] = 1
        else:
            vis_semmap[(semmap[i]>0.4).astype(np.bool)] = i+3
    
    vis_map(vis_semmap, save_dir)

def vis_map(vis_map, save_dir):
    img = Image.new("P", (vis_map.shape[1], vis_map.shape[0]))
    img.putpalette(color_pal)
    img.putdata(vis_map.flatten().astype(np.uint8))
    img = img.convert("RGB")
    img = np.flipud(img)
    img = img[:, :, [2, 1, 0]]
    img = cv2.resize(img, (480, 480), interpolation=cv2.INTER_NEAREST)
    # img = np.array(img)
    # print(img.shape)
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
        