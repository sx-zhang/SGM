import numpy as np
import quaternion
import _pickle as cPickle
import bz2
import os
import math
import gzip
import json
import sys
sys.path.append(".")
from semexp.envs.utils.fmm_planner import FMMPlanner
from semexp.constants import color_palette
import skimage.morphology
import cv2
from PIL import Image
import numpy.ma as ma
import scipy.signal as signal
 # sxz use gt val map
dataset_info_file = '..data/datasets/objectnav/gibson/v1.1/val/val_info.pbz2'
with bz2.BZ2File(dataset_info_file, "rb") as f:
    dataset_info = cPickle.load(f)
    
LOCAL_MAP_SIZE = 480  # TO DO
object_boundary = 1 - 0.5
map_resolution = 5
color_pal = [int(x * 255.0) for x in color_palette]

def origin_sem_map_vis(GT_map):
    sem_map_gt = Image.new("P", (GT_map.shape[1], GT_map.shape[0]))
    sem_map_gt.putpalette(color_pal)
    sem_map_gt.putdata(GT_map.flatten().astype(np.uint8))
    sem_map_gt = sem_map_gt.convert("RGB")
    sem_map_gt = np.flipud(sem_map_gt)

    sem_map_gt = sem_map_gt[:, :, [2, 1, 0]]
    sem_map_gt = cv2.resize(
        sem_map_gt, (480, 480), interpolation=cv2.INTER_NEAREST
    )
    cv2.imwrite('..img/origin_map.png',sem_map_gt, [int(cv2.IMWRITE_JPEG_QUALITY), 95])


def convert_3d_to_2d_pose(position, rotation):
    x = -position[2]
    y = -position[0]
    axis = quaternion.as_euler_angles(rotation)[0]
    if (axis % (2 * np.pi)) < 0.1 or (axis % (2 * np.pi)) > 2 * np.pi - 0.1:
        o = quaternion.as_euler_angles(rotation)[1]
    else:
        o = 2 * np.pi - quaternion.as_euler_angles(rotation)[1]
    if o > np.pi:
        o -= 2 * np.pi
    return x, y, o

def map_conversion(sem_map, start_x, start_y, start_o):
    output_map = np.zeros((LOCAL_MAP_SIZE, LOCAL_MAP_SIZE))
    sin = math.sin(np.pi - start_o)
    cos = math.cos(np.pi - start_o)
    for i in range(sem_map.shape[0]):
        loc = np.nonzero(sem_map[i])
        a = loc[0]-start_x
        b = loc[1]-start_y
        # loc_conversion = loc[0] - start_x + LOCAL_MAP_SIZE//2, loc[1] - start_y + LOCAL_MAP_SIZE//2  # To do
        loc_conversion = (a * cos + b * sin).astype(np.int) + LOCAL_MAP_SIZE//2, (b * cos - a * sin).astype(np.int) + LOCAL_MAP_SIZE//2
        loc_conversion = void_out_of_boundary(loc_conversion)
        if len(loc_conversion[0])==0:
            continue
        if i == 0:
            color_index = i+2.0  # free map
        else:
            color_index = i+4.0  # object semantic
        output_map[loc_conversion] = color_index
    return output_map

def void_out_of_boundary(locs):
    new_locs = [[],[]]
    for i in range(locs[0].shape[0]):
        if 0<locs[0][i]<LOCAL_MAP_SIZE and 0<locs[1][i]<LOCAL_MAP_SIZE:
            new_locs[0].append(locs[0][i])
            new_locs[1].append(locs[1][i])
        else:
            continue
    return [np.array(new_locs[0]), np.array(new_locs[1])]

def data_check(scene_name):
    episodes_file = '..data/datasets/objectnav/gibson/v1.1/val/content/{}_episodes.json.gz'.format(scene_name)
    with gzip.open(episodes_file, "r") as f:
        data = json.loads(f.read().decode("utf-8"))
        eps_data = data["episodes"]
    save_data = []
    for i, eps in enumerate(eps_data):
        # scene_name = os.path.basename(episodes_file).split("_episodes")[0]
        goal_idx = eps["object_ids"][0]
        floor_idx = 0
        scene_info = dataset_info[scene_name]
        sem_map = scene_info[floor_idx]["sem_map"]
        
        map_obj_origin = scene_info[floor_idx]["origin"]
        min_x, min_y = map_obj_origin / 100.0
        pos = eps["start_position"]
        rot = quaternion.from_float_array(eps["start_rotation"])
        x, y, o = convert_3d_to_2d_pose(pos, rot)
        start_x, start_y = int((-y - min_y) * 20.0), int((-x - min_x) * 20.0)
        sem_map = map_conversion(sem_map, start_x, start_y, o)
        origin_sem_map_vis(sem_map)
        goal_loc = (sem_map == (goal_idx+5.0))
        # if not len(np.nonzero(goal_loc)[0]):
        #     print('bad data')
        #     previous_eps = save_data[-1]
        #     eps = previous_eps.copy()
        #     eps['episode_id'] = i
        #     save_data.append(eps)
        # else:
        #     save_data.append(eps)
        exp_map = np.zeros_like(sem_map)
        exp_map[sem_map==2] = 1
        selem = skimage.morphology.disk(2)
        # exp_map = cv2.dilate(exp_map, selem)
        exp_map = signal.medfilt(exp_map,3)
        planner = FMMPlanner(exp_map)
        selem = skimage.morphology.disk(
            int(object_boundary * 100.0 / map_resolution)
        )
        goal_map = np.zeros_like(sem_map)
        goal_map[goal_loc] = 1
        goal_map = cv2.dilate(goal_map, selem)
        planner.set_multi_goal(goal_map, validate_goal=True)
        dist_map = planner.fmm_dist
        dist_map_vis = cv2.applyColorMap(dist_map.astype(np.uint8), cv2.COLORMAP_JET)
        dist_map_vis[goal_map==1] = [255,0,255]
        dist_map_vis[238:242,238:242] = [255,0,255]
        stx,sty,replan,stop = planner.get_short_term_goal([240,240])
        stx = stx.astype(np.int)
        sty = sty.astype(np.int)
        dist_map_vis[stx-2:stx+2,sty-2:sty+2] = [0,255,0]
        i = 0
        while not stop:
            stx,sty,replan,stop = planner.get_short_term_goal([stx,sty])
            stx = stx.astype(np.int)
            sty = sty.astype(np.int)
            dist_map_vis[stx-2:stx+2,sty-2:sty+2] = [0,255,0]
            i += 1
            print(i)
            
        cricle_o = np.zeros_like(dist_map)
        # cricle_o[240-2:240+2,240-2:240+2] = 1
        cricle_o[240,240] = 1
        cricle_o = cv2.dilate(cricle_o, skimage.morphology.disk(60))
        mx_crilce = ma.masked_array(exp_map, 1-cricle_o)
        mx_crilce = ma.filled(mx_crilce, 0)
        mx_vis = np.ones((480,480,3))*255
        # mx_vis[mx_crilce==1] = [100, 100, 100]
        # mx_vis[cricle_o==1] = [100, 100, 100]
        # mx_vis[exp_map==1] = [100, 100, 100]
        ret, labels = cv2.connectedComponents(mx_crilce.astype(np.int8))
        mx_vis[labels==labels[240,240]] = [100, 100, 100]
        cv2.imwrite('..sxz/img/test.png',np.flipud(mx_vis), [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        cv2.imwrite('..sxz/img/dist_map.png',np.flipud(dist_map_vis), [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        pass
    # save_data
    # data["episodes"] = save_data
    # eps_data_new = json.dumps(data).encode('utf-8')
    # episodes_file_new = '..data/datasets/objectnav/gibson/v1.1/val/content/{}_episodes.json.gz'.format(scene_name+'_new')
    # with gzip.GzipFile(episodes_file_new, 'w') as f:
    #     f.write(eps_data_new)
            
val_rooms = ['Collierville', 'Corozal', 'Darden', 'Markleeville', 'Wiconisco']
# for scene in val_rooms:
#     print(scene)
#     data_check(scene)
data_check('Darden')
