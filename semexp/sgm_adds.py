import numpy as np
import quaternion
import _pickle as cPickle
import bz2
import sys
sys.path.append("..")
import h5py
import json
import os
import math
import skimage.morphology
import cv2
from semexp.envs.utils.fmm_planner import FMMPlanner
import numpy.ma as ma
import scipy.signal as signal
import scipy.io as scio
import torch
from semexp.sxz_hoz import get_room_graph, cal_local_map_hoz
from semexp.utils.visualize_tools import *

# sxz use gt val map
dataset_info_file = '..data/datasets/objectnav/gibson/v1.1/val/val_info.pbz2'
dataset_file_path = '..data/semantic_maps/gibson/semantic_maps/'
with bz2.BZ2File(dataset_info_file, "rb") as f:
    dataset_info_1 = cPickle.load(f)
with open("..data/semantic_maps/gibson/semantic_maps/semmap_GT_info.json",'r') as fp:
    dataset_info = json.load(fp)

LOCAL_MAP_SIZE = 480  # TO DO
OBJECT_BOUNDARY = 1 - 0.5
MAP_RESOLUTION = 5
# location_shift = [(0,1),(1,1),(1,0),(1,-1),(0,-1),(-1,-1),(-1,0),(-1,1),
#                   (0,2),(1,2),(2,2),(2,1),(2,0),(2,-1),(2,-2),(1,-2),(0,-2),
#                   (-1,-2),(-2,-2),(-2,-1),(-2,0),(-2,1),(-2,2),(-1,2)]


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

def gt_sem_map(current_episodes):
    # current_episodes = envs.get_current_episodes()[e]
    scene_name = current_episodes['scene_id']
    scene_name = os.path.basename(scene_name).split(".")[0]
    scene_data_file_path = dataset_file_path + scene_name + ".h5"
    goal_idx = current_episodes["object_ids"][0]
    floor_idx = 0
    scene_info = dataset_info_1[scene_name]
    shape_of_gt_map = scene_info[floor_idx]["sem_map"].shape
    f = h5py.File(scene_data_file_path, "r")
    if scene_name=="Corozal":
        sem_map=f['0/map_semantic'][()].transpose()
    else:
        sem_map=f['1/map_semantic'][()].transpose()
        
    w1, h1 = int(sem_map.shape[0]/2), int(sem_map.shape[1]/2)
    w2, h2 = int(shape_of_gt_map[1]/2), int(shape_of_gt_map[2]/2)
    sem_map1 = sem_map[w1-w2:w1+w2,h1-h2:h1+h2]
    central_pos = dataset_info[scene_name]["central_pos"]
    map_world_shift = dataset_info[scene_name]["map_world_shift"]
    map_obj_origin = scene_info[floor_idx]["origin"]
    min_x, min_y = map_obj_origin / 100.0
    pos = current_episodes["start_position"]
    rot = quaternion.from_float_array(current_episodes["start_rotation"])
    x, y, o = convert_3d_to_2d_pose(pos, rot)
    start_x, start_y = int((-y - min_y) * 20.0), int((-x - min_x) * 20.0)
    sem_map2 = map_conversion(sem_map1, start_x, start_y, o)
    goal_loc = (sem_map2 == goal_idx+5.0)
    return goal_loc, sem_map2

def map_conversion(sem_map, start_x, start_y, start_o):
    output_map = np.zeros((LOCAL_MAP_SIZE, LOCAL_MAP_SIZE))
    sin = math.sin(np.pi*1 - start_o)
    cos = math.cos(np.pi*1 - start_o)
    for i in range(18): 
        loc = np.where(sem_map==i)
        if len(loc[0]) == 0:
            continue
        a = loc[0] - start_x
        b = loc[1] - start_y
        loc_conversion = (a * cos + b * sin).astype(np.int) + LOCAL_MAP_SIZE//2, (b * cos - a * sin).astype(np.int) + LOCAL_MAP_SIZE//2
        loc_conversion = void_out_of_boundary(loc_conversion)
        if len(loc_conversion[0]) == 0:
            continue
        if i == 0:
            pass
        elif i == 1:
            color_index = 2
            output_map[loc_conversion] = color_index
        elif i == 2:
            color_index = 1
            output_map[loc_conversion] = color_index
        else:
            color_index = i+2
            output_map[loc_conversion] = color_index
    output_map = signal.medfilt(output_map, 3)
    return output_map

def map_conversion_old(sem_map, start_x, start_y, start_o):
    output_map = np.zeros((LOCAL_MAP_SIZE, LOCAL_MAP_SIZE))
    sin = math.sin(np.pi - start_o)
    cos = math.cos(np.pi - start_o)
    for i in range(sem_map.shape[0]):
        loc = np.nonzero(sem_map[i])
        a = loc[0] - start_x
        b = loc[1] - start_y
        loc_conversion = (a * cos + b * sin).astype(np.int) + LOCAL_MAP_SIZE//2, (b * cos - a * sin).astype(np.int) + LOCAL_MAP_SIZE//2
        loc_conversion = void_out_of_boundary(loc_conversion)
        if len(loc_conversion[0]) == 0:
            continue
        if i == 0:
            color_index = i+2.0 
        else:
            color_index = i+4.0
        output_map[loc_conversion] = color_index
    output_map = signal.medfilt(output_map, 3)
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

def gt_stg(goal_loc, sem_map, current_loc):
    # distance map
    exp_map = np.zeros_like(sem_map)
    exp_map[sem_map==2] = 1
    # selem = skimage.morphology.disk(5)
    # traversible = cv2.dilate(exp_map, selem)
    planner = FMMPlanner(exp_map, step_size=5)
    selem = skimage.morphology.disk(
        int(OBJECT_BOUNDARY * 100.0 / MAP_RESOLUTION)
    )
    goal_map = np.zeros_like(sem_map)
    goal_map[goal_loc] = 1
    goal_map = cv2.dilate(goal_map, selem)
    planner.set_multi_goal(goal_map, validate_goal=True)
    dist_map = planner.fmm_dist
    # Circle
    circle_o = np.zeros_like(sem_map)
    circle_o[current_loc[0],current_loc[1]] = 1
    circle_o = cv2.dilate(circle_o, skimage.morphology.disk(80))
    mx_circle = ma.masked_array(exp_map, 1-circle_o)
    mx_circle = ma.filled(mx_circle, 0)
    ret, labels = cv2.connectedComponents(mx_circle.astype(np.int8))
    label_of_current_loc = labels[current_loc[0], current_loc[1]]
    if not label_of_current_loc:
        for s in location_shift:
            label_of_current_loc = labels[current_loc[0]+s[0], current_loc[1]+s[1]]
            if label_of_current_loc:
                break
    
    selected_index = (labels == label_of_current_loc)
    selected_map = np.zeros_like(sem_map)
    selected_map[selected_index]=1
    if not label_of_current_loc:
        selected_map = mx_circle
    circle_dist_map = ma.masked_array(dist_map, 1-selected_map)
    circle_dist_map = ma.filled(circle_dist_map, np.argmax(circle_dist_map))
    m = np.argmin(circle_dist_map)
    r, c = divmod(m, circle_dist_map.shape[1])
    
    dist_map_vis = cv2.applyColorMap(circle_dist_map.astype(np.uint8), cv2.COLORMAP_JET)
    dist_map_vis[r-3:r+3, c-3:c+3] = 100
    dist_map_vis[circle_dist_map>2.0] = 1
    
    # cv2.imwrite('..semexp/sxz/img/dist_circle_test1.png',np.flipud(dist_map_vis), [int(cv2.IMWRITE_JPEG_QUALITY), 95])

    return r, c, selected_map

def gt_stg_v2(goal_loc, sem_map, current_loc):
    goal_map = np.zeros_like(sem_map)
    goal_map[goal_loc==True] = 1
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(goal_map.astype(np.int8),connectivity=4,ltype=None)
    centroids = np.rint(centroids,).astype(np.int)
    dist = np.sum(np.square(centroids[1:]-np.array(current_loc)),axis=1)
    target_index = dist.argsort()
    target_locs = centroids[target_index+1]
    # target_locs = target_locs[np.lexsort(target_locs.T)]
    # sem_map_2 = np.zeros_like(sem_map)
    # sem_map_2[sem_map==2] = 1
    return target_locs

def cosVector(x,y):
    if(len(x)!=len(y)):
        # print('error input,x and y is not in the same space')
        return;
    result1=0.0;
    result2=0.0;
    result3=0.0;
    for i in range(len(x)):
        result1+=x[i]*y[i]*0.0001 #10000.0
        result2+=(x[i]*0.01)**2 
        result3+=(y[i]*0.01)**2 
    return (result1/((result2*result3)**0.5)) 


class Location_Check(object):
    
    def __init__(self, length=15):
        self.length = length
        self.locations = np.array([[0,0] for i in range(self.length)])
        self.deadlock = False
        self.bad_target_loc = []
        self.prev_tar_loc = np.zeros((2))
        
    def add(self, loc):
        loc = np.array(loc)
        self.locations = np.concatenate((self.locations[1:], np.expand_dims(loc, axis=0)), axis=0)
    
    def check_deadlock(self):
        avg_loc = np.average(self.locations, axis=0)
        dist = np.sqrt(np.sum(np.square(self.locations-avg_loc),axis=1))
        if dist.max() < 4.0 and dist.max()>=0.0:
            self.deadlock = True
            self.locations = np.concatenate((self.locations[1:], np.expand_dims([0,0], axis=0)), axis=0)
        else:
            self.deadlock = False
        return self.deadlock
    
    def reset(self):
        self.locations = np.array([[0,0] for i in range(self.length)])
        self.deadlock = False
        self.bad_target_loc = []
        self.prev_tar_loc = np.zeros((2))
    
    # region
    # def set_target_loc(self, all_locs):
    #     if self.prev_tar_loc[0]==0 and self.prev_tar_loc[1]==0:
    #         self.prev_tar_loc = all_locs[0]
    #     if self.deadlock:
    #         self.bad_target_loc.append(self.prev_tar_loc)
    #         for loc in all_locs:
    #             if not np.any(self.bad_target_loc==loc):
    #                 self.prev_tar_loc = loc
    #                 break
    #     return self.prev_tar_loc
    
    # def set_target_loc_v2(self, t_pfs, t_area_pfs, idx, agent_loc, area_num, thr):
    #     t_pfs=torch.where(t_pfs>thr, 1.0, 0.0)
    #     _, ta_x, ta_y = t_area_pfs.shape 
    #     t_area = t_area_pfs[0].resize(ta_x * ta_y)
    #     _, sorted_idx = torch.sort(t_area)
    #     area_loc = torch.tensor([[sorted_idx[-1]//ta_y, sorted_idx[-1]%ta_y]])
    #     for i in range(2, area_num+1):
    #         area_loc = torch.cat((area_loc, torch.tensor([[sorted_idx[-i]//ta_y, sorted_idx[-i]%ta_y]])), 0)
            
    #     cn = idx + 2
    #     cat_semantic_map = t_pfs[cn].resize(t_pfs.shape[1] * t_pfs.shape[2])
    #     if cat_semantic_map.sum()==0.0:
    #         x = torch.where(t_area_pfs[0]==torch.max(t_area_pfs[0]))
    #         self.prev_tar_loc[0]=area_loc[0][0]
    #         self.prev_tar_loc[1]=area_loc[0][1]
    #         return self.prev_tar_loc
        
    #     # tar_loc = torch.tensor([torch.argmax(cat_semantic_map)//t_pfs.shape[2], torch.argmax(cat_semantic_map)%t_pfs.shape[2]])
    #     t_pfs_1 =t_pfs[cn]
    #     tar_loc = torch.tensor([torch.where(t_pfs_1==torch.max(t_pfs_1))[0][0], torch.where(t_pfs_1==torch.max(t_pfs_1))[1][0]])
        
    #     cos_v = area_loc.mul(tar_loc.unsqueeze(0)).sum(1)/(area_loc.square().sum(1).sqrt() * tar_loc.square().sum().sqrt())
    #     # idx = torch.argmax(cos_v)
    #     idx=0
    #     for i in range(1, area_num):
    #         if cos_v[i]>cos_v[idx]:
    #             idx=i
    #     results = area_loc[idx]

    #     self.prev_tar_loc[0]=results[0]
    #     self.prev_tar_loc[1]=results[1]
    #     return self.prev_tar_loc
    
    # def set_target_loc_v2(self, t_pfs, t_area_pfs, idx, agent_loc, area_num, thr):
    #     t_pfs = (torch.from_numpy(t_pfs)>thr).bool().cpu().numpy()
    #     # t_area = np.zeros_like((t_area_pfs.shape[1] * t_area_pfs.shape[2]))
    #     cat_semantic_map = np.zeros_like((t_pfs.shape[1], t_pfs.shape[2]))
    #     t_area = t_area_pfs[0].cpu().numpy().reshape(t_area_pfs.shape[1] * t_area_pfs.shape[2])
    #     cn = idx + 2
    #     cat_semantic_map = t_pfs[cn]
            
    #     tar_loc1 = np.where(cat_semantic_map == np.max(cat_semantic_map))
    #     tar_loc = [tar_loc1[0][0], tar_loc1[1][0]]
    #     tmp = 0
    #     self.prev_tar_loc[0]=tar_loc[0]
    #     self.prev_tar_loc[1]=tar_loc[1]

    #     sorted_idx = np.argsort(t_area)
    #     area_loc = [[sorted_idx[-1]//t_area_pfs.shape[2], sorted_idx[-1]%t_area_pfs.shape[2]]]
    #     for i in range(2, area_num+1):
    #         area_loc = np.concatenate((area_loc, [[sorted_idx[-i]//t_area_pfs.shape[2], sorted_idx[-i]%t_area_pfs.shape[2]]]), 0)
            
    #     if cat_semantic_map.sum()==0.0:
    #         self.prev_tar_loc=area_loc[0]
    #         return self.prev_tar_loc
    #     x = np.dot(area_loc, tar_loc)
    #     z = np.square(area_loc)
    #     y =np.sum(z,1)
    #     cos_v = x/np.sqrt(y)
    #     idx = np.argmax(cos_v)
    #     results = area_loc[idx]
        
    #     self.prev_tar_loc[0]=results[0]
    #     self.prev_tar_loc[1]=results[1]
    #     return self.prev_tar_loc
    # endregion
    
    
    def set_target_loc_v2(self, t_pfs, t_area_pfs, idx, agent_loc, area_num, thr):
        # t_pfs[np.where(t_pfs<thr)] = 0
        # cat_semantic_map = np.zeros_like((t_pfs.shape[1], t_pfs.shape[2]))
        t_area = t_area_pfs[0]#.cpu().numpy()
        cn = idx + 2
        cat_semantic_map = t_pfs[cn]
        cat_semantic_map[np.where(cat_semantic_map<thr)] = 0
        if cat_semantic_map.sum()==0.0:
            area_loc = torch.where(t_area == torch.max(t_area))
            self.prev_tar_loc[0]=area_loc[0][0]
            self.prev_tar_loc[1]=area_loc[1][0]
            return self.prev_tar_loc
            
        tar_loc = np.where(cat_semantic_map == np.max(cat_semantic_map))
        tmp = -1
        flag = 0
        self.prev_tar_loc[0]=tar_loc[0][0]
        self.prev_tar_loc[1]=tar_loc[1][0]
        # return self.prev_tar_loc
        for i in range(area_num):
            area_loc = torch.where(t_area == torch.max(t_area))
            t_area[area_loc]= 0
            x = cosVector([tar_loc[0][0]-agent_loc[0],tar_loc[1][0]-agent_loc[1]],[area_loc[0][0]-agent_loc[0],area_loc[1][0]-agent_loc[1]])
            if (x>tmp or flag==0):
                flag = 1
                tmp = x
                self.prev_tar_loc[0]=area_loc[0][0]
                self.prev_tar_loc[1]=area_loc[1][0]
        return self.prev_tar_loc
    
    
    def set_target_loc_v3(self, t_area_pfs): # use area policy only
        # t_area = np.zeros_like((t_area_pfs.shape[1], t_area_pfs.shape[2]))
        t_area = t_area_pfs[0]
        area_loc = torch.where(t_area == torch.max(t_area))
        self.prev_tar_loc[0]=area_loc[0][0]
        self.prev_tar_loc[1]=area_loc[1][0]
        return self.prev_tar_loc
