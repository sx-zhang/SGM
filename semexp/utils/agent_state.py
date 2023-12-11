from collections import deque, defaultdict
import os
import logging
import time
import json
import gym
import torch.nn as nn
import torch
import numpy as np
import pickle

# from agent.mapping_module import Semantic_Mapping
from semexp.utils.mapping_module import Semantic_Mapping
# from constants import habitat_goal_label_to_similar_coco
from arguments import get_args
import pickle



class Agent_State:
    def __init__(self,args):
        self.args = args
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        self.nc = 2 + 4 + 1 + 2 # + self.args.use_gt_mask# num channels
        self.gt_mask_channel = 8
        if args.cuda:
            torch.cuda.manual_seed(args.seed)
        self.device = torch.device("cuda:0" if args.cuda else "cpu")

        self.map_size = args.map_size_cm // args.map_resolution
        self.full_w, self.full_h = self.map_size, self.map_size
        self.local_w = int(self.full_w / args.global_downscaling)
        self.local_h = int(self.full_h / args.global_downscaling)
        self.grid_w, self.grid_h = self.full_w//args.grid_resolution, self.full_h//args.grid_resolution
        self.grid_nc = args.record_frames + args.record_angle + 1 + 1 # blacklist and max score


        # In the actual implementation, we have two maps with different granularities to save time
        # self.grid stores information related to object goal identification and has bigger grid,
        # since we need to run flood-fill algorithm on it and we want to save time

        # self.full_map stores both obstacle related information and goal related information.
        # it has smaller grid since it is used in path planning and requires higher precision.

        # Dimensions
        # 0: number of frames, Channel 4 in the paper
        # 1: cumulative confidence, Channel 5 in the paper
        # 2, 3: not used in final implementation
        # 5: max confidence, Channel 6 in the paper.
        self.grid = torch.zeros(self.grid_nc,self.grid_w,self.grid_h).float().to(self.device)
        self.local_grid_w =self.grid_w // args.global_downscaling
        self.local_grid_h = self.local_grid_w
        self.local_grid = torch.zeros(self.grid_nc, self.local_grid_w,
                                      self.local_grid_h).float().to(self.device)


        # Initializing full and local map
        # dimensions:
        # 0: depth collision, corresponding to Channel 1 in the paper
        # 1: exploration coverage
        # 4: goal category, corresponding to Channel 6 in the paper
        # 5: conflict category, corresponding to Channel 7 in the paper
        # The rest of the dimensions of self.full_map are not used in the final implementation
        self.full_map = torch.zeros(self.nc, self.full_w, self.full_h).float().to(
            self.device)

        self.local_map = torch.zeros(self.nc, self.local_w,
                                self.local_h).float().to(self.device)
        self.global_goal_loc = torch.zeros(self.full_w,self.full_h)
        self.global_goal_index = (-1,-1)

        # Initial full and local pose
        self.full_pose = torch.zeros( 3).float().to(self.device)
        self.local_pose = torch.zeros( 3).float().to(self.device)

        # Origin of local map
        self.origins = np.zeros((3))

        # Local Map Boundaries
        self.lmb = np.zeros(( 4)).astype(int)

        # Planner pose inputs has 7 dimensions
        # 1-3 store continuous global agent location
        # 4-7 store local map boundaries
        self.planner_pose_inputs = np.zeros(( 7))

        # Global policy observation space
        self.es = 2
        self.log_history = []
        self.goal_log = []
        self.hard_goal = False

        # Semantic Mapping
        self.sem_map_module = Semantic_Mapping(args).to(self.device)
        self.sem_map_module.eval()
        self.global_goal_preset = (0.1,0.1)
        self.global_goal_rotation = [(0.1,0.1),(0.9,0.1),(0.9,0.9),(0.1,0.9)]
        self.global_goal_rotation_id = 0
        self.cat_semantic_map = torch.zeros(self.local_w, self.local_h)
        self.local_grid_vis = torch.zeros(self.local_grid_w,self.local_grid_h)
        self.score_threshold = None

        self.stuck = False
        # if args.detect_stuck == 1:
        #     self.pos_record = []

    def reset(self):
        self.l_step = 0
        self.g_step = 0
        self.step = 0
        self.avg_goal_conf = 0
        self.avg_conf_conf = 0
        self.num_conf = 0
        self.goal_cat = -1
        self.found_goal = False
        self.hard_goal = False
        self.global_goal_rotation_id = 0
        self.global_goal_preset = (0.1,0.1)
        self.goal_log = []
        self.score_threshold = 0.85
        self.init_map_and_pose()
        self.stuck = False
        # if self.args.detect_stuck == 1:
        #     self.pos_record = []

    def save_conf_stat(self,suc,epi_length, epi_ID,gt_found = False,step = -1):
        if self.num_conf == 0:
            self.avg_goal_conf = -1
            self.avg_conf_conf = -1
        else:
            self.avg_goal_conf /= self.num_conf
            self.avg_conf_conf /= self.num_conf
        self.log_history.append({
            "suc":suc,
            "epi_len":epi_length,
            "epi_ID":epi_ID,
            "goal":self.goal_cat,
            "self_suc":self.found_goal,
            "goal_score":self.avg_goal_conf,
            "conf_score":self.avg_conf_conf,
            "gt_found": gt_found,
            "goal_log": self.goal_log,
            "stuck": self.stuck,
            "step_found":step
        })
        with open(self.args.log_path, 'wb') as handle:
            pickle.dump(self.log_history, handle, protocol=pickle.HIGHEST_PROTOCOL)
        if self.num_conf == 0:
            return

    def inc_step(self):
        args = self.args
        self.l_step += 1
        self.step += 1
        self.l_step = self.step % args.num_local_steps
    
    def init_with_obs(self,obs,infos):

        self.l_step = 0
        self.step = 0

        self.poses = torch.from_numpy(np.asarray(
            infos['sensor_pose'])
        ).float().to(self.device)
        _, self.local_map, _, self.local_pose = \
            self.sem_map_module(obs, self.poses, self.local_map, self.local_pose,self)

        # Compute Global policy input
        self.locs = self.local_pose.cpu().numpy()

        r, c = self.locs[1], self.locs[0]
        loc_r, loc_c = [int(r * 100.0 / self.args.map_resolution),
                        int(c * 100.0 / self.args.map_resolution)]

        self.local_map[2:4, loc_r - 1:loc_r + 2,
        loc_c - 1:loc_c + 2] = 1.

        self.global_goals = [[int(0.1 * self.local_w), int(0.1 * self.local_h)]]
        self.global_goals = [[min(x, int(self.local_w - 1)), min(y, int(self.local_h - 1))]
                        for x, y in self.global_goals]

        self.goal_maps = np.zeros((self.local_w, self.local_h))


        self.goal_maps[self.global_goals[0][0], self.global_goals[0][1]] = 1


        p_input = {}

        p_input['map_pred'] = self.local_map[0, :, :].cpu().numpy()
        p_input['exp_pred'] = self.local_map[1, :, :].cpu().numpy()
        p_input['pose_pred'] = self.planner_pose_inputs
        p_input['goal'] = self.goal_maps  # global_goals[e]
        p_input['new_goal'] = 1
        p_input['found_goal'] = 0
        p_input['wait'] = 0  # does it matter?
        if self.args.visualize or self.args.print_images:
            self.local_map[-1, :, :] = 1e-5
            p_input['sem_map_pred'] = self.local_map[4:, :, :
                                      ].argmax(0).cpu().numpy()


        self.planner_inputs = p_input


        torch.set_grad_enabled(False)

    def get_local_map_boundaries(self, agent_loc, local_sizes, full_sizes):
        loc_r, loc_c = agent_loc
        local_w, local_h = local_sizes
        full_w, full_h = full_sizes

        if self.args.global_downscaling > 1:
            gx1, gy1 = loc_r - local_w // 2, loc_c - local_h // 2
            gx1, gy1 = gx1 - gx1%self.args.grid_resolution, gy1 - gy1%self.args.grid_resolution
            gx2, gy2 = gx1 + local_w, gy1 + local_h
            if gx1 < 0:
                gx1, gx2 = 0, local_w
            if gx2 > full_w:
                gx1, gx2 = full_w - local_w, full_w

            if gy1 < 0:
                gy1, gy2 = 0, local_h
            if gy2 > full_h:
                gy1, gy2 = full_h - local_h, full_h
        else:
            gx1, gx2, gy1, gy2 = 0, full_w, 0, full_h

        return [gx1, gx2, gy1, gy2]

    def init_map_and_pose(self):
        args = self.args
        self.full_map.fill_(0.)
        self.full_pose.fill_(0.)
        self.global_goal_loc.fill_(0.)


        if self.args.record_angle + self.args.record_frames == 4:
            self.grid[0:2].fill_(0.)
            self.grid[2].fill_(6.28)
            self.grid[3].fill_(0.)
            self.grid[4:6].fill_(0.)

            self.local_grid[0:2].fill_(0.)
            self.local_grid[2].fill_(6.28)
            self.local_grid[3].fill_(0.)
            self.local_grid[4:6].fill_(0.)



        self.full_pose[:2] = self.args.map_size_cm / 100.0 / 2.0

        locs = self.full_pose.cpu().numpy()
        self.planner_pose_inputs[:3] = locs

        r, c = locs[1], locs[0]
        loc_r, loc_c = [int(r * 100.0 / args.map_resolution),
                        int(c * 100.0 / args.map_resolution)]

        self.full_map[2:4, loc_r - 1:loc_r + 2, loc_c - 1:loc_c + 2] = 1.0

        self.lmb = self.get_local_map_boundaries((loc_r, loc_c),
                                                 (self.local_w, self.local_h),
                                                 (self.full_w, self.full_h))

        self.planner_pose_inputs[3:] = self.lmb
        self.origins = np.array([self.lmb[2] * args.map_resolution / 100.0,
                        self.lmb[0] * args.map_resolution / 100.0, 0.])

        self.local_map = self.full_map[:,
                         self.lmb[0]:self.lmb[1],
                         self.lmb[2]:self.lmb[3]]
        self.local_pose = self.full_pose - \
                          torch.from_numpy(self.origins).to(self.device).float()

    def set_hard_goal(self):
        self.hard_goal = True

    def recur_fill(self,i,j):
        if i < 0 or j < 0 or i >= self.full_w or j >= self.full_h:
            return
        if self.global_goal_loc[i,j] != 0 or self.full_map[4,i,j] <= 0.1:
            return
        self.global_goal_loc[i,j] = 1
        for di in range(-1,1):
            for dj in range(-1,1):
                self.recur_fill(i+di,j+dj)

    def recur_fill_grid(self,i,j):
        if i < 0 or j < 0 or i >= self.local_grid_w or j >= self.local_grid_h:
            return True
        if self.local_grid_vis[i,j] != 0 or self.local_grid[5,i,j] < self.score_threshold-0.01: # I feel like it's the equal sign here
            return True
        if self.local_grid[4,i,j] == 1:
            return False
        self.local_grid_vis[i,j] = 1
        for di in range(-1,1):
            for dj in range(-1,1):
                if self.recur_fill(i+di,j+dj) == False:
                    self.local_grid[4,i,j] = 1
                    return False
        r1, r2 = i * self.args.grid_resolution, (
                i + 1) * self.args.grid_resolution
        c1, c2 = j * self.args.grid_resolution, (
                j + 1) * self.args.grid_resolution
        self.cat_semantic_map[r1:r2, c1:c2] = self.local_map[ 4, r1:r2, c1:c2]
        return True

    def save_global_goal(self):
        self.found_goal = True
        self.full_map[ 4, self.lmb[ 0]:self.lmb[1],
        self.lmb[2]:self.lmb[3]] = \
            self.local_map[4]
        max_index = np.unravel_index(torch.argmax(self.full_map[ 4, :, :]).cpu().numpy(),
                                     (self.full_w,self.full_h))
        self.global_goal_index = max_index
        self.recur_fill(max_index[0],max_index[1])

    # it has to be scattered large
    def global_to_local(self):
        r1,r2,c1,c2 = self.lmb[0],self.lmb[1],self.lmb[2],self.lmb[3]
        r,c = self.global_goal_index
        goal_maps = np.zeros((self.local_w, self.local_h))
        if r1<=r and r <r2 and c1<=c and c<c2:
            return self.global_goal_loc[r1:r2,c1:c2]
        lr,lc = 0,0
        if r < r1:
            lr = 0
        elif r > r2:
            lr = self.local_w-1
        else:
            lr = r-r1

        if c < c1:
            lc = 0
        elif c > c2:
            lc = self.local_h-1
        else:
            lc = c-c1

        siz = 20
        lr1 = max(0,lr-siz)
        lr2 = min(lr+siz,self.local_w-1)
        lc1 = max(0,lc-siz)
        lc2 = min(lc+siz,self.local_h-1)
        goal_maps[lr1:lr2,lc1:lc2] = 1
        return goal_maps

    def suc_gt_map(self,goalmap,gtmap):
        ind = np.nonzero(goalmap)
        siz = len(ind[0])
        gtsiz = np.sum(gtmap[ind])
        if gtsiz > siz*0.6:
            return True
        else:
            return False

    def get_conflict(self,goalmap,gtmap):
        ind = np.nonzero(goalmap)
        siz = len(ind[0])
        gtsiz = np.sum(gtmap[ind])
        if gtsiz > siz*0.5:
            return np.max(gtmap[ind])
        else:
            return 0

    def get_black_white_conflict(self,goalmap,gtmap):
        ind = np.nonzero(goalmap)
        return np.max(gtmap[ind])

    def clear_goal(self,goalmap):
        #this is local
        self.found_goal = False
        ind = np.nonzero(goalmap)
        ind2 = (ind[0] // self.args.grid_resolution,ind[1]//self.args.grid_resolution)
        self.local_grid[4][ind2] = 1

    def goal_record(self,goalmap):
        ind = np.nonzero(goalmap)
        ind2 = (ind[0] // self.args.grid_resolution,
                ind[1] // self.args.grid_resolution)
        ans = {}
        max_score = -1
        max_cumu = -1
        max_ratio = -1

        #max score
        for i in range(len(ind2[0])):
            n,m = ind2[0][i],ind2[1][i]
            if self.local_grid[5,n,m] > max_score:
                max_score = self.local_grid[5,n,m]
                ans["score"] = self.local_grid[:,n,m].cpu().numpy()

        #highest cumu
        for i in range(len(ind2[0])):
            n,m = ind2[0][i],ind2[1][i]
            if self.local_grid[1,n,m] > max_cumu:
                max_cumu = self.local_grid[1,n,m]
                ans["cumu"] = self.local_grid[:,n,m].cpu().numpy()

        #highest ratio
        for i in range(len(ind2[0])):
            n,m = ind2[0][i],ind2[1][i]
            if self.local_grid[1,n,m]/self.local_grid[0,n,m] > max_ratio:
                max_ratio = self.local_grid[1,n,m]/self.local_grid[0,n,m]
                ans["ratio"] = self.local_grid[:,n,m].cpu().numpy()
        ans["total"] = {"score":float(max_score.cpu().numpy()),"cumu":float(max_cumu.cpu().numpy()),"ratio":float(max_ratio.cpu().numpy())}
        ans["suc"] = self.suc_gt_map(goalmap,self.local_map[self.gt_mask_channel,:,:].cpu().numpy()) if self.args.use_gt_mask else -1
        ans["step"] = self.step
        if self.args.record_conflict == 1:
            ans["conflict"] = {"normal":self.get_conflict(goalmap,self.local_map[5,:,:].cpu().numpy()),
                               "black":self.get_black_white_conflict(goalmap,self.local_map[6,:,:].cpu().numpy()),
                               "white": self.get_black_white_conflict(goalmap,
                                                                      self.local_map[
                                                                       7, :,
                                                                      :].cpu().numpy())}

        return ans

    def clear_goal_and_set_gt_map(self,goalmap):
        self.clear_goal(goalmap)
        self.goal_log.append(self.goal_record(goalmap))

    def upd_agent_state(self,obs,infos):

        args = self.args
        self.poses = torch.from_numpy(np.asarray(
            infos['sensor_pose'] )
        ).float().to(self.device)

        _, self.local_map, _, self.local_pose = \
            self.sem_map_module(obs, self.poses, self.local_map, self.local_pose,self)

        locs = self.local_pose.cpu().numpy()
        self.planner_pose_inputs[:3] = locs + self.origins
        self.local_map[2, :, :].fill_(0.)  # Resetting current location channel

        r, c = locs[1], locs[0]
        loc_r, loc_c = [int(r * 100.0 / args.map_resolution),
                        int(c * 100.0 / args.map_resolution)]
        self.local_map[2:4, loc_r - 2:loc_r + 3, loc_c - 2:loc_c + 3] = 1.
        if self.args.detect_stuck == 1:
            glo_r, glo_c = self.lmb[0] + loc_r, self.lmb[2] + loc_c
            self.pos_record.append((glo_r, glo_c))
            l = len(self.pos_record)
            if len(self.pos_record) > 110:
                stuck = True
                for i in range(2, 100):
                    dis = abs(glo_r - self.pos_record[l - i][0]) + abs(glo_c - self.pos_record[l - i][1])
                    if dis > 25:
                        stuck = False
                if stuck:
                    self.stuck = True

        if self.l_step == args.num_local_steps - 1:
            self.l_step = 0
            # For every global step, update the full and local maps

            self.full_map[:, self.lmb[0]:self.lmb[1], self.lmb[2]:self.lmb[3]] = \
                self.local_map
            res = self.args.grid_resolution

            self.grid[:, self.lmb[0] // res:self.lmb[1] // res, self.lmb[2] // res: self.lmb[3] // res] = \
                torch.clone(self.local_grid)
            self.full_pose = self.local_pose + \
                             torch.from_numpy(self.origins).to(self.device).float()

            locs = self.full_pose.cpu().numpy()
            r, c = locs[1], locs[0]
            loc_r, loc_c = [int(r * 100.0 / args.map_resolution),
                            int(c * 100.0 / args.map_resolution)]

            self.lmb = self.get_local_map_boundaries((loc_r, loc_c),
                                                     (self.local_w, self.local_h),
                                                     (self.full_w, self.full_h))

            self.planner_pose_inputs[3:] = self.lmb
            self.origins = np.array([self.lmb[2] * args.map_resolution / 100.0,
                            self.lmb[0] * args.map_resolution / 100.0, 0.])

            self.local_map = self.full_map[:,
                             self.lmb[0]:self.lmb[1],
                             self.lmb[2]:self.lmb[3]]
            self.local_pose = self.full_pose - \
                              torch.from_numpy(self.origins).to(self.device).float()
            self.local_grid = torch.clone(self.grid[:, self.lmb[0] // res:self.lmb[1] // res,
                                          self.lmb[2] // res: self.lmb[3] // res])


            locs = self.local_pose.cpu().numpy()
            if self.hard_goal:
                self.global_goal_rotation_id = (self.global_goal_rotation_id + 1)%4
                self.global_goal_preset = self.global_goal_rotation[self.global_goal_rotation_id]
                self.hard_goal = False
            self.global_goals = [[int(self.global_goal_preset[0] * self.local_w),
                             int(self.global_goal_preset[1] * self.local_h)]
                            ]
            self.global_goals = [[min(x, int(self.local_w - 1)),
                             min(y, int(self.local_h - 1))]
                            for x, y in self.global_goals]


        # ------------------------------------------------------------------

        # ------------------------------------------------------------------
        # Update long-term goal if target object is found
        found_goal = 0
        goal_maps = np.zeros((self.local_w, self.local_h))


        goal_maps[self.global_goals[0][0], self.global_goals[0][1]] = 1

        maxi = 0.0
        maxc = -1
        maxa = 0.0
        self.goal_cat = infos['goal_cat_id']
        e = 0
        cn = 4
        # use the grid to determine global goal
        # requires >= 0.75 on the cumulated channel (channel 1)
        # now change to >= 0.85 on the single channel (channel 5)
        if self.args.only_explore == 0:
            max_score = torch.max(self.local_grid[5][self.local_grid[4] == 0])
            if max_score > self.score_threshold:
                indices = torch.nonzero(self.local_grid[5] >= max_score)
                self.cat_semantic_map.fill_(0.)
                self.local_grid_vis.fill_(0.)
                for index in indices:
                    r1 = index[0] * self.args.grid_resolution
                    c1 = index[1] * self.args.grid_resolution
                    r2 = r1 + self.args.grid_resolution
                    c2 = c1 + self.args.grid_resolution
                    if self.local_grid[4, index[0], index[1]] == 0:
                        if self.recur_fill_grid(index[0], index[1]):
                            found_goal = 1

                if found_goal == 1:
                    self.found_goal = True
                    cat_semantic_scores = self.cat_semantic_map.cpu().numpy()
                    cat_semantic_scores[
                        cat_semantic_scores < self.score_threshold - 0.01] = 0.
                    goal_maps = cat_semantic_scores



        # ------------------------------------------------------------------

        # ------------------------------------------------------------------
        # Take action and get next observation
        p_input = {}

        p_input['map_pred'] = self.local_map[0, :, :].cpu().numpy()
        p_input['exp_pred'] = self.local_map[1, :, :].cpu().numpy()
        p_input['pose_pred'] = self.planner_pose_inputs
        p_input['goal'] = goal_maps  # global_goals[e]
        p_input['new_goal'] = self.l_step == args.num_local_steps - 1
        p_input['found_goal'] = found_goal
        p_input['wait'] = 0
        p_input['goal_name'] = infos['goal_name']
        if args.visualize or args.print_images:
            self.local_map[8 + self.args.use_gt_mask, :, :] = 1e-5

            p_input['sem_map_pred'] = self.local_map[4:, :,
                                      :].argmax(0).cpu().numpy()
            p_input['opp_score'] = maxi
            p_input['opp_cat'] = maxc
            p_input['itself'] = maxa

        self.inc_step()
        return p_input
