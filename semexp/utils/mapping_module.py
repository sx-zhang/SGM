import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

# from agent.utils.model import get_grid, ChannelPool, Flatten, NNBase
# import agent.utils.depth_utils as du

import semexp.envs.utils.depth_utils as du
from semexp.utils.model import get_grid, ChannelPool, Flatten, NNBase


class Semantic_Mapping(nn.Module):

    """
    Semantic_Mapping
    """

    def __init__(self, args):
        super(Semantic_Mapping, self).__init__()

        self.device = args.device
        self.screen_h = args.frame_height
        self.screen_w = args.frame_width
        self.resolution = args.map_resolution
        self.z_resolution = args.map_resolution
        self.map_size_cm = args.map_size_cm // args.global_downscaling
        self.n_channels = 3
        self.vision_range = args.vision_range
        self.dropout = 0.5
        self.fov = args.hfov
        self.du_scale = args.du_scale
        self.args = args
        self.cat_pred_threshold = args.cat_pred_threshold
        self.exp_pred_threshold = args.exp_pred_threshold
        self.map_pred_threshold = args.map_pred_threshold
        self.num_sem_categories = 3 + 2 + args.use_gt_mask # args.num_sem_categories + args.num_sem_categories_for_exp

        self.max_height = int(360 / self.z_resolution)
        self.min_height = int(-40 / self.z_resolution)
        self.agent_height = args.camera_height * 100.
        self.shift_loc = [self.vision_range *
                          self.resolution // 2, 0, np.pi / 2.0]
        self.camera_matrix = du.get_camera_matrix(
            self.screen_w, self.screen_h, self.fov)

        self.pool = ChannelPool(1)

        vr = self.vision_range

        self.init_grid = torch.zeros(
            1, 1 + self.num_sem_categories, vr, vr,
            self.max_height - self.min_height
        ).float().to(self.device)
        self.feat = torch.ones(
            1, 1 + self.num_sem_categories,
            self.screen_h // self.du_scale * self.screen_w // self.du_scale
        ).float().to(self.device)

        self.local_grid_w = args.map_size_cm // args.map_resolution // args.global_downscaling // args.grid_resolution
        self.local_grid_h = self.local_grid_w
        self.grid_nc = args.record_frames + args.record_angle

    def forward(self, obs, pose_obs, maps_last, poses_last,agent_states):
        pose_obs = pose_obs[None,:]
        maps_last = maps_last[None,:]
        poses_last = poses_last[None,:]
        bs, c, h, w = obs.size()
        depth = obs[:, 3, :, :]

        point_cloud_t = du.get_point_cloud_from_z_t(
            depth, self.camera_matrix, self.device, scale=self.du_scale)

        agent_view_t = du.transform_camera_view_t(
            point_cloud_t, self.agent_height, 0, self.device)

        agent_view_centered_t = du.transform_pose_t(
            agent_view_t, self.shift_loc, self.device)

        max_h = self.max_height
        min_h = self.min_height
        xy_resolution = self.resolution
        z_resolution = self.z_resolution
        vision_range = self.vision_range
        XYZ_cm_std = agent_view_centered_t.float()
        XYZ_cm_std[..., :2] = (XYZ_cm_std[..., :2] / xy_resolution)
        XYZ_cm_std[..., :2] = (XYZ_cm_std[..., :2] -
                               vision_range // 2.) / vision_range * 2.
        XYZ_cm_std[..., 2] = XYZ_cm_std[..., 2] / z_resolution
        XYZ_cm_std[..., 2] = (XYZ_cm_std[..., 2] -
                              (max_h + min_h) // 2.) / (max_h - min_h) * 2.
        self.feat[:, 1:, :] = nn.AvgPool2d(self.du_scale)(
            obs[:, 4:, :, :]
        ).view(bs, c - 4, h // self.du_scale * w // self.du_scale)


        XYZ_cm_std = XYZ_cm_std.permute(0, 3, 1, 2)
        XYZ_cm_std = XYZ_cm_std.view(XYZ_cm_std.shape[0],
                                     XYZ_cm_std.shape[1],
                                     XYZ_cm_std.shape[2] * XYZ_cm_std.shape[3])

        voxels = du.splat_feat_nd(
            self.init_grid * 0., self.feat, XYZ_cm_std).transpose(2, 3)

        min_z = int(25 / z_resolution - min_h)
        max_z = int((self.agent_height + 1) / z_resolution - min_h)

        agent_height_proj = voxels[..., min_z:max_z].sum(4)
        all_height_proj = voxels.sum(4)

        # suspect that it is in here we deal with mask and confidence scores

        fp_map_pred = agent_height_proj[:, 0:1, :, :]
        fp_exp_pred = all_height_proj[:, 0:1, :, :]
        fp_map_pred = fp_map_pred / self.map_pred_threshold
        fp_exp_pred = fp_exp_pred / self.exp_pred_threshold
        fp_map_pred = torch.clamp(fp_map_pred, min=0.0, max=1.0)
        fp_exp_pred = torch.clamp(fp_exp_pred, min=0.0, max=1.0)

        pose_pred = poses_last

        agent_view = torch.zeros(bs, c,
                                 self.map_size_cm // self.resolution,
                                 self.map_size_cm // self.resolution
                                 ).to(self.device)

        x1 = self.map_size_cm // (self.resolution * 2) - self.vision_range // 2
        x2 = x1 + self.vision_range
        y1 = self.map_size_cm // (self.resolution * 2)
        y2 = y1 + self.vision_range
        agent_view[:, 0:1, y1:y2, x1:x2] = fp_map_pred
        agent_view[:, 1:2, y1:y2, x1:x2] = fp_exp_pred
        # look at this line
        agent_view[:, 4:, y1:y2, x1:x2] = torch.clamp(
            agent_height_proj[:, 1:, :, :] / self.cat_pred_threshold,
            min=0.0, max=1.0)

        corrected_pose = pose_obs

        def get_new_pose_batch(pose, rel_pose_change):

            pose[:, 1] += rel_pose_change[:, 0] * \
                torch.sin(pose[:, 2] / 57.29577951308232) \
                + rel_pose_change[:, 1] * \
                torch.cos(pose[:, 2] / 57.29577951308232)
            pose[:, 0] += rel_pose_change[:, 0] * \
                torch.cos(pose[:, 2] / 57.29577951308232) \
                - rel_pose_change[:, 1] * \
                torch.sin(pose[:, 2] / 57.29577951308232)
            pose[:, 2] += rel_pose_change[:, 2] * 57.29577951308232

            pose[:, 2] = torch.fmod(pose[:, 2] - 180.0, 360.0) + 180.0
            pose[:, 2] = torch.fmod(pose[:, 2] + 180.0, 360.0) - 180.0

            return pose

        current_poses = get_new_pose_batch(poses_last, corrected_pose)
        st_pose = current_poses.clone().detach()

        st_pose[:, :2] = - (st_pose[:, :2]
                            * 100.0 / self.resolution
                            - self.map_size_cm // (self.resolution * 2)) /\
            (self.map_size_cm // (self.resolution * 2))
        st_pose[:, 2] = 90. - (st_pose[:, 2])

        rot_mat, trans_mat = get_grid(st_pose, agent_view.size(),
                                      self.device)

        rotated = F.grid_sample(agent_view, rot_mat, align_corners=True)
        translated = F.grid_sample(rotated, trans_mat, align_corners=True)
        t2 = translated.unsqueeze(1)

        for i in range(4,4+2+2+self.args.use_gt_mask):
            k = torch.max(self.feat[0,i-3,:])
            t2[0,0,i,:,:][t2[0,0,i,:,:]>0.0] = k
        #t2[t2>0.0] = 0.88
        maps2 = torch.cat((maps_last.unsqueeze(1), t2), 1)
        #total view: correspond to exp, channel 1
        if self.args.record_frames == 2:
            agent_states.local_grid[0] += torch.clone(nn.MaxPool2d(self.args.grid_resolution)(
                t2[0, 0, 1:2, :, :])[0])
            # goal item view: correspond to goal, channel 4
            agent_states.local_grid[1] += torch.clone(nn.MaxPool2d(self.args.grid_resolution)(
                t2[0, 0, 4:5, :, :])[0])
        #view angle: haha
        if self.args.record_angle == 2:
            coordinates = torch.nonzero(agent_states.local_grid[1])

            def get_grid_rc(pose):
                r, c = pose[0, 1], pose[0, 0]
                loc_r, loc_c = [int(r * 100.0 / self.args.map_resolution),
                                int(c * 100.0 / self.args.map_resolution)]
                return loc_r // self.args.grid_resolution, loc_c // self.args.grid_resolution

            if len(coordinates) != 0:
                r, c = get_grid_rc(current_poses)
                for i in range(len(coordinates)):
                    r2,c2 = coordinates[i]
                    y,x = r-r2,c-c2
                    ny = y+0.001
                    nx = x + 0.001
                    angle = torch.atan(ny/nx)
                    if x<0:
                        angle += 3.14
                    elif y<0:
                        angle += 6.28
                    agent_states.local_grid[2,r2,c2] = torch.min(agent_states.local_grid[2,r2,c2],angle)
                    agent_states.local_grid[3,r2,c2] = torch.max(agent_states.local_grid[3,r2,c2],angle)


        map_pred, _ = torch.max(maps2, 1)
        agent_states.local_grid[5,:,:] = torch.clone(nn.MaxPool2d(self.args.grid_resolution)(
            map_pred[0, 4:5, :, :])[0])

        return fp_map_pred[0], map_pred[0], pose_pred[0], current_poses[0]
