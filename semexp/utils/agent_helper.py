import math
import os
import cv2
import numpy as np
import skimage.morphology
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import math
# from agent.utils.fmm_planner import FMMPlanner
# from agent.utils.rednet import QuickSemanticPredRedNet
from constants import color_palette
# import agent.utils.pose as pu
# import agent.utils.visualization as vu
from semexp.envs.utils.fmm_planner import FMMPlanner
import semexp.envs.utils.pose as pu
import semexp.agents.utils.visualization as vu
from semexp.utils.rednet import QuickSemanticPredRedNet

# The untrap helper for the bruteforce untrap mode
class UnTrapHelper:
    def __init__(self):
        self.total_id = 0
        self.epi_id = 0

    def reset(self):
        self.total_id += 1
        self.epi_id = 0

    def get_action(self):
        self.epi_id += 1
        if self.epi_id == 1:
            if self.total_id % 2 == 0:
                return 2
            else:
                return 3
        else:
            if self.total_id % 2 == 0:
                return 3
            else:
                return 2



class Agent_Helper:
    """The Sem_Exp environment agent class. A seperate Sem_Exp_Env_Agent class
    object is used for each environment thread.

    """

    def __init__(self, args,agent_states):

        self.args = args


        # initialize transform for RGB observations
        self.res = transforms.Compose(
            [transforms.ToPILImage(),
             transforms.Resize((args.frame_height, args.frame_width),
                               interpolation=Image.NEAREST)])

        # initialize semantic segmentation prediction model
        if args.sem_gpu_id == -1:
            args.sem_gpu_id = 1

        self.sem_pred_rednet = QuickSemanticPredRedNet(args)

        # initializations for planning:
        self.selem = skimage.morphology.disk(3)

        self.obs = None
        self.obs_shape = None
        # Channel 3 in the paper
        self.collision_map = None
        # Channel 2 in the paper
        self.collision_map_big = None
        self.visited_vis = None
        self.col_width = None
        self.curr_loc = None
        self.last_loc = None
        self.last_action = None
        self.count_forward_actions = None
        self.last_start = None
        self.rank = 0
        self.episode_no = 0
        self.mask = None
        self.stg = None
        self.goal_cat = -1
        self.untrap = UnTrapHelper()
        self.use_small_num = 0
        self.agent_states = agent_states

        # We move forward 1 extra step after approaching goal to make the agent closer to goal
        self.forward_after_stop_preset = self.args.move_forward_after_stop
        self.forward_after_stop = self.forward_after_stop_preset

        self.plan_step = None
        self.map_size = args.map_size_cm // args.map_resolution
        self.full_w, self.full_h = self.map_size, self.map_size
        self.local_w = int(self.full_w / args.global_downscaling)
        self.local_h = int(self.full_h / args.global_downscaling)
        self.found_goal = None


        if args.visualize or args.print_images:
            self.legend = cv2.imread('Stubborn/docs/legend.png')
            self.vis_image = None
            self.rgb_vis = None

    def reset(self):
        args = self.args

        self.obs_shape = None

        # Episode initializations
        map_shape = (args.map_size_cm // args.map_resolution,
                     args.map_size_cm // args.map_resolution)
        self.collision_map = np.zeros(map_shape)
        self.collision_map_big = np.zeros(map_shape)
        self.use_small_num = 0
        self.visited_vis = np.zeros(map_shape)
        self.col_width = 1
        self.count_forward_actions = 0
        self.curr_loc = [args.map_size_cm / 100.0 / 2.0,
                         args.map_size_cm / 100.0 / 2.0, 0.]
        self.last_action = None
        self.episode_no += 1
        self.timestep = 0
        self.plan_step = -100
        self.prev_blocked = 0
        self._previous_action = -1
        self.block_threshold = 10
        #self.untrap = UnTrapHelper() #TODO is this needed?
        self.forward_after_stop = self.forward_after_stop_preset

    def set_small_collision_map(self):
        self.use_small_num = 1000

    def plan_act_and_preprocess(self, planner_inputs):
        """Function responsible for planning, taking the action and
        preprocessing observations

        Args:
            planner_inputs (dict):
                dict with following keys:
                    'map_pred'  (ndarray): (M, M) map prediction
                    'goal'      (ndarray): (M, M) mat denoting goal locations
                    'pose_pred' (ndarray): (7,) array denoting pose (x,y,o)
                                 and planning window (gx1, gx2, gy1, gy2)
                     'found_goal' (bool): whether the goal object is found

        Returns:
            obs (ndarray): preprocessed observations ((4+C) x H x W)
            reward (float): amount of reward returned after previous action
            done (bool): whether the episode has ended
            info (dict): contains timestep, pose, goal category and
                         evaluation metric info
        """

        # plan


        # Reset reward if new long-term goal
        self.timestep += 1
        self.goal_name = planner_inputs['goal_name']
        if self.args.visualize or self.args.print_images:
            self.vis_image = vu.init_vis_image(self.goal_name, self.legend)
        if planner_inputs["new_goal"]:
            self.info["g_reward"] = 0

        action = self._plan(planner_inputs)

        if self.args.visualize or self.args.print_images:
            self._visualize(planner_inputs)

        if action >= 0:

            # act
            action = {'action': action}
            self.last_action = action['action']
            return action

    def set_goal_cat(self,goal_cat):
        self.goal_cat = goal_cat

    def preprocess_inputs(self,rgb,depth,info,rew = 0):
        # preprocess obs
        obs = self._preprocess_obs(rgb,depth)
        self.obs = obs
        # self.info = info
        # if 'g_reward' not in info.keys():
            # info['g_reward'] = 0

        # info['g_reward'] += rew
        return obs#, info

    def _plan(self, planner_inputs):
        """Function responsible for planning

        Args:
            planner_inputs (dict):
                dict with following keys:
                    'map_pred'  (ndarray): (M, M) map prediction
                    'goal'      (ndarray): (M, M) goal locations
                    'pose_pred' (ndarray): (7,) array  denoting pose (x,y,o)
                                 and planning window (gx1, gx2, gy1, gy2)
                    'found_goal' (bool): whether the goal object is found

        Returns:
            action (int): action id
        """
        args = self.args

        self.last_loc = self.curr_loc

        # Get Map prediction
        map_pred = np.rint(planner_inputs['map_pred'])
        self.found_goal = planner_inputs['found_goal']
        goal = planner_inputs['goal']


        # Get pose prediction and global policy planning window
        start_x, start_y, start_o, gx1, gx2, gy1, gy2 = \
            planner_inputs['pose_pred']
        gx1, gx2, gy1, gy2 = int(gx1), int(gx2), int(gy1), int(gy2)
        planning_window = [gx1, gx2, gy1, gy2]

        # Get curr loc
        self.curr_loc = [start_x, start_y, start_o]
        r, c = start_y, start_x
        start = [int(r * 100.0 / args.map_resolution - gx1),
                 int(c * 100.0 / args.map_resolution - gy1)]
        start = pu.threshold_poses(start, map_pred.shape)

        # Get last loc
        last_start_x, last_start_y = self.last_loc[0], self.last_loc[1]
        r, c = last_start_y, last_start_x
        last_start = [int(r * 100.0 / args.map_resolution - gx1),
                      int(c * 100.0 / args.map_resolution - gy1)]
        last_start = pu.threshold_poses(last_start, map_pred.shape)
        self.last_start = last_start
        self.visited_vis[gx1:gx2, gy1:gy2] = \
            vu.draw_line(last_start, start,
                         self.visited_vis[gx1:gx2, gy1:gy2])

        # Collision check
        if self.last_action == 1:
            x1, y1, t1 = self.last_loc
            x2, y2, _ = self.curr_loc
            buf = 4
            length = 2

            if abs(x1 - x2) < 0.05 and abs(y1 - y2) < 0.05:
                self.col_width += 2
                if self.col_width == 7:
                    length = 4
                    buf = 3
                self.col_width = min(self.col_width, 3)
            else:
                self.col_width = 1
                # after fix
                #self.col_width = 5

            dist = pu.get_l2_distance(x1, x2, y1, y2)
            if dist < args.collision_threshold:  # Collision
                self.prev_blocked += 1
                width = self.col_width
                for i in range(length):
                    for j in range(width):
                        wx = x1 + 0.05 * \
                            ((i + buf) * np.cos(np.deg2rad(t1))
                             + (j - width // 2) * np.sin(np.deg2rad(t1)))
                        wy = y1 + 0.05 * \
                            ((i + buf) * np.sin(np.deg2rad(t1))
                             - (j - width // 2) * np.cos(np.deg2rad(t1)))
                        r, c = wy, wx
                        r, c = int(r * 100 / args.map_resolution), \
                            int(c * 100 / args.map_resolution)
                        [r, c] = pu.threshold_poses([r, c],
                                                    self.collision_map.shape)
                        self.collision_map[r, c] = 1

                width = 5
                length = 4
                buf = 3
                for i in range(length):
                    for j in range(width):
                        wx = x1 + 0.05 * \
                            ((i + buf) * np.cos(np.deg2rad(t1))
                             + (j - width // 2) * np.sin(np.deg2rad(t1)))
                        wy = y1 + 0.05 * \
                            ((i + buf) * np.sin(np.deg2rad(t1))
                             - (j - width // 2) * np.cos(np.deg2rad(t1)))
                        r, c = wy, wx
                        r, c = int(r * 100 / args.map_resolution), \
                            int(c * 100 / args.map_resolution)
                        [r, c] = pu.threshold_poses([r, c],
                                                    self.collision_map.shape)
                        self.collision_map_big[r, c] = 1
            else:
                if self.prev_blocked >= self.block_threshold:
                    self.untrap.reset()
                self.prev_blocked = 0

        stg, stop = self._get_stg(map_pred, start, np.copy(goal),
                                  planning_window)

        # Deterministic Local Policy
        if self.forward_after_stop < 0:
            self.forward_after_stop = self.forward_after_stop_preset
        if self.forward_after_stop != self.forward_after_stop_preset:
            if self.forward_after_stop == 0:
                self.forward_after_stop -= 1
                action = 0
            else:
                self.forward_after_stop -= 1
                action = 1
        elif stop and planner_inputs['found_goal'] == 1:
            if self.forward_after_stop == 0:
                action = 0  # Stop
            else:
                self.forward_after_stop -= 1
                action = 1
            #action = 0
        else:
            (stg_x, stg_y) = stg
            angle_st_goal = math.degrees(math.atan2(stg_x - start[0],
                                                    stg_y - start[1]))
            angle_agent = (start_o) % 360.0
            if angle_agent > 180:
                angle_agent -= 360

            relative_angle = (angle_agent - angle_st_goal) % 360.0
            if relative_angle > 180:
                relative_angle -= 360

            if relative_angle > self.args.turn_angle / 2.:
                action = 3  # Right
            elif relative_angle < -self.args.turn_angle / 2.:
                action = 2  # Left
            else:
                action = 1  # Forward
        if self.prev_blocked >= self.block_threshold:
            if self._previous_action == 1:
                action = self.untrap.get_action()
            else:
                action = 1
        self._previous_action = action
        return action

    def _get_stg(self, grid, start, goal, planning_window):
        """Get short-term goal"""

        [gx1, gx2, gy1, gy2] = planning_window

        x1, y1, = 0, 0
        x2, y2 = grid.shape

        def add_boundary(mat, value=1):
            h, w = mat.shape
            new_mat = np.zeros((h + 2, w + 2)) + value
            new_mat[1:h + 1, 1:w + 1] = mat
            return new_mat

        def surrounded_by_obstacle(mat,i,j):
            i1 = max(0,i-3)
            i2 = min(mat.shape[0],i+2)
            j1 = max(0,j-3)
            j2 = min(mat.shape[1],j+2)
            return np.sum(mat[i1:i2,j1:j2]) > 0


        traversible = skimage.morphology.binary_dilation(
            grid[x1:x2, y1:y2],
            self.selem) != True

        if self.use_small_num > 0:
            self.use_small_num -= 1
            traversible[self.collision_map[gx1:gx2, gy1:gy2]
                        [x1:x2, y1:y2] == 1] = 0
            if surrounded_by_obstacle(self.collision_map[gx1:gx2, gy1:gy2], start[0], start[1]) or \
                surrounded_by_obstacle(grid,start[0],start[1]):
                traversible[
                    self.visited_vis[gx1:gx2, gy1:gy2][x1:x2,
                    y1:y2] == 1] = 1
        else:
            traversible[self.collision_map_big[gx1:gx2, gy1:gy2]
                        [x1:x2, y1:y2] == 1] = 0
            if surrounded_by_obstacle(self.collision_map_big[gx1:gx2, gy1:gy2], start[0], start[1]) or \
                surrounded_by_obstacle(grid,start[0],start[1]):
                traversible[
                    self.visited_vis[gx1:gx2, gy1:gy2][x1:x2,
                    y1:y2] == 1] = 1


        traversible[int(start[0] - x1) - 1:int(start[0] - x1) + 2,
                    int(start[1] - y1) - 1:int(start[1] - y1) + 2] = 1

        traversible = add_boundary(traversible)
        goal = add_boundary(goal, value=0)

        planner = FMMPlanner(traversible)

        selem = skimage.morphology.disk(10)
        goal = skimage.morphology.binary_dilation(
            goal, selem) != True

        goal = 1 - goal * 1.
        planner.set_multi_goal(goal)

        state = [start[0] - x1 + 1, start[1] - y1 + 1]
        # assume replan true suggests failure in planning
        stg_x, stg_y, distance, stop = planner.get_short_term_goal(state)

        # Failed to plan a path
        if self.found_goal == 0 and distance > self.args.change_goal_threshold:
            self.use_small_num = 20
            # tell the global planner to change goal
            self.agent_states.set_hard_goal()


        #If we are already using the optimistic collision map, but still fail to plan a path to the goal, make goal larger
        if self.args.small_collision_map_for_goal == 0 or (self.args.small_collision_map_for_goal == 1 and self.use_small_num > 0):
            if self.found_goal == 1 and distance > self.args.magnify_goal_when_hard:
                radius = 2
                step = 0
                while distance > 100:
                    step += 1
                    if step > 10:
                        break
                    selem = skimage.morphology.disk(radius)
                    goal = skimage.morphology.binary_dilation(
                        goal, selem) != True
                    goal = 1 - goal * 1.
                    planner.set_multi_goal(goal)

                    # assume replan true suggests failure in planning
                    stg_x, stg_y, distance, stop = planner.get_short_term_goal(
                        state)
        # If we fail to plan a path to the goal, make sure to use optimistic collision map
        if self.found_goal == 1 and distance > self.args.change_goal_threshold:
            if self.args.small_collision_map_for_goal == 1:
                self.use_small_num = 20


        stg_x, stg_y = stg_x + x1 - 1, stg_y + y1 - 1
        self.stg = (stg_x, stg_y)
        return (stg_x, stg_y), stop

    def _preprocess_obs(self, rgb, depth, use_seg=True):
        args = self.args
        sem_seg_pred = self._get_sem_pred(
            rgb.astype(np.uint8), use_seg=use_seg, depth=depth)



        depth = self._preprocess_depth(depth, args.min_depth, args.max_depth)

        ds = args.env_frame_width // args.frame_width  # Downscaling factor
        if ds != 1:
            rgb = np.asarray(self.res(rgb.astype(np.uint8)))
            depth = depth[ds // 2::ds, ds // 2::ds]
            sem_seg_pred = sem_seg_pred[ds // 2::ds, ds // 2::ds]

        depth = np.expand_dims(depth, axis=2)
        state = np.concatenate((rgb, depth, sem_seg_pred),
                               axis=2).transpose(2, 0, 1)

        return state

    def _preprocess_depth(self, depth, min_d, max_d):
        depth = depth[:, :, 0] * 1

        for i in range(depth.shape[1]):
            depth[:, i][depth[:, i] == 0.] = depth[:, i].max()

        mask2 = depth > 0.99
        depth[mask2] = 0.

        mask1 = depth == 0
        depth[mask1] = 100.0
        depth = min_d * 100.0 + depth * max_d * 100.0
        return depth

    def call_sem(self,rgb,depth):
        self.sem_pred_rednet.get_prediction(rgb, depth)

    def _get_sem_pred(self, rgb, use_seg=True,depth = None):
        if self.args.print_images == 1:
            self.rgb_vis = rgb

        semantic_pred_rednet = self.sem_pred_rednet.get_prediction(rgb,depth,self.goal_cat)
        return semantic_pred_rednet.astype(np.float32)

    def save_semantic(self, img,fn):
        plt.figure(figsize=(12, 8))
        plt.imshow(img)
        plt.savefig(fn)
        plt.close()

    def _visualize(self, inputs):
        args = self.args
        dump_dir = "{}/dump/{}/".format(args.dump_location,
                                        args.exp_name)
        ep_dir = '{}/episodes/thread_{}/eps_{}/'.format(
            dump_dir, self.rank, self.episode_no)
        if not os.path.exists(ep_dir):
            os.makedirs(ep_dir)
        ep_dir = '{}/episodes/thread_{}/eps_{}/'.format(
            dump_dir, self.rank+1, self.episode_no)
        if not os.path.exists(ep_dir):
            os.makedirs(ep_dir)
        map_pred = inputs['map_pred']
        exp_pred = inputs['exp_pred']
        start_x, start_y, start_o, gx1, gx2, gy1, gy2 = inputs['pose_pred']
        goal = inputs['goal']


        sem_map = inputs['sem_map_pred']
        #exit(0)

        if 'itself' in inputs.keys():
            my_score = inputs['itself']
            opp_cat = inputs['opp_cat']
            opp_score = inputs['opp_score']
            str = '{} {} {} {}'.format(self.goal_name, my_score, opp_cat,
                                       opp_score)
            self.vis_image = vu.init_vis_image(str, self.legend)


        gx1, gx2, gy1, gy2 = int(gx1), int(gx2), int(gy1), int(gy2)

        sem_map += 5
        sem_map[self.collision_map[gx1:gx2, gy1:gy2] == 1] = 14
        sem_map[int(self.stg[0]),int(self.stg[1])] = 15
        #print(sem_map.shape,self.collision_map[gx1:gx2, gy1:gy2].shape)
        #exit(0)
        no_cat_mask = sem_map == 2+4+1+2 + self.args.use_gt_mask
        map_mask = np.rint(map_pred) == 1
        exp_mask = np.rint(exp_pred) == 1
        vis_mask = self.visited_vis[gx1:gx2, gy1:gy2] == 1#1 TODO: change back
        #vis_mask[self.last_start[0],self.last_start[1]] = True # TODO: delete

        sem_map[no_cat_mask] = 0
        m1 = np.logical_and(no_cat_mask, exp_mask)
        sem_map[m1] = 2

        m2 = np.logical_and(no_cat_mask, map_mask)
        sem_map[m2] = 1

        sem_map[vis_mask] = 3

        selem = skimage.morphology.disk(4)
        goal_mat = 1 - skimage.morphology.binary_dilation(
            goal, selem) != True

        goal_mask = goal_mat == 1
        sem_map[goal_mask] = 4

        color_pal = [int(x * 255.) for x in color_palette]
        sem_map_vis = Image.new("P", (sem_map.shape[1],
                                      sem_map.shape[0]))
        sem_map_vis.putpalette(color_pal)
        sem_map_vis.putdata(sem_map.flatten().astype(np.uint8))
        sem_map_vis = sem_map_vis.convert("RGB")
        sem_map_vis = np.flipud(sem_map_vis)

        sem_map_vis = sem_map_vis[:, :, [2, 1, 0]]
        sem_map_vis = cv2.resize(sem_map_vis, (480, 480),
                                 interpolation=cv2.INTER_NEAREST)
        self.vis_image[50:530, 15:655] = self.rgb_vis
        self.vis_image[50:530, 670:1150] = sem_map_vis

        pos = (
            (start_x * 100. / args.map_resolution - gy1)
            * 480 / map_pred.shape[0],
            (map_pred.shape[1] - start_y * 100. / args.map_resolution + gx1)
            * 480 / map_pred.shape[1],
            np.deg2rad(-start_o)
        )

        agent_arrow = vu.get_contour_points(pos, origin=(670, 50))
        color = (int(color_palette[11] * 255),
                 int(color_palette[10] * 255),
                 int(color_palette[9] * 255))
        #cv2.drawContours(self.vis_image, [agent_arrow], 0, color, -1) # TODO: change back

        if args.visualize:
            # Displaying the image
            cv2.imshow("Thread {}".format(self.rank), self.vis_image)
            cv2.waitKey(1)

        if args.print_images:
            fn = '{}/episodes/thread_{}/eps_{}/{}-{}-Vis-{}.png'.format(
                dump_dir, self.rank, self.episode_no,
                self.rank, self.episode_no, self.timestep)

            cv2.imwrite(fn, self.vis_image)
            fn2 = '{}/episodes/thread_{}/eps_{}/{}-{}-Vis-{}.png'.format(
                dump_dir, self.rank+1, self.episode_no,
                self.rank, self.episode_no, self.timestep)
            #if self.mask is not None:
            #    self.save_semantic(self.mask.cpu().numpy(),fn2)




