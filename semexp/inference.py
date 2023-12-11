import sys
import os
# import requests

import torch
import numpy as np

import matplotlib.pyplot as plt
from PIL import Image
import SGM.semexp.models_sgm_cross as models_sgm_cross
import cv2
import bz2
import _pickle as cPickle

# from mask_transform import RandomMaskingGenerator
from utils.visualize_tools import visualize_semmap 
# define the utils
import torchvision.transforms.functional as F
from torchvision.transforms import Resize 
import torch.nn.functional as nnf
import skimage.morphology

import json
import logging
import math
import os
import re
import time
from collections import deque
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import gym
import numpy as np
import SGM.semexp.models_sgm_cross as models_sgm

import sys
sys.path.append("..")

import semexp.envs.utils.pose as pu
import torch
import torch.nn as nn
from semexp.arguments import get_args
from semexp.envs import make_vec_envs

from semexp.model import Semantic_Mapping
from semexp.model_pf import RL_Policy
from semexp.utils.storage import GlobalRolloutStorage
from torch.utils.tensorboard import SummaryWriter

from semexp.sgm_adds import *

import torchvision.transforms.functional as F
from torchvision.transforms import Resize 
import torch.nn.functional as nnf
import skimage.morphology

def prepare_model(chkpt_dir, model):
    # load model
    checkpoint = torch.load(chkpt_dir, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    print(msg)
    return model

def get_mask(data):
    data = data.squeeze(dim=0).sum(dim=1)
    mask = torch.where(data>50, 0, 1)
    vis_patch_num = data.shape[0] - mask.sum()
    return mask.unsqueeze(0).bool(), vis_patch_num

def model_inference(x, model, s=16, return_fig=False):
    h = x.size(1) // s
    
    # # mask generate
    # mask_generator = RandomMaskingGenerator(h, 0.75, True)
    # mask = torch.tensor(mask_generator()).to(x.device).to(torch.bool).unsqueeze(0)
    
    # make it a batch-like
    x = x.unsqueeze(dim=0)

    px = model.patchify(x)
    mask, vis_num = get_mask(px)
    loss, y, mask = model(x.float(), mask)
    
    B = torch.tensor(range(0, h*h)).to(x.device).unsqueeze(0)
    order = mask.float() * 1000 + B
    ids = torch.argsort(order, dim=1)
    ids_ = torch.argsort(ids, dim=1)
    y = torch.gather(y, dim=1, index=ids_.unsqueeze(-1).expand(-1, -1, y.size(-1)))
    # unnormalize y
    y = torch.where(y>0.7,1.0,0.0)
    y = model.unpatchify(y, c=x.shape[1])
    # channel filter
    # n,c,h,w = y.shape
    # t_y = y.view(c, h*w).transpose(0,1)
    # in_t_y = t_y.max(1)[1]
    y = y.detach().cpu()
    debug = y[0,:,208:,208:]
    # visualize the mask
    mask = mask.detach()
    mask = mask.unsqueeze(-1).repeat(1, 1, s**2 *x.shape[1])  # (N, H*W, p*p*3)
    mask = model.unpatchify(mask, c=x.shape[1])  # 1 is removing, 0 is keeping

    # masked image
    im_masked = x * (1 - mask.float())

    im_paste = x * (1 - mask.float()) + y * mask.float()

    visualize_semmap(y[0].numpy(),'..img_test/reconstruction.png')
    visualize_semmap(im_masked[0].numpy(),'..img_test/masked.png')
    visualize_semmap(im_paste[0].numpy(),'..img_test/reconstruction + visible_norm2.png')
    return im_paste[0]

def data_precompute(data):
    nz_c = torch.nonzero(torch.sum(data, dim=0))
    x_min,x_max,y_min,y_max = nz_c[:,0].min(), nz_c[:,0].max(), nz_c[:,1].min(), nz_c[:,1].max()
    wh = max(x_max-x_min,y_max-y_min)
    # resize = Resize([224,224])
    input_map = F.resize(F.crop(data, x_min,y_min,wh,wh),224)
    return input_map, (x_min,y_min,wh)

def map_zoom(org_map, pre_map, loc):
    org_map = org_map.detach()
    pre_map = pre_map.detach()
    x_min,y_min,wh = loc
    if wh>pre_map.shape[-1]:
        pre_map = nnf.interpolate(pre_map.unsqueeze(dim=0), size=[wh, wh])
        pre_map = pre_map.squeeze(dim=0)
    else:
        pre_map = F.resize(pre_map, wh)
    org_map[:,x_min:x_min+wh,y_min:y_min+wh] = pre_map
    org_map = org_map.numpy()
    # free place
    selem = skimage.morphology.disk(3)
    free_temp = cv2.erode(org_map[0], selem)
    org_map[0] = cv2.dilate(free_temp, selem)
    # obstacle
    selem = skimage.morphology.square(2)
    free_temp = cv2.erode(org_map[1], selem)
    org_map[1] = cv2.dilate(free_temp, selem)
    visualize_semmap(org_map,'..SGM/img_test/pre_map.png')
    return org_map

file_dir = '..data/semantic_maps/gibson/precomputed_dataset_24.0_123_spath_square/val/Darden_1/sample_00009.pbz2'
with bz2.BZ2File(file_dir, 'rb') as fp:
    data = cPickle.load(fp)
in_semmap = torch.from_numpy(data['in_semmap']).float() #17*480*480
reconstruction_gt = torch.from_numpy(data['semmap']).float()
visualize_semmap(in_semmap.numpy(),'..img_test/in_semmap.png')
visualize_semmap(reconstruction_gt.numpy(),'..img_test/semmap.png')

chkpt_dir = 'checkpoint-199.pth'
model = models_sgm_cross.sgm_vit_base_patch16_dec512d2b()
model_sgm = prepare_model(chkpt_dir, model)

# # make random mask reproducible (comment out to make it change)
torch.manual_seed(4)
np.random.seed(2)
print('SGM with pixel reconstruction:')
model_input, locs = data_precompute(in_semmap)
visualize_semmap(model_input.numpy(),'..img_test/input.png')
reconstruction_map = model_inference(model_input, model_sgm)
reconstruction_map = map_zoom(in_semmap, reconstruction_map, locs)
