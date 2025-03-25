import random
import math
from einops.einops import rearrange
import numpy as np
import torchvision.transforms as transforms

class RandomMaskingGenerator:
    def __init__(self, input_size, mask_ratio, regular=False):
        if not isinstance(input_size, tuple):
            input_size = (input_size,) * 2

        self.height, self.width = input_size

        self.num_patches = self.height * self.width
        self.num_mask = int(mask_ratio * self.num_patches)
        self.regular = regular

        if regular:
            assert mask_ratio == 0.75
        
            candidate_list = []
            while True: # add more
                for j in range(4):
                    candidate = np.ones(4)
                    candidate[j] = 0
                    candidate_list.append(candidate)
                if len(candidate_list) * 4 >= self.num_patches * 2:
                    break
            self.mask_candidate = np.vstack(candidate_list) 
            print('using regular, mask_candidate shape = ', 
                  self.mask_candidate.shape)

    def __repr__(self):
        repr_str = "Mask: total patches {}, mask patches {}, regular {}".format(
            self.num_patches, self.num_mask, self.regular
        )
        return repr_str

    def __call__(self):
        if not self.regular:
            mask = np.hstack([
                np.zeros(self.num_patches - self.num_mask),
                np.ones(self.num_mask),
            ])
            np.random.shuffle(mask)
        else:
            mask = self.mask_candidate.copy()
            np.random.shuffle(mask)
            mask = rearrange(mask[:self.num_patches//4], '(h w) (p1 p2) -> (h p1) (w p2)', 
                             h=self.height//2, w=self.width//2, p1=2, p2=2)
            mask = mask.flatten()

        return mask 


class RandomMaskingGenerator_new:
    def __init__(self, input_size, mask_ratio, regular, batch_size, vis_num):
        if not isinstance(input_size, tuple):
            input_size = (input_size,) * 2

        self.height, self.width = input_size

        self.num_patches = self.height * self.width
        # self.num_mask = int(mask_ratio * self.num_patches)
        self.num_mask = self.num_patches - vis_num
        self.regular = regular
        self.batch_size = batch_size

        if regular:
            assert mask_ratio == 0.75
        
            candidate_list = []
            while True: # add more
                for j in range(4):
                    candidate = np.ones(4)
                    candidate[j] = 0
                    candidate_list.append(candidate)
                if len(candidate_list) * 4 >= self.num_patches * 2:
                    break
            self.mask_candidate = np.vstack(candidate_list) 
            print('using regular, mask_candidate shape = ', 
                  self.mask_candidate.shape)

    def __repr__(self):
        repr_str = "Mask: total patches {}, mask patches {}, regular {}".format(
            self.num_patches, self.num_mask, self.regular
        )
        return repr_str

    def __call__(self):
        mask_all = np.zeros([self.batch_size, self.num_patches])
        if not self.regular:
            for i in range(self.batch_size):
                mask = np.hstack([
                    np.zeros(self.num_patches - self.num_mask),
                    np.ones(self.num_mask),
                ])
                np.random.shuffle(mask)
                mask_all[i] = mask
        else:
            mask = self.mask_candidate.copy()
            np.random.shuffle(mask)
            mask = rearrange(mask[:self.num_patches//4], '(h w) (p1 p2) -> (h p1) (w p2)', 
                             h=self.height//2, w=self.width//2, p1=2, p2=2)
            mask = mask.flatten()

        return mask_all


class MaskTransform(object):
    def __init__(self, args):
        self.transform = transforms.Compose([
				transforms.RandomResizedCrop(args.input_size, scale=(
					0.2, 1.0), interpolation=3),  # 3 is bicubic
				transforms.RandomHorizontalFlip(),
				transforms.ToTensor(),
				transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        if not hasattr(args, 'mask_regular'):
            args.mask_regular = False

        self.masked_position_generator = RandomMaskingGenerator(
            args.token_size, args.mask_ratio, args.mask_regular
        )
        self.masked_maplike_generator = MapMaskingGenerator(
            args.token_size, args.mask_ratio, args.mask_regular
        )

    def __call__(self, image):
        return self.transform(image), self.masked_position_generator()

    def __repr__(self):
        repr = "(MaskTransform,\n"
        repr += "  transform = %s,\n" % str(self.transform)
        repr += "  Masked position generator = %s,\n" % str(self.masked_position_generator)
        repr += ")"
        return repr

class MapMaskingGenerator:
    def __init__(self, input_size, mask_ratio, regular=False):
        if not isinstance(input_size, tuple):
            input_size = (input_size,) * 2

        self.height, self.width = input_size

        self.num_patches = self.height * self.width
        self.num_mask = int(mask_ratio * self.num_patches)

    def __repr__(self):
        repr_str = "Mask: total patches {}, mask patches {}, regular {}".format(
            self.num_patches, self.num_mask, self.regular
        )
        return repr_str

    def __call__(self):
        x_start = np.random.randint(self.width)
        y_start = np.random.randint(self.height)
        mask_candidate = []
        mask_candidate.append((x_start, y_start))
        potential_candidate = []
        potential_candidate = self.add_potential_candidate(potential_candidate, self.adjacent_coordinates(x_start,y_start),mask_candidate)
        for i in range(self.num_patches-self.num_mask-1):
            # idx = np.random.randint(len(potential_candidate))
            samp =random.sample(potential_candidate, 1)          
            x,y = samp[0]
            mask_candidate.append((x,y))
            potential_candidate.remove((x,y))
            potential_candidate = self.add_potential_candidate(potential_candidate, self.adjacent_coordinates(x,y),mask_candidate)
        mask = np.ones((self.width, self.height))
        mask_id = np.array(mask_candidate).swapaxes(0, 1)
        mask[mask_id[0],mask_id[1]] = 0
        mask = mask.flatten()
        nnn = mask.sum(0)
        return  mask
    
    def adjacent_coordinates(self, x, y):
        x_l = max(x-1, 0)
        x_r = min(x+1, self.width-1)
        y_u = min(y+1, self.height-1)
        y_l = max(y-1, 0)
        return_list = set([(x_l,y),(x_r,y),(x,y_u),(x,y_l)])
        if (x,y) in return_list:
            return_list.remove((x,y))
        return return_list
    
    def add_potential_candidate(self, p_list, new_list, m_list):
        for itm in new_list:
            if (itm not in p_list) and (itm not in m_list):
                p_list.append(itm)
        return p_list