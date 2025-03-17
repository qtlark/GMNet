import random
import numpy as np
import torch
import torch.utils.data as data
import data.util as util
import cv2

import os.path as osp

class LQGT_dataset(data.Dataset):
    '''
    Read SDR image (img_LQ), Normalized Gain Map (img_GT), 
    SDR thumbnail (img_MN), Maximum value Qmax (qmax).
    Gain Map (img_QGM) is generated on-the-fly, 
    and Original SDR image (img_org) is just for test.
    The SDR-GM pair is ensured by 'sorted' function, so please check the name convention.
    '''

    def __init__(self, opt):
        super(LQGT_dataset, self).__init__()
        self.opt = opt
        self.data_type = self.opt['data_type']
        self.paths_LQ, self.paths_GT = None, None
        self.sizes_LQ, self.sizes_GT = None, None
        self.LQ_env, self.GT_env = None, None  # environment for lmdb

        self.sizes_GT, self.paths_GT = util.get_image_paths(self.data_type, opt['dataroot_GT'])
        self.sizes_LQ, self.paths_LQ = util.get_image_paths(self.data_type, opt['dataroot_LQ'])
        assert self.paths_GT, 'Error: GT path is empty.'

        if self.paths_LQ and self.paths_GT:
            assert len(self.paths_LQ) == len(
                self.paths_GT
            ), 'GT and LQ datasets have different number of images - {}, {}.'.format(
                len(self.paths_LQ), len(self.paths_GT))

    def __getitem__(self, index):
        peak = self.opt['peak']
        scale = self.opt['scale']
        GT_size = self.opt['GT_size']
        GT_path, LQ_path = None, None
        
        # get input SDR image
        LQ_path = self.paths_LQ[index]
        img_LQ = util.read_imgdata(LQ_path, ratio=255.0).astype("float32")
        img_org = img_LQ.copy()     # origin image for test
        img_LQ = cv2.resize(img_LQ, None, None, fx=1/scale, fy=1/scale, interpolation=cv2.INTER_CUBIC)
        img_LQ = img_LQ.clip(0,1)

        # get Normalized Gain Map (NGM)
        GT_path = self.paths_GT[index]
        img_GT = util.read_imgdata(GT_path, ratio=255.0).astype("float32")
        img_GT = img_GT[..., np.newaxis]   # (h,w) -> (h,w,1)

        # get metadata and thumbnail
        fname = osp.basename(LQ_path)[:7]   # patch name to full name: IMG_365_s024.png -> IMG_365
        qmax = np.load(osp.join(self.opt['dataroot_QM'], '%s.npy'%fname)).astype(np.float32)
        img_MN = util.read_imgdata(osp.join(self.opt['dataroot_MN'], '%s.png'%fname), ratio=255.0).astype("float32")

        # crop patches to smaller patches
        if self.opt['phase'] == 'train':
            H, W, C = img_LQ.shape
            H_gt, W_gt, C_gt = img_GT.shape

            if H != H_gt:
                print('*******wrong image*******:{}'.format(LQ_path))
            LQ_size = GT_size

            # randomly crop
            if GT_size is not None:
                rnd_h = random.randint(0, max(0, H - LQ_size))
                rnd_w = random.randint(0, max(0, W - LQ_size))
                img_LQ = img_LQ[rnd_h:rnd_h + LQ_size, rnd_w:rnd_w + LQ_size, :]
                rnd_h_GT, rnd_w_GT = rnd_h, rnd_w
                img_GT = img_GT[rnd_h_GT:rnd_h_GT + GT_size, rnd_w_GT:rnd_w_GT + GT_size, :]

            # augmentation - flip, rotate
            img_LQ, img_GT = util.augment([img_LQ, img_GT], self.opt['use_flip'], self.opt['use_rot'])


        # get Gain Map (GM)
        # and normalize GM by dataset maximum
        img_QGM = img_GT * qmax              
        img_QGM = img_QGM / np.log2(peak)   

        # BRG to RGB
        # img_GT and img_QGM is gray, and img_org is handled below
        img_LQ = img_LQ[:, :, [2, 1, 0]]
        img_MN = img_MN[:, :, [2, 1, 0]]
        
        # Numpy to Torch
        img_LQ  = torch.from_numpy(np.ascontiguousarray(np.transpose(img_LQ,  (2, 0, 1)))).float()
        img_MN  = torch.from_numpy(np.ascontiguousarray(np.transpose(img_MN,  (2, 0, 1)))).float()
        img_GT  = torch.from_numpy(np.ascontiguousarray(np.transpose(img_GT,  (2, 0, 1)))).float()
        img_QGM = torch.from_numpy(np.ascontiguousarray(np.transpose(img_QGM, (2, 0, 1)))).float()

        # process original SDR if needed
        if self.opt['phase'] != 'train':
            img_org = img_org[:, :, [2, 1, 0]]
            img_org = np.power(img_org, 2.2)
            img_org = torch.from_numpy(np.ascontiguousarray(np.transpose(img_org, (2, 0, 1)))).float()
        else:
            del img_org         # del origin image in train phase to free GPU memory
            img_org = LQ_path   # placeholder for consistency

        return {'LQ': img_LQ, 'MN':img_MN, 'GT': img_GT, 'GT2': img_QGM, 'LQ_path': LQ_path, 'GT_path': GT_path, 'org': img_org}

    def __len__(self):
        return len(self.paths_GT)
