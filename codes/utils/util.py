import os
import math
from datetime import datetime
import random
import logging
from collections import OrderedDict
import numpy as np
import cv2
import torch
import shutil

import yaml
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper


def OrderedYaml():
    '''yaml orderedDict support'''
    _mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG

    def dict_representer(dumper, data):
        return dumper.represent_dict(data.items())

    def dict_constructor(loader, node):
        return OrderedDict(loader.construct_pairs(node))

    Dumper.add_representer(OrderedDict, dict_representer)
    Loader.add_constructor(_mapping_tag, dict_constructor)
    return Loader, Dumper


####################
# miscellaneous
####################


def get_timestamp():
    return datetime.now().strftime('%y%m%d-%H%M%S')


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def mkdirs(paths):
    if isinstance(paths, str):
        mkdir(paths)
    else:
        for path in paths:
            mkdir(path)


def mkdir_and_rename(path):
    if os.path.exists(path):
        logger = logging.getLogger('base')
        logger.info('Path already exists. Renove it.')
        shutil.rmtree(path)
        # new_name = path + '_archived_' + get_timestamp()
        # print('Path already exists. Rename it to [{:s}]'.format(new_name))
        # logger = logging.getLogger('base')
        # logger.info('Path already exists. Rename it to [{:s}]'.format(new_name))
        # os.rename(path, new_name)
    os.makedirs(path)


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_logger(logger_name, root, phase, level=logging.INFO, screen=False, tofile=False):
    '''set up logger'''
    lg = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s',
                                  datefmt='%y-%m-%d %H:%M:%S')
    lg.setLevel(level)
    if tofile:
        log_file = os.path.join(root, phase + '_{}.log'.format(get_timestamp()))
        fh = logging.FileHandler(log_file, mode='w')
        fh.setFormatter(formatter)
        lg.addHandler(fh)
    if screen:
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        lg.addHandler(sh)


def tensor2img(tensor, out_type=np.uint8, min_max=(0, 1)):
    '''Converts a torch Tensor into an image Numpy array'''
    if torch.isnan(tensor).any(): raise ValueError("NaN Raised")
    tensor = tensor.squeeze().float().cpu().clamp_(*min_max)  # clamp
    tensor = (tensor - min_max[0]) / (min_max[1] - min_max[0])  # to range [0,1]
    img_np = tensor.numpy()

    if len(img_np.shape) == 3:
        img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))  # HWC, BGR
    else:
        img_np = np.transpose(img_np[np.newaxis, :, :], (1, 2, 0))

    if out_type == np.uint8:
        img_np = (img_np * 255.0).round()
    elif out_type == np.uint16:
        img_np = (img_np * 65535.0).round()
        # Important. Unlike matlab, numpy.unit8() WILL NOT round by default.
    return img_np.astype(out_type)


def save_img(img, img_path, mode='RGB'):
    cv2.imwrite(img_path, img)


def save_npy(img, img_path):
    img = np.squeeze(img)
    np.save(img_path, img)


def calculate_psnr(img1, img2):
    img1 = img1.astype(np.float32) # np.float64
    img2 = img2.astype(np.float32) # np.float64
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    # return 20 * math.log10(255.0 / math.sqrt(mse))
    return 20 * math.log10(1.0 / math.sqrt(mse))

def calculate_normalized_psnr(img1, img2, norm):
    img1 = img1.astype("float64")
    img2 = img2.astype("float64")
    norm = norm.astype("float64")
    normalized_psnr = -10*np.log10(np.mean(np.power(img1/norm - img2/norm, 2)))
    if normalized_psnr == 0:
        return float('inf')
    return normalized_psnr


def pq_oetf(x):
    m1 = 0.1593017578125
    m2 = 78.84375
    c1 = 0.8359375
    c2 = 18.8515625
    c3 = 18.6875

    Ym = np.power(x/10000, m1)
    return np.power((c1 + c2 * Ym) / (1.0 + c3 * Ym), m2)

def pq_eotf(x):
    m1 = 0.1593017578125
    m2 = 78.84375
    c1 = 0.8359375
    c2 = 18.8515625
    c3 = 18.6875

    x = x.clip(0,None)
    p = np.power(x, 1/m2)
    num = (p-c1).clip(0,None)
    den = c2 - c3*p
    
    return 10000.0 * np.power(num/den, 1/m1)


def tensor2numpy(tensor):
    img_np = tensor.numpy()
    img_np[img_np < 0] = 0
    if img_np.shape[0]==3:
        img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))  # HWC, BGR
    else:
        img_np = np.transpose(img_np, (1, 2, 0))
    return img_np.astype(np.float32)