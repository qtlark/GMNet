import os.path as osp
import logging
import time
import argparse
from collections import OrderedDict

import options.options as option
import utils.util as util
from data import create_dataset, create_dataloader
from models import create_model

import os
import cv2
import numpy as np


#### options
parser = argparse.ArgumentParser()
parser.add_argument('-opt', type=str, required=True, help='Path to options YMAL file.')
opt_path = parser.parse_args().opt
opt = option.parse(opt_path, is_train=False)
opt = option.dict_to_nonedict(opt)


util.mkdir_and_rename(opt['path']['results_root']) 
util.mkdirs((path for key, path in opt['path'].items() if not key == 'experiments_root'
                and 'pretrain_model' not in key and 'resume' not in key))
util.setup_logger('base', opt['path']['log'], 'test_' + opt['name'], level=logging.INFO,
                  screen=True, tofile=True)
logger = logging.getLogger('base')
logger.info(option.dict2str(opt))

#### Create test dataset and dataloader
test_loaders = []
for phase, dataset_opt in sorted(opt['datasets'].items()):
    test_set = create_dataset(dataset_opt)
    test_loader = create_dataloader(test_set, dataset_opt)
    logger.info('Number of test images in [{:s}]: {:d}'.format(dataset_opt['name'], len(test_set)))
    test_loaders.append(test_loader)

model = create_model(opt)
for test_loader in test_loaders:
    test_set_name = test_loader.dataset.opt['name']
    logger.info('\nTesting [{:s}]...'.format(test_set_name))
    test_start_time = time.time()
    dataset_dir = osp.join(opt['path']['results_root'], test_set_name)
    util.mkdir(dataset_dir)

    peak = opt['peak']
    scale = opt['scale']
    
    avg_psnr = 0.0
    avg_norm_psnr = 0.0
    idx = 0
    for test_data in test_loader:
        idx += 1
        
        model.feed_data(test_data)
        model.test()

        visuals = model.get_current_visuals()

        lq_img = util.tensor2numpy(test_data['org'][0])
        sr_img = util.tensor2numpy(visuals['QGM'])  # float32
        gt_img = util.tensor2numpy(visuals['GT2'])  # float32

        sr_img = cv2.resize(sr_img, None, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)[..., np.newaxis]
        gt_img = cv2.resize(gt_img, None, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)[..., np.newaxis]

        sr_img = lq_img*np.power(2, np.clip(sr_img,0,1)*np.log2(peak)) / peak
        gt_img = lq_img*np.power(2, np.clip(gt_img,0,1)*np.log2(peak)) / peak

        if opt['datasets']['test']['save_img']:
            img_name = os.path.splitext(os.path.basename(test_data['LQ_path'][0]))[0]
            cv2.imwrite(os.path.join(dataset_dir, '{:s}_pd.tif'.format(img_name)), sr_img, [cv2.IMWRITE_TIFF_COMPRESSION, 1])
            cv2.imwrite(os.path.join(dataset_dir, '{:s}_gt.tif'.format(img_name)), gt_img, [cv2.IMWRITE_TIFF_COMPRESSION, 1])

        psnr = util.calculate_normalized_psnr(sr_img, gt_img, np.array(1.0))
        norm_psnr = util.calculate_normalized_psnr(sr_img, gt_img, np.max(gt_img))

        logger.info('# Index: {:03d} # PSNR: {:.4f} # norm_psnr: {:.4f}'.format(idx, psnr, norm_psnr))

        # calculate PSNR
        avg_psnr += psnr
        avg_norm_psnr += norm_psnr

    avg_psnr = avg_psnr / idx
    avg_norm_psnr = avg_norm_psnr / idx

    # log
    logger.info('# Validation # PSNR: {:.4f}  # norm_PSNR: {:.4f}'.format(avg_psnr, avg_norm_psnr))