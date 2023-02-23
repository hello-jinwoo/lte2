# modified from: https://github.com/yinboc/liif

import argparse
import os
import numpy as np
import glob
import imageio.v2 as imageio

import yaml
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
import torch.nn.functional as F
import torchvision

import datasets
import models
import utils

import random
from bicubic_pytorch import core
import cv2

# from utils import to_pixel_samples


def to_pixel_samples(img):
    """ Convert the image to coord-RGB pairs.
        img: Tensor, (3, H, W)
    """

    b, _,_,_ = img.size()
    coord = utils.make_coord(img.shape[-2:])
    rgb = img.view(b, 3, -1).permute(0,2, 1)
    return coord, rgb

def make_data_loader(spec, tag=''):
    if spec is None:
        return None

    dataset = datasets.make(spec['dataset'])
    dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})

    log('{} dataset: size={}'.format(tag, len(dataset)))
    for k, v in dataset[0].items():
        log('  {}: shape={}'.format(k, tuple(v.shape)))

    loader = DataLoader(dataset, batch_size=spec['batch_size'],
        shuffle=(tag == 'train'), num_workers=8, pin_memory=True)
    return loader


def make_data_loaders():
    train_loader = make_data_loader(config.get('train_dataset'), tag='train')
    return train_loader


def prepare_training():
#     if config.get('resume') is not None:
    if config['is_resume'] and os.path.exists(config.get('resume')):
        sv_file = torch.load(config['resume'])
        model = models.make(sv_file['model'], load_sd=True).cuda()
        optimizer = utils.make_optimizer(
            model.parameters(), sv_file['optimizer'], load_sd=True)
        epoch_start = sv_file['epoch'] + 1
        if config.get('multi_step_lr') is None:
            lr_scheduler = None
        else:
            lr_scheduler = MultiStepLR(optimizer, **config['multi_step_lr'])
        for _ in range(epoch_start - 1):
            lr_scheduler.step()
    else:
        model = models.make(config['model']).cuda()
        optimizer = utils.make_optimizer(
            model.parameters(), config['optimizer'])
        epoch_start = 1
        if config.get('multi_step_lr') is None:
            lr_scheduler = None
        else:
            lr_scheduler = MultiStepLR(optimizer, **config['multi_step_lr'])

    log('model: #params={}'.format(utils.compute_num_params(model, text=True)))
    return model, optimizer, epoch_start, lr_scheduler



def eval(model, data_name, save_dir, scale_factor=4, config=None):
    model.eval()
    test_path = './load/' + data_name + '/HR'
    gt_images = sorted(glob.glob(test_path + '/*.png'))

    save_path = os.path.join(save_dir,  data_name)
    os.makedirs(save_path, exist_ok=True)

    total_psnrs = []

    for gt_path in gt_images:
        # print(gt_path)
        filename = os.path.basename(gt_path).split('.')[0] 
        gt = imageio.imread(gt_path)
        
        if gt.ndim == 2:
            gt = np.expand_dims(gt, axis=2)
            gt = np.repeat(gt, 3, axis=2)
        h, w, c = gt.shape
        # new_h, new_w = h - h % self.args.size_must_mode, w - w % self.args.size_must_mode
        # gt = gt[:new_h, :new_w, :]
        gt_tensor = utils.numpy2tensor(gt).cuda()
        gt_tensor, pad = utils.pad_img(gt_tensor, int(24*scale_factor))#self.args.size_must_mode*self.args.scale)
        _,_, new_h, new_w = gt_tensor.size()
        input_tensor = core.imresize(gt_tensor, scale=1/scale_factor)
        blurred_tensor1 = F.interpolate(input_tensor, scale_factor=scale_factor, mode='nearest')
        blurred_tensor2 = core.imresize(input_tensor, scale=scale_factor)
        # if type(config['model']['args']['upsample_mode']) == str:
        #     input_tensor = F.interpolate(gt_tensor, 
        #         scale_factor=1/scale_factor, 
        #         mode=config['model']['args']['upsample_mode'])
        #     blurred_tensor = F.interpolate(input_tensor, 
        #         scale_factor=scale_factor, 
        #         mode=config['model']['args']['upsample_mode'])
        # else:
        #     input_tensor = F.interpolate(gt_tensor, 
        #         scale_factor=1/scale_factor, 
        #         mode='bicubic')
        #     blurred_tensor = F.interpolate(input_tensor, 
        #         scale_factor=scale_factor, 
        #         mode='bicubic')

        with torch.no_grad():
            output = model(x=(input_tensor - 0.5) / 0.5, 
                           scale_factor=None, 
                           size=(new_h, new_w),
                           mode='test')
            output = output * 0.5 + 0.5

        if 'output_sharpening' in config and config['output_sharpening']:
            output = torchvision.transforms.functional.adjust_sharpness(output, 1.0)

        output_img = utils.tensor2numpy(output[0:1,:, pad[2]:new_h-pad[3], pad[0]:new_w-pad[1]])            
        if 'output_bilateral' in config and config['output_bilateral']:
            output_img = cv2.bilateralFilter(output_img, -1, 10, 5)
        input_img1 = utils.tensor2numpy(blurred_tensor1[0:1,:, pad[2]:new_h-pad[3], pad[0]:new_w-pad[1]])
        input_img2 = utils.tensor2numpy(blurred_tensor2[0:1,:, pad[2]:new_h-pad[3], pad[0]:new_w-pad[1]])            
        gt_img = utils.tensor2numpy(gt_tensor[0:1,:, pad[2]:new_h-pad[3], pad[0]:new_w-pad[1]])            
        output_error = gt_img - output_img
        input1_error = gt_img - input_img1
        input2_error = gt_img - input_img2
        gt_error = gt_img - gt_img # zero map
        psnr = utils.psnr_measure(output_img, gt_img)

        img_files = glob.glob(f"{save_path}/{filename}_{scale_factor}*")
        for f in img_files:
            os.remove(f)

        canvas = np.concatenate((input_img1, input_img2, output_img, gt_img), 1)
        canvas_error = np.concatenate((input1_error, input2_error, output_error, gt_error), 1)
        canvas = np.concatenate((canvas, canvas_error), 0)
        utils.save_img_np(canvas, '{}/{}_{}_{:.2f}.png'.format(save_path, filename, scale_factor, psnr))

        total_psnrs.append(psnr)

        
    total_psnrs = np.mean(np.array(total_psnrs))
    

    return  total_psnrs


def main(config_, save_path):
    global config, log, writer
    config = config_
    
    log, writer = utils.set_save_path(save_path, remove=False)
    log_info = []
    
    model, optimizer, epoch_start, lr_scheduler = prepare_training()

    n_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    if n_gpus > 1:
        model = nn.parallel.DataParallel(model)

    if n_gpus > 1 and (config.get('eval_bsize') is not None):
        model_ = model.module
    else:
        model_ = model

    if config['scale']['mode'] == 'fixed':
        if config['scale']['factor'] == 1:
            # scale_factors = [1, 1.25, 1.5, 1.75, 2, 3, 4]
            scale_factors = [1.]
        else:
            scale_factors = [2, 3, 4]
    else:
        scale_factors = [2, 3, 4, 6, 8, 12]
    model.eval()

    for sf in scale_factors:
        val_res_set14 = eval(model, 'Set14', save_path, scale_factor=sf, config=config)
        val_res_set5 = eval(model, 'Set5', save_path, scale_factor=sf, config=config)
        val_res_div = eval(model, 'DIV2K_train_Set10', save_path, scale_factor=sf, config=config)
        if sf == 4:
            val_sf4 = val_res_set14
        log_info.append('SF{}:{:.4f}/{:.4f}/{:.4f}'.format(sf,val_res_set5, val_res_set14, val_res_div))
    
    log(', '.join(log_info))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--name', default=None)
    parser.add_argument('--tag', default=None)
    parser.add_argument('--gpu', default='0')

    # parser.add_argument('--version', default='v1', type=str)

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    version = args.config.split('.')[-2].split('/')[-1]
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        print('config loaded.', version)

    save_path = os.path.join('./save', '{}'.format(version))
    os.makedirs(save_path, exist_ok=True)
    # if save_name is None:
    #     save_name = '_' + args.config.split('/')[-1][:-len('.yaml')]
    # if args.tag is not None:
    #     save_name += '_' + args.tag
    # save_path = os.path.join('./save', save_name)
    
    main(config, save_path)