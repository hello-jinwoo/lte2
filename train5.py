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

import datasets
import models
import utils

import random
from bicubic_pytorch import core


# from utils import to_pixel_samples

"""
train4 -> 5 : batched output (non-overlap)
"""


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


def batched_output(model, input, scale_factor, mode, option='non-overlap', input_size=48):
    B, _, h, w = input.size()
    input_size = min(input_size, h, w)
    
    pad_w = 0
    if w % input_size > 0:
        pad_w = input_size - w % input_size
    pad_h = 0
    if h % input_size > 0:
        pad_h = input_size - h % input_size
    pad_size = [0, pad_w, 0, pad_h]
    h_padded = h + pad_size[3]
    w_padded = w + pad_size[1]
    input = F.pad(input, pad_size, mode='reflect')

    H, W = int(h * scale_factor), int(w * scale_factor)
    H_padded, W_padded = int(h_padded * scale_factor), int(w_padded * scale_factor)
    output = torch.zeros(B, 3, H_padded, W_padded).to(input.device)

    if option == 'overlap':
        pass

    elif option == 'non-overlap':
        for i_h in range(0, h_padded, input_size):
            for i_w in range(0, w_padded, input_size):
                input_patch = input[:, :, i_h: i_h + input_size, i_w: i_w + input_size]
                output_patch = model(x=(input_patch - 0.5) / 0.5, 
                                     scale_factor=scale_factor, 
                                     mode='test')
                out_h_start = int(i_h  * scale_factor)
                out_h_end = int((i_h + input_size) * scale_factor)
                out_w_start = int(i_w * scale_factor)
                out_w_end = int((i_w + input_size) * scale_factor)
                output[:, :, out_h_start:out_h_end , out_w_start:out_w_end] = output_patch

    return output[:, :, :H, :W]


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


        with torch.no_grad():
            # output = model(x=(input_tensor - 0.5) / 0.5, 
            #                scale_factor=None, 
            #                size=(new_h, new_w),
            #                mode='test')
            output = batched_output(model=model,
                                    input=input_tensor, 
                                    scale_factor=scale_factor,
                                    mode='test')
            output = output * 0.5 + 0.5

        output_img = utils.tensor2numpy(output[0:1,:, pad[2]:new_h-pad[3], pad[0]:new_w-pad[1]])            
        input_img1 = utils.tensor2numpy(blurred_tensor1[0:1,:, pad[2]:new_h-pad[3], pad[0]:new_w-pad[1]])
        input_img2 = utils.tensor2numpy(blurred_tensor2[0:1,:, pad[2]:new_h-pad[3], pad[0]:new_w-pad[1]])            
        gt_img = utils.tensor2numpy(gt_tensor[0:1,:, pad[2]:new_h-pad[3], pad[0]:new_w-pad[1]])            
        psnr = utils.psnr_measure(output_img, gt_img)

        img_files = glob.glob(f"{save_path}/{filename}_{scale_factor}*")
        for f in img_files:
            os.remove(f)

        canvas = np.concatenate((input_img1, input_img2, output_img, gt_img), 1)
        utils.save_img_np(canvas, '{}/{}_{}_{:.2f}.png'.format(save_path, filename, scale_factor, psnr))

        total_psnrs.append(psnr)

        
    total_psnrs = np.mean(np.array(total_psnrs))
    

    return  total_psnrs


def train(train_loader, model, optimizer, epoch, config):
    model.train()
    loss_fn = nn.L1Loss()
    train_loss = utils.Averager()
    metric_fn = utils.calc_psnr

    data_norm = config['data_norm']
    t = data_norm['inp']
    inp_sub = torch.FloatTensor(t['sub']).view(1, -1, 1, 1).cuda()
    inp_div = torch.FloatTensor(t['div']).view(1, -1, 1, 1).cuda()
    t = data_norm['gt']
    gt_sub = torch.FloatTensor(t['sub']).view(1, 1, -1).cuda()
    gt_div = torch.FloatTensor(t['div']).view(1, 1, -1).cuda()
    
    num_dataset = 800 # DIV2K
    iter_per_epoch = int(num_dataset / config.get('train_dataset')['batch_size'] \
                        * config.get('train_dataset')['dataset']['args']['repeat'])
    iteration = 0
    
    descript = 'epoch : {}/{}'.format(epoch, config['epoch_max'])
    for batch in tqdm(train_loader, leave=False, desc=descript, mininterval=2):
        for k, v in batch.items():
            batch[k] = v.cuda()

        # normalize
        gt_img = (batch['gt_img'] - inp_sub) / inp_div
    

        # if config['mode'] == 0: 
        #     sf = random.uniform(1,4) # floating point
        # else:
        #     sf = random.randint(2,4) # integer 
        # sf = 4
        if config['scale']['mode'] == 'fixed':
            sf = config['scale']['factor']
        elif config['scale']['mode'] == 'multi_fixed':
            sf = random.choice(config['scale']['factor_list'])
        elif config['scale']['mode'] == 'multi_arbitrary':
            sf = random.uniform(config['scale']['factor_range'][0], config['scale']['factor_range'][1])
        else:
            sf = random.uniform(1, 4)

        # with torch.no_grad():
        inp_size = config.get('train_dataset')['wrapper']['args']['inp_size']
        inp = core.imresize(gt_img, sizes=(inp_size,inp_size))
        gt_img = core.imresize(gt_img, sizes=(round(inp_size*sf),round(inp_size*sf)))
        
        pred = model(inp, scale_factor=None, size=(round(inp_size*sf),round(inp_size*sf)), mode='train')
  
        loss = loss_fn(pred, gt_img)
        psnr = metric_fn(pred, gt_img)
        
        # tensorboard
        writer.add_scalars('loss', {'train': loss.item()}, (epoch-1)*iter_per_epoch + iteration)
        writer.add_scalars('psnr', {'train': psnr}, (epoch-1)*iter_per_epoch + iteration)
        iteration += 1
        
        train_loss.add(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pred = None; loss = None
        
    return train_loss.item()


def main(config_, save_path):
    global config, log, writer
    config = config_
    log, writer = utils.set_save_path(save_path, remove=False)
    with open(os.path.join(save_path, 'config.yaml'), 'w') as f:
        yaml.dump(config, f, sort_keys=False)

    train_loader  = make_data_loaders()
    if config.get('data_norm') is None:
        config['data_norm'] = {
            'inp': {'sub': [0], 'div': [1]},
            'gt': {'sub': [0], 'div': [1]}
        }

    model, optimizer, epoch_start, lr_scheduler = prepare_training()

    n_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    if n_gpus > 1:
        model = nn.parallel.DataParallel(model)

    epoch_max = config['epoch_max']
    epoch_val = config.get('epoch_val')
    epoch_save = config.get('epoch_save')
    max_val_v = -1e18

    timer = utils.Timer()

    for epoch in range(epoch_start, epoch_max + 1):
        t_epoch_start = timer.t()
        log_info = ['epoch {}/{}'.format(epoch, epoch_max)]

        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)

        train_loss = train(train_loader, model, optimizer, epoch, config)
        if lr_scheduler is not None:
            lr_scheduler.step()

        log_info.append('train: loss={:.4f}'.format(train_loss))
        log_info.append('lr={:.4e}'.format(optimizer.param_groups[0]['lr']))
#         writer.add_scalars('loss', {'train': train_loss}, epoch)

        if n_gpus > 1:
            model_ = model.module
        else:
            model_ = model
        model_spec = config['model']
        model_spec['sd'] = model_.state_dict()
        optimizer_spec = config['optimizer']
        optimizer_spec['sd'] = optimizer.state_dict()
        sv_file = {
            'model': model_spec,
            'optimizer': optimizer_spec,
            'epoch': epoch
        }

        torch.save(sv_file, os.path.join(save_path, 'epoch-last.pth'))

        if (epoch_save is not None) and (epoch % epoch_save == 0):
            torch.save(sv_file,
                os.path.join(save_path, 'epoch-{}.pth'.format(epoch)))

        if (epoch_val is not None) and (epoch % epoch_val == 0):
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
                scale_factors = [2, 2.5, 3, 3.5, 4, 6, 12]
            model.eval()

            for sf in scale_factors:
                val_res_set14 = eval(model, 'Set14', save_path, scale_factor=sf, config=config)
                val_res_set5 = eval(model, 'Set5', save_path, scale_factor=sf, config=config)
                if sf == 4:
                    val_sf4 = val_res_set14
                log_info.append('SF{}:{:.4f}/{:.4f}'.format(sf,val_res_set5, val_res_set14))


            model.train()
            if 4 in scale_factors:
                if val_sf4 > max_val_v:
                    max_val_v = val_sf4
                    torch.save(sv_file, os.path.join(save_path, 'epoch-best.pth'))

        t = timer.t()
        prog = (epoch - epoch_start + 1) / (epoch_max - epoch_start + 1)
        t_epoch = utils.time_text(t - t_epoch_start)
        t_elapsed, t_all = utils.time_text(t), utils.time_text(t / prog)
        log_info.append('{} {}/{}'.format(t_epoch, t_elapsed, t_all))

        log(', '.join(log_info))
        writer.flush()

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