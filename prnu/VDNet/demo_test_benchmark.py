#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Zongsheng Yue 2019-05-16 16:20:01

import sys
sys.path.append('./')
import numpy as np
import torch
from .networks import VDN
from skimage.measure import compare_psnr, compare_ssim
from skimage import img_as_float, img_as_ubyte
from .utils import load_state_dict_cpu
from matplotlib import pyplot as plt
import time
from scipy.io import loadmat, savemat
import imageio
import os
import random

def extract_noise_VDNet(png_noisy):
    id =  random.randrange(10000,1000000)
    use_gpu = True
    C = 3
    dep_U = 4

    checkpoint = torch.load('prnu/VDNet/model_state/model_state_SIDD')
    net = VDN(C, dep_U=dep_U, wf=64)
    if use_gpu:
        net = torch.nn.DataParallel(net).cuda()
        net.load_state_dict(checkpoint)
    else:
        load_state_dict_cpu(net, checkpoint)
    net.eval()

    #converts .png to .mat
    png_noisy = (png_noisy - np.min(png_noisy)) / np.max(png_noisy - np.min(png_noisy))
    png_noisy = np.float32(png_noisy)
    mdic = {'InoisySRGB': png_noisy}
    savemat('./tmp_'+str(id)+'.mat', mdic)
    im_noisy = loadmat('./tmp_'+str(id)+'.mat')['InoisySRGB']
    os.remove('./tmp_'+str(id)+'.mat')


    H, W, _ = im_noisy.shape

    if H % 2**dep_U != 0:
        H -= H % 2**dep_U
    if W % 2**dep_U != 0:
        W -= W % 2**dep_U
    im_noisy = im_noisy[:H, :W, ]

    im_noisy = torch.from_numpy(im_noisy.transpose((2,0,1))[np.newaxis,])
    if use_gpu:
        im_noisy = im_noisy.cuda()
    with torch.autograd.set_grad_enabled(False):
        torch.cuda.synchronize()
        tic = time.perf_counter()
        phi_Z = net(im_noisy, 'test')
        torch.cuda.synchronize()
        toc = time.perf_counter()
        err = phi_Z.cpu().numpy()
    if use_gpu:
        im_noisy = im_noisy.cpu().numpy()
    else:
        im_noisy = im_noisy.numpy()

    residual = err[:, :C,]
    im_denoise = im_noisy - residual
    im_denoise = np.transpose(im_denoise.squeeze(), (1,2,0))
    im_denoise = img_as_ubyte(im_denoise.clip(0,1))
    im_noisy = np.transpose(im_noisy.squeeze(), (1,2,0))
    im_noisy = img_as_ubyte(im_noisy.clip(0,1))


    residual = residual.reshape([residual.shape[1],residual.shape[2],residual.shape[3]])
    residual = residual.transpose(1,2,0)

    return residual

