# -*- coding: UTF-8 -*-
"""
@author: Luca Bondi (luca.bondi@polimi.it)
@author: Paolo Bestagini (paolo.bestagini@polimi.it)
@author: Nicolò Bonettini (nicolo.bonettini@polimi.it)
Politecnico di Milano 2018
"""
import os
from glob import glob
from multiprocessing import cpu_count, Pool

import numpy as np
from PIL import Image
import time

import prnu
import sys
import matplotlib.pyplot as plt
import argparse



parser = argparse.ArgumentParser(description='This program extracts camera fingerprint using VDNet and VDID and compares them with the original implementation')
parser.add_argument("-denoiser", help="[original (default) | vdnet | vdid]", default='original')
parser.add_argument("-rm_zero_mean", help='Removes zero mean normalization', action='store_true',
                    default=False)
parser.add_argument("-rm_wiener", help='Removes Wiener filter', action='store_true',
                    default=False)
args = parser.parse_args()


def main():
    """
    Main example script. Load a subset of flatfield and natural images from Dresden.
    For each device compute the fingerprint from all the flatfield images.
    For each natural image compute the noise residual.
    Check the detection performance obtained with cross-correlation and PCE
    :return:
    """
    start = time.time()

    denoiser = args.denoiser
    remove_zero_m = args.rm_zero_mean
    remove_wiener = args.rm_wiener
    prnu.define_param(denoiser, remove_zero_m, remove_wiener)

    print('Denoiser: ' + denoiser)
    print('Remove zero mean: ' + str(remove_zero_m))
    print('Remove wiener: ' + str(remove_wiener) + '\n')

    ff_dirlist = np.array(sorted(glob('test/data/ff-revision-2/*.jpg')))
    ff_device = np.array([os.path.split(i)[1].rsplit('_', 1)[0] for i in ff_dirlist])

    nat_dirlist = np.array(sorted(glob('test/data/nat-revision-2/*.jpg')))
    nat_device = np.array([os.path.split(i)[1].rsplit('_', 1)[0] for i in nat_dirlist])

    print('Computing fingerprints')
    fingerprint_device = sorted(np.unique(ff_device))

    k = []
    for device in fingerprint_device:
        imgs = []
        for img_path in ff_dirlist[ff_device == device]:
            im = Image.open(img_path)
            im_arr = np.asarray(im)
            if im_arr.dtype != np.uint8:
                print('Error while reading image: {}'.format(img_path))
                continue
            if im_arr.ndim != 3:
                print('Image is not RGB: {}'.format(img_path))
                continue
            im_cut = prnu.cut_ctr(im_arr, (512, 512, 3))
            imgs += [im_cut]
        k += [prnu.extract_multiple_aligned(imgs, processes=1)]


    k = np.stack(k, 0)


    print('Computing residuals')

    imgs = []
    for img_path in nat_dirlist:
        imgs += [prnu.cut_ctr(np.asarray(Image.open(img_path)), (512, 512, 3))]


    w = []
    for img in imgs:
        w.append(prnu.extract_single(img))


    w = np.stack(w, 0)

    # Computing Ground Truth
    gt = prnu.gt(fingerprint_device, nat_device)

    print('Computing cross correlation')
    cc_aligned_rot = prnu.aligned_cc(k, w)['cc']

    print('Computing statistics cross correlation')
    stats_cc = prnu.stats(cc_aligned_rot, gt)

    print('Computing PCE')
    pce_rot = np.zeros((len(fingerprint_device), len(nat_device)))

    for fingerprint_idx, fingerprint_k in enumerate(k):
        tn, tp, fp, fn = 0, 0, 0, 0
        pce_values = []
        natural_indices = []
        for natural_idx, natural_w in enumerate(w):

            cc2d = prnu.crosscorr_2d(fingerprint_k, natural_w)
            prnu_pce = prnu.pce(cc2d)['pce']
            pce_rot[fingerprint_idx, natural_idx] = prnu_pce
            pce_values.append(prnu_pce)
            natural_indices.append(natural_idx)
            if fingerprint_device[fingerprint_idx] == nat_device[natural_idx]:
                if prnu_pce > 60.:
                    tp += 1.
                else:
                    fn += 1.
            else:
                if prnu_pce > 60.:
                    fp += 1.
                else:
                    tn += 1.
        tpr = tp / (tp + fn)
        fpr = fp / (fp + tn)

        plt.title('PRNU for ' + str(fingerprint_device[fingerprint_idx]) + ' - ' + denoiser)
        plt.xlabel('query images')
        plt.ylabel('PRNU')

        plt.bar(natural_indices, pce_values)
        plt.text(0.85, 0.85, 'TPR: ' + str(round(tpr, 2)) + '\nFPR: '+ str(round(fpr, 2)),
         fontsize=10, color='k',
         ha='left', va='bottom',
         transform=plt.gca().transAxes)
        plt.axhline(y=60, color='r', linestyle='-')
        plt.xticks(natural_indices)
        plt.savefig('plots/'+ denoiser + '/' +str(fingerprint_device[fingerprint_idx])+'.png')

        plt.clf()

    print('Computing statistics on PCE')
    stats_pce = prnu.stats(pce_rot, gt)

    print('AUC on CC {:.2f}'.format(stats_cc['auc']))
    print('AUC on PCE {:.2f}'.format(stats_pce['auc']))

    end = time.time()
    elapsed = int(end - start)
    print('Elapsed time: '+ str(elapsed) + ' seconds')

if __name__ == '__main__':
    main()
