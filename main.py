import os
import yaml
import math
import random
import argparse
import itertools
import numpy as np
import os.path as osp
import matplotlib.pyplot as plt

from PIL import Image
from time import time
from os import path as osp
from multiprocessing import Pool

import torch
from torch.utils import data

from torchvision import transforms
from torchvision import datasets

from core import dist_util
from core.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    create_classifier,
    args_to_dict,
    add_dict_to_argparser,
)
from core.sample_utils import (
    get_DiME_iterative_sampling,
    get_FastDiME_iterative_sampling,
    get_GMD_iterative_sampling,
    clean_class_cond_fn,
    dist_cond_fn,
    ImageSaver,
    ImageSaverCF,
    SlowSingleLabel,
    Normalizer,
    load_from_DDP_model,
    PerceptualLoss,
    X_T_Saver,
    Z_T_Saver,
    ChunkedDataset,
    generate_mask,
)
from core.image_datasets import CelebADataset, CelebAMiniVal, ShortcutCelebADataset 
from core.gaussian_diffusion import _extract_into_tensor
from core.classifier.densenet import ClassificationModel
from copy import deepcopy
import matplotlib
matplotlib.use('Agg')  # to disable display



# =======================================================
# =======================================================
# Functions
# =======================================================
# =======================================================


def create_args():
    defaults = dict(
        gpu='0',
        method = 'fastdime',
        num_batches=50,
        batch_size=16,
        query_label=-1,
        target_label=-1,
        task_label=39,
        use_train=False,
        dataset='CelebA',
        shortcut_label_name='Smiling',
        task_label_name='Young',
        percentage=0.5,
        n_samples=1000,

        # path args
        exp_name='',
        output_path='',
        classifier_path='models/classifier.pth',
        oracle_path='models/oracle.pth',
        model_path="models/ddpm-celeba.pt",
        data_dir="/scratch/ppar/data/img_align_celeba/",
        
        # sampling args
        seed=4,
        clip_denoised=True,
        use_logits=False,
        use_ddim=False,
        use_sampling_on_x_t=True,
        scale_grads=False,
        guided_iterations=9999999,
        classifier_scales='8,10,15',
        l1_loss=0.0,
        l2_loss=0.0,
        l_perc=0.0,
        l_perc_layer=18,
        start_step=60,
        warmup_step=30,
        dilation=5,
        masking_threshold=0.15,
        self_optimized_masking=True,

        # evaluation args
        merge_and_eval=False,  # when all chunks have finished, run it with this flag

        # misc args
        num_chunks=1,
        chunk=0,
        save_x_t=False,
        save_z_t=False,
        save_images=True
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser.parse_args()


# =======================================================
# =======================================================
# Merge all chunks' information and compute the
# overall metrics
# =======================================================
# =======================================================


def mean(array):
    m = np.mean(array).item()
    return 0 if math.isnan(m) else m


def merge_and_compute_overall_metrics(args, device):

    def div(q, p):
        if p == 0:
            return 0
        return q / p

    print('Merging all results ...')

    # read all yaml files containing the info to add them together
    summary = {
        'class-cor': {'cf-cor': {'bkl': 0, 'l_1': 0, 'n': 0, 'FVA': 0, 'MNAC': 0},
                      'cf-inc': {'bkl': 0, 'l_1': 0, 'n': 0, 'FVA': 0, 'MNAC': 0},
                      'bkl': 0, 'l_1': 0, 'n': 0, 'FVA': 0, 'MNAC': 0},
        'class-inc': {'cf-cor': {'bkl': 0, 'l_1': 0, 'n': 0, 'FVA': 0, 'MNAC': 0},
                      'cf-inc': {'bkl': 0, 'l_1': 0, 'n': 0, 'FVA': 0, 'MNAC': 0},
                      'bkl': 0, 'l_1': 0, 'n': 0, 'FVA': 0, 'MNAC': 0},
        'cf-cor': {'bkl': 0, 'l_1': 0, 'n': 0, 'FVA': 0, 'MNAC': 0},
        'cf-inc': {'bkl': 0, 'l_1': 0, 'n': 0, 'FVA': 0, 'MNAC': 0},
        'clean acc': 0,
        'cf acc': 0,
        'bkl': 0, 'l_1': 0, 'n': 0, 'FVA': 0, 'MNAC': 0,
    }

    for chunk in range(args.num_chunks):
        yaml_path = osp.join(args.output_path, 'Results', args.exp_name,
                             f'chunk-{chunk}_num-chunks-{args.num_chunks}_summary.yaml')

        with open(yaml_path, 'r') as f:
            chunk_summary = yaml.load(f, Loader=yaml.FullLoader)

        summary['clean acc'] += chunk_summary['clean acc'] * chunk_summary['n']
        summary['cf acc'] += chunk_summary['cf acc'] * chunk_summary['n']
        
        summary['n'] += chunk_summary['n']

        summary['class-cor']['n'] += chunk_summary['class-cor']['n']
        summary['class-inc']['n'] += chunk_summary['class-inc']['n']

        summary['cf-cor']['n'] += chunk_summary['cf-cor']['n']
        summary['cf-inc']['n'] += chunk_summary['cf-inc']['n']

        summary['class-cor']['cf-cor']['n'] += chunk_summary['class-cor']['cf-cor']['n']
        summary['class-cor']['cf-inc']['n'] += chunk_summary['class-cor']['cf-inc']['n']
        summary['class-inc']['cf-cor']['n'] += chunk_summary['class-inc']['cf-cor']['n']
        summary['class-inc']['cf-inc']['n'] += chunk_summary['class-inc']['cf-inc']['n']


        for k in ['bkl', 'l_1', 'FVA', 'MNAC']:
            summary[k] += chunk_summary[k] * chunk_summary['n']

            summary['class-cor'][k] += chunk_summary['class-cor'][k] * chunk_summary['class-cor']['n']
            summary['class-inc'][k] += chunk_summary['class-inc'][k] * chunk_summary['class-inc']['n']

            summary['cf-cor'][k] += chunk_summary['cf-cor'][k] * chunk_summary['cf-cor']['n']
            summary['cf-inc'][k] += chunk_summary['cf-inc'][k] * chunk_summary['cf-inc']['n']

            summary['class-cor']['cf-cor'][k] += chunk_summary['class-cor']['cf-cor'][k] * chunk_summary['class-cor']['cf-cor']['n']
            summary['class-cor']['cf-inc'][k] += chunk_summary['class-cor']['cf-inc'][k] * chunk_summary['class-cor']['cf-inc']['n']
            summary['class-inc']['cf-cor'][k] += chunk_summary['class-inc']['cf-cor'][k] * chunk_summary['class-inc']['cf-cor']['n']
            summary['class-inc']['cf-inc'][k] += chunk_summary['class-inc']['cf-inc'][k] * chunk_summary['class-inc']['cf-inc']['n']

    for k in ['cf acc', 'clean acc']:
        summary[k] = div(summary[k], summary['n'])

    for k in ['bkl', 'l_1', 'FVA', 'MNAC']:
        summary[k] = div(summary[k], summary['n'])

        summary['class-cor'][k] = div(summary['class-cor'][k], summary['class-cor']['n'])
        summary['class-inc'][k] = div(summary['class-inc'][k], summary['class-inc']['n'])

        summary['cf-cor'][k] = div(summary['cf-cor'][k], summary['cf-cor']['n'])
        summary['cf-inc'][k] = div(summary['cf-inc'][k], summary['cf-inc']['n'])

        summary['class-cor']['cf-cor'][k] = div(summary['class-cor']['cf-cor'][k], summary['class-cor']['cf-cor']['n'])
        summary['class-cor']['cf-inc'][k] = div(summary['class-cor']['cf-inc'][k], summary['class-cor']['cf-inc']['n'])
        summary['class-inc']['cf-cor'][k] = div(summary['class-inc']['cf-cor'][k], summary['class-inc']['cf-cor']['n'])
        summary['class-inc']['cf-inc'][k] = div(summary['class-inc']['cf-inc'][k], summary['class-inc']['cf-inc']['n'])

    # summary is ready to save
    print('done')
    print('Acc on the set:', summary['clean acc'])
    print('CF Acc on the set:', summary['cf acc'])

    with open(osp.join(args.output_path, 'Results', args.exp_name, 'summary.yaml'), 'w') as f:
        yaml.dump(summary, f)


# =======================================================
# =======================================================
# Main
# =======================================================
# =======================================================


def main():

    args = create_args()
    # print(args)
    os.makedirs(osp.join(args.output_path, 'Results', args.exp_name),
                exist_ok=True)
    print()
    print(f'PID: {os.getpid()}\n')
    
    print(f'Counterfactual Method {args.method}')

    # ========================================
    # Evaluate all feature in case of 
    if args.merge_and_eval:
        merge_and_compute_overall_metrics(args, dist_util.dev())
        return  # finish the script

    # ========================================
    # Set seeds

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # ========================================
    # Load Dataset

    if args.dataset == 'CelebA':
        dataset = CelebADataset(image_size=args.image_size,
                                data_dir=args.data_dir,
                                partition='train' if args.use_train else 'val',
                                random_crop=False,
                                random_flip=False,
                                query_label=args.query_label)

    elif 'CelebAMV' in args.dataset:
        if args.dataset == 'CelebAMV':
            csv_file ='utils/minival.csv'
        elif args.dataset == 'CelebAMV4':
            csv_file ='utils/minival4k.csv'
            
        dataset = CelebAMiniVal(image_size=args.image_size,
                                data_dir=args.data_dir,
                                csv_file=csv_file,
                                random_crop=False,
                                random_flip=False,
                                query_label=args.query_label)
    

    elif args.dataset == 'ShortcutCelebA':
        dataset = ShortcutCelebADataset(image_size=args.image_size,
                                      data_dir=args.data_dir,
                                      partition='train' if args.use_train else 'val',
                                      random_crop=False,
                                      random_flip=False,
                                      query_label=31,
                                      task_label=39,
                                      shortcut_label_name='Smiling',
                                      task_label_name='Young',
                                      percentage=0.5,
                                      n_samples=1000)

        save_cf_imgs = ImageSaverCF(args.output_path, args.exp_name, extention='.jpg') if args.save_images else None
    
    else:

        raise Exception(f'dataset {args.dataset} not implemented')

    if len(dataset) - args.batch_size * args.num_batches > 0:
        dataset = SlowSingleLabel(query_label=1 - args.target_label if args.target_label != -1 else -1,
                                  dataset=dataset,
                                  maxlen=args.batch_size * args.num_batches)

    # breaks the dataset into chunks
    dataset = ChunkedDataset(dataset=dataset,
                             chunk=args.chunk,
                             num_chunks=args.num_chunks)

    print('Images on the dataset:', len(dataset))

    loader = data.DataLoader(dataset, batch_size=args.batch_size,
                             shuffle=False,
                             num_workers=4, pin_memory=True)

    # ========================================
    # load models

    print('Loading Model and diffusion model')
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    def model_fn(x, t, y=None):
        assert y is not None
        return model(x, t, y if args.class_cond else None)
    
    print('Loading Classifier')

    classifier = ClassificationModel(args.classifier_path, args.query_label).to(dist_util.dev())
    classifier.eval()

    # ========================================
    # Distance losses

    if args.l_perc != 0:
        print('Loading Perceptual Loss')
        vggloss = PerceptualLoss(layer=args.l_perc_layer,
                                 c=args.l_perc).to(dist_util.dev())
        vggloss.eval()
    else:
        vggloss = None


    cond_fn = clean_class_cond_fn
    # ========================================
    # get sampling function self-optmized masking variable
    if args.method == 'fastdime':
        sample_fn = get_FastDiME_iterative_sampling(use_sampling=args.use_sampling_on_x_t)
        self_optimized_masking = True
        
    elif args.method == 'fastdime2':
        sample_fn = get_FastDiME_iterative_sampling(use_sampling=args.use_sampling_on_x_t)
        self_optimized_masking = False # no masking included in first step
        sample_fn_step_2 = get_FastDiME_iterative_sampling(use_sampling=args.use_sampling_on_x_t)

    elif args.method == 'fastdime2+':
        sample_fn = get_FastDiME_iterative_sampling(use_sampling=args.use_sampling_on_x_t)
        self_optimized_masking = True # self-optimized masking included in first step
        sample_fn_step_2 = get_FastDiME_iterative_sampling(use_sampling=args.use_sampling_on_x_t)
        
    elif args.method == 'fastdimenomask':
        sample_fn = get_FastDiME_iterative_sampling(use_sampling=args.use_sampling_on_x_t)
        self_optimized_masking = False
        
    elif args.method == 'gmd':
        sample_fn = get_GMD_iterative_sampling(use_sampling=args.use_sampling_on_x_t)
        self_optimized_masking = True
        
    elif args.method == 'dime':
        sample_fn = get_DiME_iterative_sampling(use_sampling=args.use_sampling_on_x_t)
        self_optimized_masking = False

    else:
        raise Exception(f'method {args.method} not implemented')

    x_t_saver = X_T_Saver(args.output_path, args.exp_name) if args.save_x_t else None
    z_t_saver = Z_T_Saver(args.output_path, args.exp_name) if args.save_z_t else None
    save_imgs = ImageSaver(args.output_path, args.exp_name, extention='.jpg') if args.save_images else None

    current_idx = 0
    start_time = time()
    batch_times = []

    stats = {
        'n': 0,
        'flipped': 0,
        'bkl': [],
        'mad': [],
        'l_1': [],
        'pred': [],
        'cf pred': [],
        'target': [],
        'label': [],
    }

    acc = 0
    n = 0
    classifier_scales = [float(x) for x in args.classifier_scales.split(',')]

    print('Starting Image Generation')
    for idx, (indexes, img, lab, task_label) in enumerate(loader):
        print(f'[Chunk {args.chunk + 1} / {args.num_chunks}] {idx} / {min(args.num_batches, len(loader))} | Time: {int(time() - start_time)}s')
        # print(idx, indexes)
        img = img.to(dist_util.dev())
        I = (img / 2) + 0.5
        lab = lab.to(dist_util.dev(), dtype=torch.long)
        t = torch.zeros(img.size(0), device=dist_util.dev(),
                        dtype=torch.long)

        # Initial Classification, no noise included
        with torch.no_grad():
            logits = classifier(img)
            pred = (logits > 0).long() 

        acc += (pred == lab).sum().item()
        n += lab.size(0)

        # as the model is binary, the target will always be the inverse of the prediction
        target = 1 - pred

        t = torch.ones_like(t) * args.start_step

        # add noise to the input image 
        noise_img = diffusion.q_sample(img, t)

        transformed = torch.zeros_like(lab).bool()

        batch_start_time = time()

        for jdx, classifier_scale in enumerate(classifier_scales):

            # choose the target label
            model_kwargs = {}
            model_kwargs['y'] = target[~transformed]

            # sample image from the noisy_img
            cfs, xs_t_s, zs_t_s = sample_fn(
                diffusion,
                model_fn,
                img[~transformed, ...].shape,
                args.start_step,
                img[~transformed, ...],
                t,
                z_t=noise_img[~transformed, ...],
                clip_denoised=args.clip_denoised,
                model_kwargs=model_kwargs,
                device=dist_util.dev(),
                class_grad_fn=cond_fn,
                class_grad_kwargs={'y': target[~transformed],
                                   'classifier': classifier,
                                   's': classifier_scale,
                                   'use_logits': args.use_logits},
                dist_grad_fn=dist_cond_fn,
                dist_grad_kargs={'l1_loss': args.l1_loss,
                                 'l2_loss': args.l2_loss,
                                 'l_perc': vggloss},
                guided_iterations=args.guided_iterations,
                is_x_t_sampling=False,
                fast_dime_kwargs={'warmup_step':args.warmup_step,
                                  'dilation':args.dilation,
                                  'masking_threshold':args.masking_threshold,
                                  'self_optimized_masking': self_optimized_masking,
                                  'boolmask': None},
                scale_grads=args.scale_grads

            )

            if '2' in args.method:
                # extract fixed mask for our 2-step approaches
                mask, dil_mask = generate_mask(img[~transformed, ...], cfs, args.dilation)
                boolmask = (dil_mask < args.masking_threshold).float()

                cfs, xs_t_s, zs_t_s = sample_fn_step_2(
                diffusion, 
                model_fn,
                img[~transformed, ...].shape,
                args.start_step,
                img[~transformed, ...],
                t,
                z_t=noise_img[~transformed, ...],
                clip_denoised=args.clip_denoised,
                model_kwargs=model_kwargs,
                device=dist_util.dev(),
                class_grad_fn=cond_fn,
                class_grad_kwargs={'y': target[~transformed],
                                   'classifier': classifier,
                                   's': classifier_scale,
                                   'use_logits': args.use_logits},
                dist_grad_fn=dist_cond_fn,
                dist_grad_kargs={'l1_loss': args.l1_loss,
                                 'l2_loss': args.l2_loss,
                                 'l_perc': vggloss},
                guided_iterations=args.guided_iterations,
                is_x_t_sampling=False,
                fast_dime_kwargs={'self_optimized_masking': False,
                                  'boolmask': boolmask},
                scale_grads=args.scale_grads
            )


            # evaluate the cf and check whether the model flipped the prediction
            with torch.no_grad():
                cfsl = classifier(cfs)
                cfsp = cfsl > 0
            
            if jdx == 0:
                cf = cfs.clone().detach()
                x_t_s = [xp.clone().detach() for xp in xs_t_s]
                z_t_s = [zp.clone().detach() for zp in zs_t_s]

            cf[~transformed] = cfs
            for kdx in range(len(x_t_s)):
                x_t_s[kdx][~transformed] = xs_t_s[kdx]
                z_t_s[kdx][~transformed] = zs_t_s[kdx]
            transformed[~transformed] = target[~transformed] == cfsp

            if transformed.float().sum().item() == transformed.size(0):
                break
        

        batch_end_time = time()
        batch_times.append(batch_end_time - batch_start_time)

        if args.save_x_t:
            x_t_saver(x_t_s, indexes=indexes)

        if args.save_z_t:
            z_t_saver(z_t_s, indexes=indexes)

        with torch.no_grad():
            logits_cf = classifier(cf)
            pred_cf = (logits_cf > 0).long() 

            # process images
            cf = ((cf + 1) * 127.5).clamp(0, 255).to(torch.uint8)
            cf = cf.permute(0, 2, 3, 1)
            cf = cf.contiguous().cpu()

            I = (I * 255).to(torch.uint8)
            I = I.permute(0, 2, 3, 1)
            I = I.contiguous().cpu()

            noise_img = ((noise_img + 1) * 127.5).clamp(0, 255).to(torch.uint8)
            noise_img = noise_img.permute(0, 2, 3, 1)
            noise_img = noise_img.contiguous().cpu()

            # add metrics
            dist_cf = torch.sigmoid(logits_cf)
            dist_cf[target == 0] = 1 - dist_cf[target == 0]
            bkl = (1 - dist_cf).detach().cpu()

            cf_output = torch.sigmoid(logits_cf)
            original_output = torch.sigmoid(logits)
            mad = torch.abs(original_output - cf_output).detach().cpu()

            # dists
            I_f = (I.to(dtype=torch.float) / 255).view(I.size(0), -1)
            cf_f = (cf.to(dtype=torch.float) / 255).view(I.size(0), -1)
            l_1 = (I_f - cf_f).abs().mean(dim=1).detach().cpu()

            stats['l_1'].append(l_1)
            stats['n'] += I.size(0)
            stats['bkl'].append(bkl)
            stats['mad'].append(mad)
            stats['flipped'] += (pred_cf == target).sum().item()
            stats['cf pred'].append(pred_cf.detach().cpu())
            stats['target'].append(target.detach().cpu())
            stats['label'].append(lab.detach().cpu())
            stats['pred'].append(pred.detach().cpu())

        if args.save_images:
            if 'Shortcut' not in args.dataset:
                save_imgs(I.numpy(), cf.numpy(), noise_img.numpy(),
                        target, lab, pred, pred_cf,
                        bkl.numpy(), mad.numpy(),
                        original_output.detach().cpu().numpy(),
                        cf_output.detach().cpu().numpy(),
                        l_1, indexes=indexes.numpy())
            else:
                save_cf_imgs(I.numpy(), cf.numpy(), noise_img.numpy(),
                    target, lab, task_label, pred, pred_cf,
                    bkl.numpy(),
                    l_1, indexes=indexes.numpy())


        if (idx + 1) == min(args.num_batches, len(loader)):
            print(f'[Chunk {args.chunk + 1} / {args.num_chunks}] {idx + 1} / {min(args.num_batches, len(loader))} | Time: {int(time() - start_time)}s')
            print('\nDone')
            break

        current_idx += I.size(0)

    
    # Calculate mean and std of batch times
    mean_batch_time = np.mean(batch_times)
    std_batch_time = np.std(batch_times)

    print()
    print(f'Method {args.method}')
    print(f'Mean time per batch: {np.round(mean_batch_time,1)}s')
    print(f'Std time per batch: {np.round(std_batch_time,1)}s')
    print()

    # write summary for all four combinations
    summary = {
        'class-cor': {'cf-cor': {'bkl': 0, 'mad': 0,'l_1': 0, 'n': 0},
                      'cf-inc': {'bkl': 0, 'mad': 0, 'l_1': 0, 'n': 0},
                      'bkl': 0, 'mad': 0, 'l_1': 0, 'n': 0},
        'class-inc': {'cf-cor': {'bkl': 0, 'mad': 0, 'l_1': 0, 'n': 0},
                      'cf-inc': {'bkl': 0, 'mad': 0, 'l_1': 0, 'n': 0},
                      'bkl': 0, 'l_1': 0, 'mad': 0, 'n': 0},
        'cf-cor': {'bkl': 0, 'mad': 0, 'l_1': 0, 'n': 0},
        'cf-inc': {'bkl': 0, 'mad': 0, 'l_1': 0, 'n': 0},
        'clean acc': 100 * acc / n,
        'cf acc': stats['flipped'] / n,
        'bkl': 0, 'mad': 0, 'l_1': 0, 'n': 0, 'FVA': 0, 'MNAC': 0,
    }

    for k in stats.keys():
        if k in ['flipped', 'n']:
            continue
        stats[k] = torch.cat(stats[k]).numpy()

    for k in ['bkl', 'mad', 'l_1']:

        summary['class-cor']['cf-cor'][k] = mean(stats[k][(stats['label'] == stats['pred']) & (stats['target'] == stats['cf pred'])])
        summary['class-inc']['cf-cor'][k] = mean(stats[k][(stats['label'] != stats['pred']) & (stats['target'] == stats['cf pred'])])
        summary['class-cor']['cf-inc'][k] = mean(stats[k][(stats['label'] == stats['pred']) & (stats['target'] != stats['cf pred'])])
        summary['class-inc']['cf-inc'][k] = mean(stats[k][(stats['label'] != stats['pred']) & (stats['target'] != stats['cf pred'])])

        summary['class-cor'][k] = mean(stats[k][stats['label'] == stats['pred']])
        summary['class-inc'][k] = mean(stats[k][stats['label'] != stats['pred']])

        summary['cf-cor'][k] = mean(stats[k][stats['target'] == stats['cf pred']])
        summary['cf-inc'][k] = mean(stats[k][stats['target'] != stats['cf pred']])

        summary[k] = mean(stats[k])

    summary['class-cor']['cf-cor']['n'] = len(stats[k][(stats['label'] == stats['pred']) & (stats['target'] == stats['cf pred'])])
    summary['class-inc']['cf-cor']['n'] = len(stats[k][(stats['label'] != stats['pred']) & (stats['target'] == stats['cf pred'])])
    summary['class-cor']['cf-inc']['n'] = len(stats[k][(stats['label'] == stats['pred']) & (stats['target'] != stats['cf pred'])])
    summary['class-inc']['cf-inc']['n'] = len(stats[k][(stats['label'] != stats['pred']) & (stats['target'] != stats['cf pred'])])

    summary['class-cor']['n'] = len(stats[k][stats['label'] == stats['pred']])
    summary['class-inc']['n'] = len(stats[k][stats['label'] != stats['pred']])

    summary['cf-cor']['n'] = len(stats[k][stats['target'] == stats['cf pred']])
    summary['cf-inc']['n'] = len(stats[k][stats['target'] != stats['cf pred']])

    summary['n'] = n

    print('ACC ON THIS SET:', 100 * acc / n)
    stats['acc'] = 100 * acc / n

    prefix = f'chunk-{args.chunk}_num-chunks-{args.num_chunks}_' if args.num_chunks != 1 else ''
    torch.save(stats, osp.join(args.output_path, 'Results', args.exp_name, prefix + 'stats.pth'))

    # save summary
    with open(osp.join(args.output_path, 'Results', args.exp_name, prefix + 'summary.yaml'), 'w') as f:
        yaml.dump(summary, f)


if __name__ == '__main__':
    main()