# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# training code for DUSt3R
# --------------------------------------------------------
# References:
# MAE: https://github.com/facebookresearch/mae
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import argparse
import datetime
import json
import numpy as np
import os
from os.path import join
import sys
import time
import math
from collections import defaultdict
from pathlib import Path
from typing import Sized

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
torch.backends.cuda.matmul.allow_tf32 = True  # for gpu >= Ampere and pytorch >= 1.12

from dust3r.model import AsymmetricCroCo3DStereo, inf  # noqa: F401, needed when loading the model
from dust3r.datasets import get_data_loader  # noqa
from dust3r.losses import *  # noqa: F401, needed when loading the model
from dust3r.inference import loss_of_one_batch  # noqa

import dust3r.utils.path_to_croco  # noqa: F401
import croco.utils.misc as misc  # noqa
from croco.utils.misc import NativeScalerWithGradNormCount as NativeScaler  # noqa


def get_args_parser():
    parser = argparse.ArgumentParser('DUST3R training', add_help=False)
    # model and criterion
    parser.add_argument('--model', default="AsymmetricCroCo3DStereo(patch_embed_cls='ManyAR_PatchEmbed')",
                        type=str, help="string containing the model to build")
    parser.add_argument('--pretrained', default=None, help='path of a starting checkpoint')
    parser.add_argument('--train_criterion', default="ConfLoss(Regr3D(L21, norm_mode='avg_dis'), alpha=0.2)",
                        type=str, help="train criterion")
    parser.add_argument('--test_criterion', default=None, type=str, help="test criterion")

    # dataset
    parser.add_argument('--train_dataset', required=True, type=str, help="training set")
    parser.add_argument('--test_dataset', default='[None]', type=str, help="testing set")

    # training
    parser.add_argument('--seed', default=0, type=int, help="Random seed")
    parser.add_argument('--batch_size', default=64, type=int,
                        help="Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus")
    parser.add_argument('--accum_iter', default=1, type=int,
                        help="Accumulate gradient iterations (for increasing the effective batch size under memory constraints)")
    parser.add_argument('--epochs', default=800, type=int, help="Maximum number of epochs for the scheduler")

    parser.add_argument('--weight_decay', type=float, default=0.05, help="weight decay (default: 0.05)")
    parser.add_argument('--lr', type=float, default=None, metavar='LR', help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1.5e-4, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N', help='epochs to warmup LR')

    parser.add_argument('--amp', type=int, default=0,
                        choices=[0, 1], help="Use Automatic Mixed Precision for pretraining")
    parser.add_argument("--disable_cudnn_benchmark", action='store_true', default=False,
                        help="set cudnn.benchmark = False")
    # others
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    parser.add_argument('--eval_freq', type=int, default=1, help='Test loss evaluation frequency')
    parser.add_argument('--save_freq', default=1, type=int,
                        help='frequence (number of epochs) to save checkpoint in checkpoint-last.pth')
    parser.add_argument('--keep_freq', default=20, type=int,
                        help='frequence (number of epochs) to save checkpoint in checkpoint-%d.pth')
    parser.add_argument('--print_freq', default=20, type=int,
                        help='frequence (number of iterations) to print infos while training')

    # output dir
    parser.add_argument('--output_dir', default='./output/', type=str, help="path where to save the output")
    return parser


def collate_fn(batch):  # batch:[(view1, view2) * batch_size]
    # print(view1["img"].size(), view1["plan_xys"].size())  # Should be torch.Size([8, 3, 224, 224]) torch.Size([8, 733, 2])   
    print("in collate_fn...")
    max_xys_len = max(item[0]["plan_xys"].shape[0] for item in batch)
    print(f"max_xys_len: {max_xys_len}")
    
    view1_img_batched = []
    view2_img_batched = [] 
    plan_xys_batched = []
    image_xys_batched = []
    for view1, view2 in batch:  #(['img', 'plan_xys', 'image_xys'])
        view1_img_batched.append(torch.Tensor(view1["img"]))
        view2_img_batched.append(torch.Tensor(view2["img"]))

        plan_xys_batched.append(torch.from_numpy(np.pad(view1["plan_xys"], ((0, max_xys_len - view1["plan_xys"].shape[0]), (0, 0)), mode="constant", constant_values=0)))
        image_xys_batched.append(torch.from_numpy(np.pad(view1["image_xys"], ((0, max_xys_len - view1["image_xys"].shape[0]), (0, 0)), mode="constant", constant_values=0)))

    view1_img_batched = torch.stack(view1_img_batched)
    view2_img_batched = torch.stack(view2_img_batched)
    plan_xys_batched = torch.stack(plan_xys_batched)
    image_xys_batched = torch.stack(image_xys_batched)
    final_view1 = dict(
        img=view1_img_batched, 
        plan_xys=plan_xys_batched,
        image_xys=image_xys_batched
    )
    final_view2 = dict(       
        img=view2_img_batched,
        plan_xys=plan_xys_batched,
        image_xys=image_xys_batched
    )
    return final_view1, final_view2


def train(args):
    misc.init_distributed_mode(args)
    global_rank = misc.get_rank()
    world_size = misc.get_world_size()

    print("output_dir: " + args.output_dir)
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # auto resume
    last_ckpt_fname = os.path.join(args.output_dir, f'checkpoint-last.pth')
    args.resume = last_ckpt_fname if os.path.isfile(last_ckpt_fname) else None

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    # fix the seed
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = not args.disable_cudnn_benchmark

    # # training dataset and loader
    # print('Building train dataset {:s}'.format(args.train_dataset))
    # #  dataset and loader
    # data_loader_train = build_dataset(args.train_dataset, args.batch_size, args.num_workers, test=False)
    # print('Building test dataset {:s}'.format(args.train_dataset))
    # data_loader_test = {dataset.split('(')[0]: build_dataset(dataset, args.batch_size, args.num_workers, test=True)
    #                     for dataset in args.test_dataset.split('+')}
    data_dir = "/share/phoenix/nfs06/S9/kh775/dataset/megascenes_augmented_exhaustive"
    from torch.utils.data import DataLoader
    from dust3r.datasets.megascenes_augmented import MegaScenesAugmented
    print("Loading training data...")
    data_train = MegaScenesAugmented(join(args.train_dataset, "image_pairs.csv"), data_dir, join(args.train_dataset, "coords"))  # B=2, len(dataloader_train)=8204
    data_loader_train = DataLoader(
        data_train,
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    # for batch in data_loader_train:
    #     view1, view2 = batch
    #     print(view1["img"])
    #     print(view2["img"])
    #     print(view1["img"].size(), view1["plan_xys"].size())  # Should be torch.Size([8, 3, 224, 224]) torch.Size([8, 733, 2])   
    #     break
    # raise Exception
    print("Training data loaded")
    print("Loading testing data...")
    data_test = MegaScenesAugmented(join(args.test_dataset, "image_pairs.csv"), data_dir, join(args.test_dataset, "coords"))
    data_loader_test = DataLoader(
        data_test,
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    print("Test data loaded")

    # model
    # print('Loading model: {:s}'.format(args.model))
    model = eval(args.model)
    print(f'>> Creating train criterion = {args.train_criterion}')
    train_criterion = eval(args.train_criterion).to(device)
    print(f'>> Creating test criterion = {args.test_criterion or args.train_criterion}')
    test_criterion = eval(args.test_criterion or args.criterion).to(device)

    model.to(device)
    model_without_ddp = model
    # print("Model = %s" % str(model_without_ddp))

    if args.pretrained and not args.resume:
        print('Loading pretrained: ', args.pretrained)
        ckpt = torch.load(args.pretrained, map_location=device)
        # print(model.load_state_dict(ckpt['model'], strict=False))
        del ckpt  # in case it occupies memory

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256
    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)
    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    # following timm: set wd as 0 for bias and norm layers
    param_groups = misc.get_parameter_groups(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)
    loss_scaler = NativeScaler()

    def write_log_stats(epoch, train_stats, test_stats):
        if misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()

            log_stats = dict(epoch=epoch, **{f'train_{k}': v for k, v in train_stats.items()})
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")
            log_stats = dict(epoch=epoch, **{f'test_{k}': v for k, v in test_stats.items()})

            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    def save_model(epoch, fname, best_so_far):
        misc.save_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer,
                        loss_scaler=loss_scaler, epoch=epoch, fname=fname, best_so_far=best_so_far)

    best_so_far = misc.load_model(args=args, model_without_ddp=model_without_ddp,
                                  optimizer=optimizer, loss_scaler=loss_scaler)
    if best_so_far is None:
        best_so_far = float('inf')
    log_writer = SummaryWriter(log_dir=args.output_dir)
    # if global_rank == 0 and args.output_dir is not None:
    #     log_writer = SummaryWriter(log_dir=args.output_dir)
    # else:
    #     log_writer = None

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    train_stats = test_stats = {}
    for epoch in range(args.start_epoch, args.epochs + 1):

        # Save immediately the last checkpoint
        if epoch > args.start_epoch:
            if args.save_freq and epoch % args.save_freq == 0 or epoch == args.epochs:
                save_model(epoch - 1, 'last', best_so_far)

        # Train
        print("Starting training...")
        train_stats = train_one_epoch(
            model, train_criterion, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer,
            args=args)

        # Test on multiple datasets
        new_best = False
        if (epoch > 0 and args.eval_freq > 0 and epoch % args.eval_freq == 0):
            test_stats = {}
            # test_name = args.test_dataset.split("/")[-1]
            test_name = "test"
            # for test_name, testset in data_loader_test.items():
            stats = test_one_epoch(model, test_criterion, data_loader_test,
                                    device, epoch, log_writer=log_writer, args=args, prefix=test_name)
            test_stats[test_name] = stats

            # Save best of all
            if stats['loss'] < best_so_far:
                best_so_far = stats['loss']
                new_best = True

        # Save more stuff
        write_log_stats(epoch, train_stats, test_stats)

        if epoch > args.start_epoch:
            if args.keep_freq and epoch % args.keep_freq == 0:
                save_model(epoch - 1, str(epoch), best_so_far)
            if new_best:
                save_model(epoch - 1, 'best', best_so_far)
        if epoch >= args.epochs:
            break  # exit after writing last test to disk

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

    save_final_model(args, args.epochs, model_without_ddp, best_so_far=best_so_far)


def save_final_model(args, epoch, model_without_ddp, best_so_far=None):
    output_dir = Path(args.output_dir)
    checkpoint_path = output_dir / 'checkpoint-final.pth'
    to_save = {
        'args': args,
        'model': model_without_ddp if isinstance(model_without_ddp, dict) else model_without_ddp.cpu().state_dict(),
        'epoch': epoch
    }
    if best_so_far is not None:
        to_save['best_so_far'] = best_so_far
    print(f'>> Saving model to {checkpoint_path} ...')
    misc.save_on_master(to_save, checkpoint_path)


def build_dataset(dataset, batch_size, num_workers, test=False):
    split = ['Train', 'Test'][test]
    print(f'Building {split} Data loader for dataset: ', dataset)
    loader = get_data_loader(dataset,
                             batch_size=batch_size,
                             num_workers=num_workers,
                             pin_mem=True,
                             shuffle=not (test),
                             drop_last=not (test))

    print(f"{split} dataset length: ", len(loader))
    return loader

def reverse_ImgNorm(np_array):
    np_array = np_array * 0.5 + 0.5
    np_array *= 255.0
    np_array = np_array.clip(0, 255).astype(np.uint8)
    return np_array


import io
import matplotlib.pyplot as plt
import PIL.Image
from torchvision.transforms import ToTensor
def get_viz(view1, view2, pred1, pred2):
    def gen_plot(view1, view2, pred1, pred2):
        print("starting gen plot")
        view1_img = view1["img"].permute(0, 2, 3, 1).cpu().numpy()
        view2_img = view2["img"].permute(0, 2, 3, 1).cpu().numpy()

        B = view1_img.shape[0]
        fig, axes = plt.subplots(B, 3, figsize=(10, B*3))
        titles = ["gt", "pred", "image"]
        print("starting for loop")
        for b in range(B):
            view1_img_scaled = reverse_ImgNorm(view1_img[b])
            axes[b, 0].imshow(view1_img_scaled)
            view1_points = view1["plan_xys"][b].cpu().numpy()
            axes[b, 0].scatter(view1_points[:,0], view1_points[:,1], s=5)
            axes[b, 0].set_title(titles[0])
            
            axes[b, 1].imshow(view1_img_scaled)    
            pred2_points = pred2["pts3d_in_other_view"][b].detach().cpu().numpy()
            x_coords = view2["image_xys"][b][:,0].cpu().numpy()
            y_coords = view2["image_xys"][b][:,1].cpu().numpy()
            pred2_points = pred2_points[y_coords, x_coords, :2]
            pred_min = pred2_points.min()
            pred_max = pred2_points.max()
            pred_norm_scaled = (pred2_points - pred_min) / (pred_max - pred_min) * 255.
            axes[b, 1].scatter(pred_norm_scaled[:,0], pred_norm_scaled[:,1], s=5)
            axes[b, 1].set_title(titles[1])  

            view2_img_scaled = reverse_ImgNorm(view2_img[b])
            axes[b, 2].imshow(view2_img_scaled)   
            view2_points = view1["image_xys"][b].cpu().numpy()   
            axes[b, 2].scatter(view2_points[:,0], view2_points[:,1], s=5) 
            axes[b, 2].set_title(titles[2])   
        plt.tight_layout()
        return fig
    viz = gen_plot(view1, view2, pred1, pred2)
    # batch, _, _, _ = view1["img"].size()
    # combined_overlays = []
    # for b in range(batch):
    #     view1_img = view1["img"][b]  #CHW
    #     view2_img = view2["img"][b]  #CHW
    #     view1_points = view1["plan_xys"][b]
    #     view2_points = view2["image_xys"][b]
    #     pred2_points = pred2["pts3d_in_other_view"][b]
    #     x_coords = view2["image_xys"][b][:,0]
    #     y_coords = view2["image_xys"][b][:,1]
    #     pred2_points = pred2_points[y_coords, x_coords, :2]
    #     assert pred2_points.size() == view1_points.size()
    #     viz = gen_plot(view1_img, view2_img, view1_points, pred2_points)
    return viz


def is_main_process():
    return torch.distributed.get_rank() == 0

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Sized, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    args,
                    log_writer=None):
    assert torch.backends.cuda.matmul.allow_tf32 == True

    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    accum_iter = args.accum_iter
    
    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    if hasattr(data_loader, 'dataset') and hasattr(data_loader.dataset, 'set_epoch'):
        data_loader.dataset.set_epoch(epoch)
    if hasattr(data_loader, 'sampler') and hasattr(data_loader.sampler, 'set_epoch'):
        data_loader.sampler.set_epoch(epoch)

    optimizer.zero_grad()

    
    for data_iter_step, batch in enumerate(metric_logger.log_every(data_loader, args.print_freq, header)):
        # # batch: (view1=dict("img":BCHW), view2)
        # view1, view2 = batch
        # import matplotlib.pyplot as plt
        # view1_imgs = view1["img"]
        # view2_imgs = view2["img"]
        # for b in range(view1_imgs.size()[0]):
        #     view1_img = view1_imgs[b].permute(1, 2, 0).cpu().numpy()
        #     plt.imshow(view1_img)
        #     plt.savefig(join("/share/phoenix/nfs06/S9/kh775/code/dust3r/dust3r/utils/test/trainingpy_before_model", f"view1_img_{b}.png"))
        #     plt.close()
        #     view2_img = view2_imgs[b].permute(1, 2, 0).cpu().numpy()
        #     plt.imshow(view2_img)
        #     plt.savefig(join("/share/phoenix/nfs06/S9/kh775/code/dust3r/dust3r/utils/test/trainingpy_before_model", f"view2_img_{b}.png"))
        #     plt.close()
        # raise Exception
        print(f"Start training for {data_iter_step} data_iter_step")
        start_time = time.time()

        epoch_f = epoch + data_iter_step / len(data_loader)

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            misc.adjust_learning_rate(optimizer, epoch_f, args)

        result = loss_of_one_batch(batch, model, criterion, device,
                                       symmetrize_batch=True,
                                       use_amp=bool(args.amp))
        loss, loss_details = result["loss"]
        loss_value = float(loss)

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print(f"Training time for {data_iter_step} data_iter_step: {total_time_str} seconds")

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value), force=True)
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        del loss
        del batch

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(epoch=epoch_f)
        metric_logger.update(lr=lr)
        metric_logger.update(loss=loss_value, **loss_details)

        if (data_iter_step + 1) % accum_iter == 0: # and ((data_iter_step + 1) % (accum_iter * args.print_freq)) == 0:
            loss_value_reduce = misc.all_reduce_mean(loss_value)  # MUST BE EXECUTED BY ALL NODES
            if log_writer is None:
                continue
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            
            epoch_1000x = int(epoch_f * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('train_lr', lr, epoch_1000x)
            log_writer.add_scalar('train_iter', epoch_1000x, epoch_1000x)
            # for name, val in loss_details.items():
            #     log_writer.add_scalar('train_' + name, val, epoch_1000x)
    if epoch == 0 or (epoch + 1) % 10 == 0:
        viz = get_viz(result["view1"], result["view2"], result["pred1"], result["pred2"])
        log_writer.add_figure('train_samples', viz, epoch)
        # print("Added image to log_writer")

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def test_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                   data_loader: Sized, device: torch.device, epoch: int,
                   args, log_writer=None, prefix='test'):

    model.eval()
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.meters = defaultdict(lambda: misc.SmoothedValue(window_size=9**9))
    header = 'Test Epoch: [{}]'.format(epoch)

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    if hasattr(data_loader, 'dataset') and hasattr(data_loader.dataset, 'set_epoch'):
        data_loader.dataset.set_epoch(epoch)
    if hasattr(data_loader, 'sampler') and hasattr(data_loader.sampler, 'set_epoch'):
        data_loader.sampler.set_epoch(epoch)

    for batch_num, batch in enumerate(metric_logger.log_every(data_loader, args.print_freq, header)):
        result = loss_of_one_batch(batch, model, criterion, device,
                                       symmetrize_batch=True,
                                       use_amp=bool(args.amp))
        loss_value, loss_details = result["loss"]  # criterion returns two values
        metric_logger.update(loss=float(loss_value))
        # loss_value, loss_details = loss_tuple  # criterion returns two values
        # metric_logger.update(loss=float(loss_value), **loss_details)
    if (epoch == 0 or (epoch + 1) % 10 == 0):
        viz = get_viz(result["view1"], result["view2"], result["pred1"], result["pred2"])
        log_writer.add_figure('test_samples', viz, epoch)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    results = {f'{k}': getattr(meter, 'global_avg') for k, meter in metric_logger.meters.items()}

    print(results)
    print(results.keys())
    #  'loss_avg': 9177.31206644813, 'loss_med': 7096.86865234375, 'PointfLoss(MSELoss())_avg': 9177.31206644813, 'PointfLoss(MSELoss())_med': 7096.86865234375}
    # [11:10:57.758835] dict_keys(['loss_avg', 'loss_med', 'PointfLoss(MSELoss())_avg', 'PointfLoss(MSELoss())_med'])

    if log_writer is not None:
        for name, val in results.items():
            print(prefix, name, val, 1000*epoch)
            log_writer.add_scalar(prefix + '_' + name, val, 1000*epoch)

    return results
