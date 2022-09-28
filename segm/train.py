import sys
from pathlib import Path
import yaml
import json
import numpy as np
import torch
import click
import argparse
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn as nn
import torch.distributed
import pickle
from prepare.cut_concat_image import process_image_split
from segm.utils import distributed
import segm.utils.torch as ptu
from segm import config
from segm.model.factory import create_segmenter
from segm.optim.factory import create_optimizer, create_scheduler
from segm.data.factory import create_dataset
from segm.model.utils import num_params

from timm.utils import NativeScaler
from contextlib import suppress
import os

from segm.utils.distributed import sync_model
from segm.engine import train_one_epoch, evaluate
import collections

from prepare import convert_spherically_out_image_and_reverse
from prepare import main_output_index_and_image_once
from prepare import resize_image
from prepare import rgb_to_gray


@click.command(help="")
@click.option("--log-dir", type=str, help="logging directory")
@click.option("--dataset", default='coco', type=str)
@click.option('--dataset_dir', default='', type=str)
@click.option("--im_size", default=16 * 10, type=int,
              help="dataset resize size")  # 256 patch size==16 have to be n*16
@click.option("--crop-size", default=16 * 10, type=int)  # 256 == 16*16
@click.option("--window-size", default=16 * 10, type=int)  # 256 == 16*16
@click.option("--window-stride", default=None, type=int)
@click.option("--backbone", default="vit_large_patch16_384",
              type=str)  # try this, and freeze first several blocks.
@click.option("--decoder", default="mask_transformer", type=str)
@click.option("--optimizer", default="sgd", type=str)
@click.option("--scheduler", default="polynomial", type=str)
@click.option("--weight-decay", default=0.0, type=float)
@click.option("--dropout", default=0.0, type=float)
@click.option("--drop-path", default=0.1, type=float)
@click.option("--batch-size", default=None, type=int)
@click.option("--epochs", default=None, type=int)
@click.option("-lr", "--learning-rate", default=None, type=float)
@click.option("--normalization", default=None, type=str)
@click.option("--eval-freq", default=None, type=int)
@click.option("--amp/--no-amp", default=False, is_flag=True)
@click.option("--resume/--no-resume", default=True, is_flag=True)
@click.option('--local_rank', type=int)
@click.option('--only_test', type=bool, default=False)
@click.option('--add_mask', type=bool, default=False)  # valid original: True
@click.option('--partial_finetune', type=bool,
              default=False)  # compare validation, last finetune all blocks.
@click.option('--add_l1_loss', type=bool,
              default=True)  # add after classification.
@click.option('--l1_weight', type=float, default=10)
@click.option('--color_position', type=bool,
              default=True)  # add color position in color token.
@click.option('--change_mask', type=bool,
              default=False)  # change mask, omit the attention between color tokens.
@click.option('--color_as_condition', type=bool,
              default=False)  # use self-attn to embedding color tokens, and use color to represent patch tokens.
@click.option('--multi_scaled', type=bool,
              default=False)  # multi-scaled decoder.
@click.option('--downchannel', type=bool,
              default=False)  # multi-scaled, upsample+downchannel. (should be correct??)
@click.option('--add_conv', type=bool,
              default=True)  # add conv after transformer blocks.
@click.option('--before_classify', type=bool,
              default=False)  # classification at 16x16 resolution, and use CNN upsampler for 256x256, then use l1-loss.
@click.option('--l1_conv', type=bool,
              default=True)  # patch--upsample--> [B, 256x256, 180]--conv3x3-> [B, 256x256, 2]
@click.option('--l1_linear', type=bool,
              default=False)  # pacth: [B, 16x16, 180]---linear-> [B, 16x16, 2x16x16]
@click.option('--add_fm', type=bool,
              default=False)  # add Feature matching loss.
@click.option('--fm_weight', type=float, default=1)
@click.option('--add_edge', type=bool,
              default=False)  # add sobel-conv to extract edge.
@click.option('--edge_loss_weight', type=float,
              default=0.05)  # edge_loss_weight.
@click.option('--mask_l_num', type=int,
              default=4)  # mask for L ranges: 4, 10, 20, 50, 100
@click.option('--n_blocks', type=int,
              default=1)  # per block have 2 layers. block_num = 2
@click.option('--n_layers', type=int, default=2)
@click.option('--without_colorattn', type=bool, default=False)
@click.option('--without_colorquery', type=bool, default=False)
@click.option('--without_classification', type=bool, default=False)
@click.option('--color_token_num', type=int, default=313)
@click.option('--sin_color_pos', type=bool, default=False)
@click.option('--need_gpus', type=int, default=0)
@click.option('--save_ssd', type=bool, default=True)
@click.option("--origin_image_path", type=str, default='', help="")
@click.option("--gray_image_dir", type=str, default='', help="")
@click.option("--origin_gray_image_dir", type=str, default='', help="")
def main(
        log_dir,
        dataset,
        dataset_dir,
        im_size,
        crop_size,
        window_size,
        window_stride,
        backbone,
        decoder,
        optimizer,
        scheduler,
        weight_decay,
        dropout,
        drop_path,
        batch_size,
        epochs,
        learning_rate,
        normalization,
        eval_freq,
        amp,
        resume,
        local_rank,
        only_test,
        add_mask,
        partial_finetune,
        add_l1_loss,
        l1_weight,
        color_position,
        change_mask,
        color_as_condition,
        multi_scaled,
        downchannel,
        add_conv,
        before_classify,
        l1_conv,
        l1_linear,
        add_fm,
        fm_weight,
        add_edge,
        edge_loss_weight,
        mask_l_num,
        n_blocks,
        n_layers,
        without_colorattn,
        without_colorquery,
        without_classification,
        color_token_num,
        sin_color_pos,
        need_gpus,
        save_ssd,
        origin_image_path,
        gray_image_dir,
        origin_gray_image_dir
):
    # check if required folders exist. if not create.
    print('folders are checked')

    if not os.path.exists(dataset_dir):
        os.mkdir(dataset_dir)
    train_dir = dataset_dir + "/train/"
    if not os.path.exists(train_dir):
        os.mkdir(train_dir)
    train_image_dir = train_dir + "/image/"
    # process_image_split(origin_image_path, train_image_dir, 185)
    # process_image_split(origin_image_path, train_image_dir, 370)
    # process_image_split(origin_gray_image_dir, gray_image_dir, 185)
    # process_image_split(origin_gray_image_dir, gray_image_dir, 370)
    # start distributed mode
    # os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    ptu.set_gpu_mode(True, local_rank,need_gpus)
    # ptu.set_gpu_mode(True, local_rank, need_gpus)
    # distributed.init_process()
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12345'

    torch.distributed.init_process_group(backend="gloo", init_method='env://',
                                         rank=local_rank,
                                         world_size=ptu.world_size)

    # set up configuration
    cfg = config.load_config()
    model_cfg = cfg["model"][backbone]
    dataset_cfg = cfg["dataset"][dataset]
    if "mask_transformer" in decoder:
        decoder_cfg = cfg["decoder"]["mask_transformer"]
    else:
        decoder_cfg = cfg["decoder"][decoder]

    # model config
    if not im_size:
        im_size = dataset_cfg["im_size"]  # 256
    if not crop_size:
        crop_size = dataset_cfg.get("crop_size", im_size)  # 256
    if not window_size:
        window_size = dataset_cfg.get("window_size", im_size)
    if not window_stride:
        window_stride = dataset_cfg.get("window_stride", im_size)
    if not dataset_dir:
        dataset_dir = dataset_cfg.get('dataset_dir', None)

    model_cfg["image_size"] = (crop_size, crop_size)
    model_cfg["backbone"] = backbone
    model_cfg["dropout"] = dropout  # 0
    model_cfg["drop_path_rate"] = drop_path  # 0.1
    decoder_cfg["name"] = decoder
    model_cfg["decoder"] = decoder_cfg

    # dataset config
    world_batch_size = dataset_cfg["batch_size"]
    num_epochs = dataset_cfg["epochs"]
    lr = dataset_cfg["learning_rate"]

    if batch_size:
        world_batch_size = batch_size
    if epochs:
        num_epochs = epochs
    if learning_rate:
        lr = learning_rate
    if eval_freq is None:
        eval_freq = dataset_cfg.get("eval_freq", 1)

    if normalization:
        model_cfg["normalization"] = normalization

    # experiment config
    # print('ptu.world_size', ptu.world_size)
    batch_size = world_batch_size // ptu.world_size
    # print('bs', batch_size)
    variant = dict(
        world_batch_size=world_batch_size,
        version="normal",
        resume=resume,
        dataset_kwargs=dict(
            dataset=dataset,
            image_size=im_size,
            crop_size=crop_size,
            batch_size=batch_size,
            normalization=model_cfg["normalization"],
            split="train",
            num_workers=10,
            dataset_dir=dataset_dir,
            gray_image_dir=gray_image_dir,
            add_mask=add_mask,
            patch_size=model_cfg["patch_size"],
            change_mask=change_mask,
            multi_scaled=multi_scaled,
            mask_num=mask_l_num,
            n_cls=color_token_num,
        ),
        algorithm_kwargs=dict(
            batch_size=batch_size,
            start_epoch=0,
            num_epochs=num_epochs,
            eval_freq=eval_freq,
        ),
        optimizer_kwargs=dict(
            opt=optimizer,
            lr=lr,
            weight_decay=weight_decay,
            momentum=0.9,
            clip_grad=None,
            sched=scheduler,
            epochs=num_epochs,
            min_lr=1e-5,
            poly_power=0.9,
            poly_step_size=1,
        ),
        net_kwargs=model_cfg,
        amp=amp,
        log_dir=log_dir,
        inference_kwargs=dict(
            im_size=im_size,
            window_size=window_size,
            window_stride=window_stride,
        ),
    )

    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # checkpoint_path = log_dir / 'checkpoint.pth'  # tiny.

    # dataset
    dataset_kwargs = variant["dataset_kwargs"]

    train_loader = create_dataset(dataset_kwargs)
    # model
    net_kwargs = variant["net_kwargs"]
    net_kwargs["n_cls"] = color_token_num
    net_kwargs['partial_finetune'] = partial_finetune
    net_kwargs['decoder']['add_l1_loss'] = add_l1_loss
    net_kwargs['decoder']['color_position'] = color_position
    net_kwargs['decoder']['change_mask'] = change_mask
    net_kwargs['decoder']['color_as_condition'] = color_as_condition
    net_kwargs['decoder']['multi_scaled'] = multi_scaled
    net_kwargs['decoder']['crop_size'] = crop_size
    net_kwargs['decoder']['downchannel'] = downchannel
    net_kwargs['decoder']['add_conv'] = add_conv
    net_kwargs['decoder']['before_classify'] = before_classify
    net_kwargs['decoder']['l1_conv'] = l1_conv
    net_kwargs['decoder']['l1_linear'] = l1_linear
    net_kwargs['decoder']['add_edge'] = add_edge
    net_kwargs['decoder']['n_blocks'] = n_blocks
    net_kwargs['decoder']['n_layers'] = n_layers
    net_kwargs['decoder']['without_colorattn'] = without_colorattn
    net_kwargs['decoder']['without_colorquery'] = without_colorquery
    net_kwargs['decoder']['without_classification'] = without_classification
    net_kwargs['decoder']['sin_color_pos'] = sin_color_pos
    model = create_segmenter(net_kwargs)
    model.to(ptu.device)

    # optimizer
    optimizer_kwargs = variant["optimizer_kwargs"]
    optimizer_kwargs["iter_max"] = len(train_loader) * optimizer_kwargs[
        "epochs"]
    optimizer_kwargs["iter_warmup"] = 0.0
    opt_args = argparse.Namespace()
    opt_vars = vars(opt_args)
    for k, v in optimizer_kwargs.items():
        opt_vars[k] = v
    optimizer = create_optimizer(opt_args, model, partial_finetune)
    lr_scheduler = create_scheduler(opt_args, optimizer)
    num_iterations = 0
    amp_autocast = suppress
    loss_scaler = None
    #  autocast + gradscaler
    if amp:
        amp_autocast = torch.cuda.amp.autocast
        loss_scaler = NativeScaler()

    # resume
    if resume:
        checkpoint_files = os.listdir(log_dir)
        checkpoint_exist = False
        checkpoint_path_file = None
        for checkpoint_file_ in checkpoint_files:
            i = 0
            _, ext = os.path.splitext(checkpoint_file_)
            if ext == '.pth':
                i += 1
                if i > 1:
                    if checkpoint_path_file < checkpoint_file_:
                        checkpoint_path_file = checkpoint_file_
                else:
                    checkpoint_path_file = checkpoint_file_
                checkpoint_path = os.path.join(log_dir, checkpoint_path_file)
            if ext == ".pkl":
                i += 1
                if i > 1:
                    if checkpoint_path_file < checkpoint_file_:
                        checkpoint_path_file = checkpoint_file_
                else:
                    checkpoint_path_file = checkpoint_file_
                checkpoint_path = os.path.join(log_dir, checkpoint_path_file)
            if i:
                checkpoint_exist = True

    if resume and checkpoint_exist:
        print(f"Resuming training from checkpoint: {checkpoint_path}")
        with open(checkpoint_path, 'rb') as f:
            net_state_dict = pickle.load(f)
        checkpoint = net_state_dict
        # checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint["model"])  # for pos encoding
        optimizer.load_state_dict(checkpoint["optimizer"])
        if loss_scaler and "loss_scaler" in checkpoint:
            loss_scaler.load_state_dict(checkpoint["loss_scaler"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        variant["algorithm_kwargs"]["start_epoch"] = checkpoint["epoch"] + 1

    if ptu.distributed:
        print('Distributed:', ptu.distributed)
        model = DDP(model, device_ids=[ptu.device], find_unused_parameters=True)

    # save config
    variant_str = yaml.dump(variant)
    print(f"Configuration:\n{variant_str}")
    variant["net_kwargs"] = net_kwargs
    variant["dataset_kwargs"] = dataset_kwargs
    log_dir.mkdir(parents=True, exist_ok=True)
    with open(log_dir / "variant.yml", "w") as f:
        f.write(variant_str)

    # train
    start_epoch = variant["algorithm_kwargs"]["start_epoch"]
    num_epochs = variant["algorithm_kwargs"]["num_epochs"]
    # eval_freq = variant["algorithm_kwargs"]["eval_freq"]

    model_without_ddp = model
    if hasattr(model, "module"):
        model_without_ddp = model.module

    # val_seg_gt = val_loader.dataset.get_gt_seg_maps()

    print(f"Train dataset length: {len(train_loader.dataset)}")
    # print(f"Val dataset length: {len(val_loader.dataset)}")
    print(f"Encoder parameters: {num_params(model_without_ddp.encoder)}")
    print(f"Decoder parameters: {num_params(model_without_ddp.decoder)}")

    for epoch in range(start_epoch, num_epochs):
        torch.cuda.empty_cache()
        # train for one epoch
        print('Training...', epoch)
        train_logger = train_one_epoch(
            model,
            train_loader,
            optimizer,
            lr_scheduler,
            epoch,
            amp_autocast,
            loss_scaler,
            add_mask,
            add_l1_loss,
            l1_weight,
            partial_finetune,
            l1_conv,
            l1_linear,
            add_edge,
            edge_loss_weight,
            without_classification,
            log_dir,
        )

        print('Epoch: [{}] loss: {}'.format(epoch, train_logger.loss))
        # # # save checkpoint
        if ptu.dist_rank == 0:
            snapshot = dict(
                model=model_without_ddp.state_dict(),
                optimizer=optimizer.state_dict(),
                n_cls=model_without_ddp.n_cls,
                lr_scheduler=lr_scheduler.state_dict(),
            )
            if loss_scaler is not None:
                snapshot["loss_scaler"] = loss_scaler.state_dict()
            snapshot["epoch"] = epoch
            save_path = os.path.join(log_dir, 'checkpoint_epoch_%d.pth' % (epoch))
            save_path_pickle = os.path.join(log_dir, 'checkpoint_epoch_%d.pkl' % (epoch))

            '''
            in order to save ssd life, if --save_ssd == True, 
            only save every 10th model starting from the first checkpoint, 
            the last one will be saved
            '''
            if save_ssd and epoch % 1 == 0:
                # with open(save_path_pickle, "wb") as f:
                #     pickle.dump(snapshot, f)
                torch.save(snapshot, save_path)
                print('save ssd mode, '
                      'save every 10th model starting from the first checkpoint,'
                      'save model into:',
                      save_path_pickle)
            elif save_ssd and epoch == (num_epochs - 1):
                torch.save(snapshot, save_path)
                print('save ssd mode, save the last model into:', save_path)
            elif not save_ssd:
                torch.save(snapshot, save_path)
                print('save model into:', save_path)

        '''
        in order to save space, only save 5 check points
        '''
        checkpoint_files = os.listdir(log_dir)
        num_checkpoint = 0
        for checkpoint_file_ in checkpoint_files:
            _, ext = os.path.splitext(checkpoint_file_)
            if ext == '.pth':
                num_checkpoint += 1

        if num_checkpoint > 4:
            for checkpoint_file_ in checkpoint_files:
                i = 0
                _, ext = os.path.splitext(checkpoint_file_)
                if ext == '.pth':
                    i += 1
                    if i > 1:
                        if delete_checkpoint_path_file > checkpoint_file_:
                            delete_checkpoint_path_file = checkpoint_file_
                    else:
                        delete_checkpoint_path_file = checkpoint_file_

            print('only need to save the recent 5 check points')
            delete_path_with_file = os.path.join(log_dir,
                                                 delete_checkpoint_path_file)
            # delete specified check point
            os.remove(delete_path_with_file)
            print('deleted out of date check point:', delete_path_with_file)

    distributed.barrier()
    distributed.destroy_process()
    # sys.exit(1)


if __name__ == "__main__":
    main()
