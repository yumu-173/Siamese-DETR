# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import datetime
import json



import random
import time
from pathlib import Path
import os, sys
from typing import Optional
from util.get_param_dicts import get_param_dict

from util.logger import setup_logger

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist
from models.dino.tracker import Tracker

import datasets
import util.misc as utils
from datasets import build_dataset, get_coco_api_from_dataset
from engine import evaluate, train_one_epoch, test, track_test, ov_test, det_with_gtbox, test_panda
import models
from util.slconfig import DictAction, SLConfig
from util.utils import ModelEma, BestMetricHolder

from bbox_adjust import bbox_adjustment
from tools.tools.coco_categories import fitler_coco_category
from tools.tools.merge_anno_in_coco_lasot_got import build_annotations
from torchreid.utils import FeatureExtractor

def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--config_file', '-c', default='config/DINO/DINO_4scale.py', type=str, required=False)
    parser.add_argument('--options',
        default={'dn_scalar': 100, 'embed_init_tgt': True, 'dn_label_coef': 1.0, 'dn_bbox_coef': 1.0, 'use_ema': False, 'dn_box_noise_scale': 1.0},
        nargs='+',
        # action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file.')

    # dataset parameters
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')
    parser.add_argument('--fix_size', action='store_true')

    # COCO + LaSOT + GOT10K
    parser.add_argument('--coco_path', type=str, default='G:/tracking/GMOT40/COCO/')
    parser.add_argument('--coco_lasot_got_path', type=str, default='G:/tracking/GMOT40/COCO/')
    # parser.add_argument('--o365_path', type=str, default='G:/tracking/GMOT40/COCO/')

    # training parameters
    parser.add_argument('--output_dir', default='logs/DINO/R50-MS4-1',
                        help='path where to save, empty for no saving')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--note', default='',
                        help='add some notes to the experiment')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', help='resume from checkpoint', default=False)
    parser.add_argument('--pretrain_model_path', help='load from other checkpoint', default='')
    # default='D:/giga/detection/DINO/pth/checkpoint0033_4scale.pth'
    parser.add_argument('--finetune_ignore', type=str, nargs='+')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--find_unused_params', action='store_true')
    

    parser.add_argument('--save_results', default=True, action='store_true')
    parser.add_argument('--save_log', default=True, action='store_true')
    
    #new moudle
    parser.add_argument('--attnpool', type=bool, default=False, help='use attention pooling')
    parser.add_argument('--temp_weight', type=bool, default=False, help='share template weight with image')
    parser.add_argument('--denoise_query', type=bool, default=False, help='change denoise query with template')
    parser.add_argument('--box_adjustment', type=bool, default=False, help='adjust fsc box')
    parser.add_argument('--ov_coco', type=bool, default=False, help='remove some class in train')
    parser.add_argument('--keep_template_look', type=bool, default=False, help='Use appearance features as a basis for classification so that the network retains appearance features')
    parser.add_argument('--test_track', default=False, action='store_true', help='test gmot with tracktor')
    parser.add_argument('--test_ov', default=False, action='store_true', help='test ov coco')
    parser.add_argument('--dn_type', default='sample', help='you can chose dsn, dn and no dn')
    parser.add_argument('--number_template', default=2, type=int, help='number of template use in one image')
    parser.add_argument('--temp_in_image', default=False, type=bool, help='find template in current image')
    parser.add_argument('--det_with_gt', default=False, type=bool, help='use gt as query box')
    parser.add_argument('--test_panda', default=False, action='store_true', help='test panda')
    parser.add_argument('--train_with_coco_lasot_got', default=False, action='store_true', help='train with coco, lasot and got')
    parser.add_argument('--train_with_o365', default=False, action='store_true', help='train with objects365.json')
    parser.add_argument('--train_with_od', default=False, action='store_true', help='train with od.json')
    parser.add_argument('--train_with_gmot', default=False, action='store_true', help='train with gmot')
    parser.add_argument('--dn_for_track', default=False, action='store_true', help='set dn scale')


    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--rank', default=0, type=int,
                        help='number of distributed processes')
    parser.add_argument("--local_rank", type=int, help='local rank for DistributedDataParallel')
    parser.add_argument('--amp', action='store_true',
                        help="Train with mixed precision")
    parser.add_argument('--n_nodes', type=int,
                        help='number of distributed nodes')
    
    return parser


def build_model_main(args):
    # we use register to maintain models from catdet6 on.
    from models.registry import MODULE_BUILD_FUNCS
    assert args.modelname in MODULE_BUILD_FUNCS._module_dict
    build_func = MODULE_BUILD_FUNCS.get(args.modelname)
    model, criterion, postprocessors = build_func(args)
    return model, criterion, postprocessors

def update_data_laoder(optimizer):
    dataset_train = build_dataset(image_set='train_adj', args=args)
    if args.distributed:
        sampler_train = DistributedSampler(dataset_train)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn, num_workers=args.num_workers)
    if args.onecyclelr:
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, steps_per_epoch=len(data_loader_train), epochs=args.epochs, pct_start=0.2)
    elif args.multi_step_lr:
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_drop_list)
    else:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    return data_loader_train, lr_scheduler

def main(args):
    utils.init_distributed_mode(args)
    # load cfg file and update the args
    print("Loading config file from {}".format(args.config_file))
    time.sleep(args.rank * 0.02)
    cfg = SLConfig.fromfile(args.config_file)
    if args.options is not None:
        print(args.options)
        cfg.merge_from_dict(args.options)
    if args.rank == 0:
        save_cfg_path = os.path.join(args.output_dir, "config_cfg.py")
        cfg.dump(save_cfg_path)
        save_json_path = os.path.join(args.output_dir, "config_args_raw.json")
        with open(save_json_path, 'w') as f:
            json.dump(vars(args), f, indent=2)
    cfg_dict = cfg._cfg_dict.to_dict()
    args_vars = vars(args)
    for k, v in cfg_dict.items():
        if k not in args_vars:
            setattr(args, k, v)
        else:
            raise ValueError("Key {} can used by args only".format(k))

    # update some new args temporally
    if args.test or args.test_track:
        args.number_template = 1
    if not getattr(args, 'use_ema', None):
        args.use_ema = False
    if not getattr(args, 'debug', None):
        args.debug = False

    # setup logger
    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logger(output=os.path.join(args.output_dir, 'info.txt'), distributed_rank=args.rank, color=False, name="detr")
    logger.info("git:\n  {}\n".format(utils.get_sha()))
    logger.info("Command: "+' '.join(sys.argv))
    if args.rank == 0:
        save_json_path = os.path.join(args.output_dir, "config_args_all.json")
        with open(save_json_path, 'w') as f:
            json.dump(vars(args), f, indent=2)
        logger.info("Full config saved to {}".format(save_json_path))
    logger.info('world size: {}'.format(args.world_size))
    logger.info('rank: {}'.format(args.rank))
    logger.info('local_rank: {}'.format(args.local_rank))
    logger.info("args: " + str(args) + '\n')


    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"
    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # build model
    model, criterion, postprocessors = build_model_main(args)
    wo_class_error = False
    model.to(device)

    # ema
    if args.use_ema:
        ema_m = ModelEma(model, args.ema_decay)
    else:
        ema_m = None

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=args.find_unused_params)
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info('number of params:'+str(n_parameters))
    logger.info("params:\n"+json.dumps({n: p.numel() for n, p in model.named_parameters() if p.requires_grad}, indent=2))

    param_dicts = get_param_dict(args, model_without_ddp)

    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)
    
    if args.train_with_coco_lasot_got:
        dataset_train = build_dataset(image_set='train_coco_lasot_got', args=args)
        dataset_val = build_dataset(image_set='val_coco_lasot_got', args=args)
    elif args.train_with_od:
        dataset_train = build_dataset(image_set='train_od', args=args)
        dataset_val = build_dataset(image_set='val_od', args=args)
    elif args.train_with_gmot:
        print('args.train_with_gmot',args.train_with_gmot)
        dataset_train = build_dataset(image_set='train_gmot', args=args)
        dataset_val = build_dataset(image_set='train_gmot', args=args)
    elif args.train_with_o365:
        dataset_train = build_dataset(image_set='train_o365', args=args)
        dataset_val = build_dataset(image_set='val_o365', args=args)
    elif args.ov_coco:
        dataset_train = build_dataset(image_set='train_ov', args=args)
        dataset_val = build_dataset(image_set='val_ov', args=args)
    elif args.temp_in_image:
        dataset_train = build_dataset(image_set='train_cur', args=args)
        dataset_val = build_dataset(image_set='val', args=args)
    else:
        dataset_train = build_dataset(image_set='train', args=args)
        dataset_val = build_dataset(image_set='val', args=args)
    

    if args.distributed:
        sampler_train = DistributedSampler(dataset_train)
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn, num_workers=args.num_workers)
    data_loader_val = DataLoader(dataset_val, 1, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)

    if args.onecyclelr:
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, steps_per_epoch=len(data_loader_train), epochs=args.epochs, pct_start=0.2)
    elif args.multi_step_lr:
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_drop_list)
    else:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)


    if args.dataset_file == "coco_panoptic":
        # We also evaluate AP during panoptic training, on original coco DS
        coco_val = datasets.coco.build("val", args)
        base_ds = get_coco_api_from_dataset(coco_val)
    else:
        base_ds = get_coco_api_from_dataset(dataset_val)

    if args.frozen_weights is not None:
        checkpoint = torch.load(args.frozen_weights, map_location='cpu')
        model_without_ddp.detr.load_state_dict(checkpoint['model'])

    output_dir = Path(args.output_dir)
    if os.path.exists(os.path.join(args.output_dir, 'checkpoint.pth')):
        args.resume = os.path.join(args.output_dir, 'checkpoint.pth')
        if args.box_adjustment:
            data_loader_train, lr_scheduler = update_data_laoder(optimizer)
            print('update dataloader train')
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        if args.use_ema:
            if 'ema_model' in checkpoint:
                ema_m.module.load_state_dict(utils.clean_state_dict(checkpoint['ema_model']))
            else:
                del ema_m
                ema_m = ModelEma(model, args.ema_decay)                

        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1

    if (not args.resume) and args.pretrain_model_path:
        checkpoint = torch.load(args.pretrain_model_path, map_location='cpu')['model']
        from collections import OrderedDict
        _ignorekeywordlist = args.finetune_ignore if args.finetune_ignore else []
        ignorelist = []

        def check_keep(keyname, ignorekeywordlist):
            for keyword in ignorekeywordlist:
                if keyword in keyname:
                    ignorelist.append(keyname)
                    return False
            return True

        logger.info("Ignore keys: {}".format(json.dumps(ignorelist, indent=2)))
        _tmp_st = OrderedDict({k:v for k, v in utils.clean_state_dict(checkpoint).items() if check_keep(k, _ignorekeywordlist)})

        _load_output = model_without_ddp.load_state_dict(_tmp_st, strict=False)
        logger.info(str(_load_output))

        if args.use_ema:
            if 'ema_model' in checkpoint:
                ema_m.module.load_state_dict(utils.clean_state_dict(checkpoint['ema_model']))
            else:
                del ema_m
                ema_m = ModelEma(model, args.ema_decay)        

    if args.test_ov:
        dataset_test_ov = build_dataset(image_set='test_ov', args=args)
        dataset_test_ov.template_list = dataset_train.template_list
        # import pdb; pdb.set_trace()
        sampler_test_ov = torch.utils.data.RandomSampler(dataset_test_ov)
        batch_sampler_test_ov = torch.utils.data.BatchSampler(sampler_test_ov, args.batch_size, drop_last=True)
        data_loader_test = DataLoader(dataset_test_ov, batch_sampler=batch_sampler_test_ov,
                                    collate_fn=utils.collate_fn, num_workers=args.num_workers)
        test_stats = ov_test(model, criterion, postprocessors, dataset_test_ov, 
                                              data_loader_test, base_ds, device, args.output_dir, wo_class_error=wo_class_error, args=args)
        return
    
    if args.det_with_gt:
        dataset_test = build_dataset(image_set='test', args=args)
        sampler_test = torch.utils.data.RandomSampler(dataset_test)
        batch_sampler_test = torch.utils.data.BatchSampler(sampler_test, args.batch_size, drop_last=True)
        data_loader_test = DataLoader(dataset_test, batch_sampler=batch_sampler_test,
                                    collate_fn=utils.collate_fn, num_workers=args.num_workers)
        test_stats = det_with_gtbox(model, criterion, postprocessors,
                                              data_loader_test, base_ds, device, args.output_dir, wo_class_error=wo_class_error, args=args)
        return
    
    if args.test:
        dataset_test = build_dataset(image_set='test', args=args)
        sampler_test = torch.utils.data.RandomSampler(dataset_test)
        batch_sampler_test = torch.utils.data.BatchSampler(sampler_test, args.batch_size, drop_last=True)
        data_loader_test = DataLoader(dataset_test, batch_sampler=batch_sampler_test,
                                    collate_fn=utils.collate_fn, num_workers=args.num_workers)
        test_stats = test(model, criterion, postprocessors,
                                              data_loader_test, base_ds, device, args.output_dir, wo_class_error=wo_class_error, args=args)
        return
    
    if args.test_panda:
        start = time.time()
        dataset_test = build_dataset(image_set='panda', args=args)
        sampler_test = torch.utils.data.RandomSampler(dataset_test)
        batch_sampler_test = torch.utils.data.BatchSampler(sampler_test, args.batch_size, drop_last=True)
        data_loader_test = DataLoader(dataset_test, batch_sampler=batch_sampler_test,
                                    collate_fn=utils.collate_fn, num_workers=args.num_workers)
        test_stats = test_panda(model, criterion, postprocessors,
                                              data_loader_test, base_ds, device, args.output_dir, wo_class_error=wo_class_error, args=args)
        end = time.time()
        print('every image cost {}s'.format((end - start)))
        return
    
    if args.test_track:
        reid_model = Path('ckpts/osnet_ain_x1_0_msmt17.pth')
        assert os.path.isfile(reid_model)
        reid_network = FeatureExtractor(
            model_name='osnet_ain_x1_0',
            model_path=reid_model,
            verbose=False,
            device='cuda' if torch.cuda.is_available() else 'cpu')

        tracker = Tracker(model, reid_network, args)
        dataset_test = build_dataset(image_set='test_track', args=args)
        # sampler_test = torch.utils.data.RandomSampler(dataset_test)
        # batch_sampler_test = torch.utils.data.BatchSampler(sampler_test, args.batch_size, drop_last=True)
        # data_loader_test = DataLoader(dataset_test, batch_sampler=batch_sampler_test,
        #                             collate_fn=utils.collate_fn, num_workers=args.num_workers)
        # import pdb; pdb.set_trace()
        test_stats = track_test(model, criterion, postprocessors,
                                              dataset_test, base_ds, device, args.output_dir, tracker, wo_class_error=wo_class_error, args=args)
        return
    
    if args.eval:
        os.environ['EVAL_FLAG'] = 'TRUE'
        test_stats, coco_evaluator = evaluate(model, criterion, postprocessors,
                                              data_loader_val, base_ds, device, args.output_dir, wo_class_error=wo_class_error, args=args)
        if args.output_dir:
            utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, output_dir / "eval.pth")

        log_stats = {**{f'test_{k}': v for k, v in test_stats.items()} }
        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

        return

    print("Start training")
    start_time = time.time()
    best_map_holder = BestMetricHolder(use_ema=args.use_ema)
    print(args.epochs)
    for epoch in range(args.start_epoch, args.epochs):
        print('epoch:', epoch)
        epoch_start_time = time.time()
        if args.distributed:
            sampler_train.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch, postprocessors,
            args.clip_max_norm, wo_class_error=wo_class_error, lr_scheduler=lr_scheduler, args=args, logger=(logger if args.save_log else None), ema_m=ema_m)
        
        print('train')
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']

        if not args.onecyclelr:
            lr_scheduler.step()
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            # extra checkpoint before LR drop and every 100 epochs
            if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % args.save_checkpoint_interval == 0:
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
            for checkpoint_path in checkpoint_paths:
                weights = {
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }
                if args.use_ema:
                    weights.update({
                        'ema_model': ema_m.module.state_dict(),
                    })
                utils.save_on_master(weights, checkpoint_path)
        if args.box_adjustment:
            test_res = test(model, criterion, postprocessors,
                                                data_loader_train, base_ds, device, args.output_dir, wo_class_error=wo_class_error, args=args)
            # online adjust gt--------------------------------------------------------------------------------------
            bbox_adjustment(test_res, args.coco_path)
            # ------------------------------------------------------------------------------------------------------
#             print('Before adj Process Barrier')
#             torch.distributed.barrier()
#             print('After adj Process Barrier')
            data_loader_train, lr_scheduler = update_data_laoder(optimizer)
                
        # eval
        test_stats, coco_evaluator = evaluate(
            model, criterion, postprocessors, data_loader_val, base_ds, device, args.output_dir,
            wo_class_error=wo_class_error, args=args, logger=(logger if args.save_log else None)
        )
        map_regular = test_stats['coco_eval_bbox'][0]
        _isbest = best_map_holder.update(map_regular, epoch, is_ema=False)
        if _isbest:
            checkpoint_path = output_dir / 'checkpoint_best_regular.pth'
            utils.save_on_master({
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'args': args,
            }, checkpoint_path)
        log_stats = {
            **{f'train_{k}': v for k, v in train_stats.items()},
            # **{f'test_{k}': v for k, v in test_stats.items()},
        }

        # eval ema
        if args.use_ema:
            ema_test_stats, ema_coco_evaluator = evaluate(
                ema_m.module, criterion, postprocessors, data_loader_val, base_ds, device, args.output_dir,
                wo_class_error=wo_class_error, args=args, logger=(logger if args.save_log else None)
            )
            log_stats.update({f'ema_test_{k}': v for k,v in ema_test_stats.items()})
            map_ema = ema_test_stats['coco_eval_bbox'][0]
            _isbest = best_map_holder.update(map_ema, epoch, is_ema=True)
            if _isbest:
                checkpoint_path = output_dir / 'checkpoint_best_ema.pth'
                utils.save_on_master({
                    'model': ema_m.module.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)
        log_stats.update(best_map_holder.summary())

        ep_paras = {
                'epoch': epoch,
                'n_parameters': n_parameters
            }
        log_stats.update(ep_paras)
        try:
            log_stats.update({'now_time': str(datetime.datetime.now())})
        except:
            pass
        
        epoch_time = time.time() - epoch_start_time
        epoch_time_str = str(datetime.timedelta(seconds=int(epoch_time)))
        log_stats['epoch_time'] = epoch_time_str

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

            # for evaluation logs
            if coco_evaluator is not None:
                (output_dir / 'eval').mkdir(exist_ok=True)
                if "bbox" in coco_evaluator.coco_eval:
                    filenames = ['latest.pth']
                    if epoch % 50 == 0:
                        filenames.append(f'{epoch:03}.pth')
                    for name in filenames:
                        torch.save(coco_evaluator.coco_eval["bbox"].eval,
                                   output_dir / "eval" / name)
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

    # remove the copied files.
    copyfilelist = vars(args).get('copyfilelist')
    if copyfilelist and args.local_rank == 0:
        from datasets.data_util import remove
        for filename in copyfilelist:
            print("Removing: {}".format(filename))
            remove(filename)



if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    # args.config_file = 'config/DINO/DINO_4scale.py'
    # args.output_dir = 'logs/DINO/R50-MS4-%j'
    # args.coco_path = 'D:/giga/detection/gigadataset/image_train/image_train/01_University_Canteen'
    # args.resume = 'D:/giga/detection/DINO/pth/checkpoint0033_4scale.pth'
    # args.eval = True
    # args.options = 'dn_scalar=100 embed_init_tgt=TRUE dn_label_coef=1.0 dn_bbox_coef=1.0 use_ema=False dn_box_noise_scale=1.0'
    print(args)
    if args.ov_coco:
        fitler_coco_category(args.coco_path, 'train')
        fitler_coco_category(args.coco_path, 'val')
    # if args.train_with_coco_lasot_got and args.rank==0:
    #     print('train_with_coco_lasot_got')
    #     build_annotations(args.coco_lasot_got_path)
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
