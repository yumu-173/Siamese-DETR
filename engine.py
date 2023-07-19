# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
"""

import math
import os
import sys
from typing import Iterable
import torchvision
from util.utils import slprint, to_device
from util.nms_utils import cpu_nms, set_cpu_nms
import numpy as np
import torch
from torch.utils.data import DataLoader
import time
import tqdm
import cv2
from collections import Counter
import matplotlib.pyplot as plt
import datasets.transforms as T

import util.misc as utils
from datasets.coco_eval import CocoEvaluator
from datasets.panoptic_eval import PanopticEvaluator
from pathlib import Path

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, postprocessors,
                    max_norm: float = 0, 
                    wo_class_error=False, lr_scheduler=None, args=None, logger=None, ema_m=None):
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    try:
        need_tgt_for_training = args.use_dn
    except:
        need_tgt_for_training = False

    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    if not wo_class_error:
        metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 100

    _cnt = 0
    template_class_list = []
    for samples, targets, templates, num_temp, temp_cls in metric_logger.log_every(data_loader, print_freq, header, logger=logger):
        # template_class_list.extend(temp_cls)
        # # print(template_class_list)
        # continue

        samples = samples.to(device)
        merge_targets = []
        num_temp = min(num_temp)
        for target_list in targets:
            merge_targets.extend(target_list[:num_temp])
        targets = merge_targets 
        # import pdb; pdb.set_trace()
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        # import pdb; pdb.set_trace()
        # print('enter', targets[0]['boxes'].shape)

        with torch.cuda.amp.autocast(enabled=args.amp):
            if need_tgt_for_training:
                # ------------------------------------------------------------------------------------------------------
                # print('targets:', len(targets))
                # import pdb; pdb.set_trace()
                # print('poss:', temp_pos)
                # ------------------------------------------------------------------------------------------------------
                outputs, targets = model(samples, templates, targets, num_temp=num_temp)
            else:
                outputs, _ = model(samples)
            # import pdb; pdb.set_trace()
            # print('loss', targets[0]['boxes'].shape)
            loss_dict, targets = criterion(outputs, targets)
            weight_dict = criterion.weight_dict
            # import ipdb; ipdb.set_trace()
            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)


        # amp backward function
        if args.amp:
            optimizer.zero_grad()
            scaler.scale(losses).backward()
            if max_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            # original backward function
            optimizer.zero_grad()
            losses.backward()
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()

        if args.onecyclelr:
            lr_scheduler.step()
        if args.use_ema:
            if epoch >= args.ema_epoch:
                ema_m.update(model)

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        if 'class_error' in loss_dict_reduced:
            metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        _cnt += 1
        if args.debug:
            if _cnt % 15 == 0:
                print("BREAK!"*5)
                break

    # counter = Counter(template_class_list)
    # # draw class hist in coco
    # x_value = [int(x) for x in counter.keys()]
    # y_value = [counter[x]/len(template_class_list) for x in counter.keys()]
    # plt.bar(x_value, y_value, width=0.8, bottom=None)
    # plt.savefig('template/template_class_hist.png')
    # exit(0)

    if getattr(criterion, 'loss_weight_decay', False):
        criterion.loss_weight_decay(epoch=epoch)
    if getattr(criterion, 'tuning_matching', False):
        criterion.tuning_matching(epoch)


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    resstat = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}
    if getattr(criterion, 'loss_weight_decay', False):
        resstat.update({f'weight_{k}': v for k,v in criterion.weight_dict.items()})
    return resstat


@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, base_ds, device, output_dir, wo_class_error=False, args=None, logger=None):
    try:
        need_tgt_for_training = args.use_dn
    except:
        need_tgt_for_training = False

    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    if not wo_class_error:
        metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    useCats = True
    try:
        useCats = args.useCats
    except:
        useCats = True
    if not useCats:
        print("useCats: {} !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!".format(useCats))
    coco_evaluator = CocoEvaluator(base_ds, iou_types, useCats=useCats)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    panoptic_evaluator = None
    if 'panoptic' in postprocessors.keys():
        panoptic_evaluator = PanopticEvaluator(
            data_loader.dataset.ann_file,
            data_loader.dataset.ann_folder,
            output_dir=os.path.join(output_dir, "panoptic_eval"),
        )

    _cnt = 0
    output_state_dict = {} # for debug only
    for samples, targets, templates, num_temp, temp_cls in metric_logger.log_every(data_loader, 100, header, logger=logger):
        samples = samples.to(device)
        # import ipdb; ipdb.set_trace()
        merge_targets = []
        for target_list in targets:
            merge_targets.extend(target_list)
        targets = merge_targets
        targets = [{k: to_device(v, device) for k, v in t.items()} for t in targets]
        num_temp = min(num_temp)
        # import pdb; pdb.set_trace()

        with torch.cuda.amp.autocast(enabled=args.amp):
            if need_tgt_for_training:
                # import pdb; pdb.set_trace()
                outputs, targets = model(samples, templates, targets, num_temp=num_temp)
            else:
                outputs, _ = model(samples, templates, num_temp=num_temp)
            # outputs = model(samples)

            loss_dict, targets = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)
        if 'class_error' in loss_dict_reduced:
            metric_logger.update(class_error=loss_dict_reduced['class_error'])
        # import pdb; pdb.set_trace()
        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        # import pdb; pdb.set_trace()
        results = postprocessors['bbox'](outputs, orig_target_sizes)
        # [scores: [100], labels: [100], boxes: [100, 4]] x B
        if 'segm' in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        # import ipdb; ipdb.set_trace()
        if coco_evaluator is not None:
            coco_evaluator.update(res)

        if panoptic_evaluator is not None:
            res_pano = postprocessors["panoptic"](outputs, target_sizes, orig_target_sizes)
            for i, target in enumerate(targets):
                image_id = target["image_id"].item()
                file_name = f"{image_id:012d}.png"
                res_pano[i]["image_id"] = image_id
                res_pano[i]["file_name"] = file_name

            panoptic_evaluator.update(res_pano)
        
        if args.save_results:
            # res_outputs['res_score']
            # res_label = outputs['res_label']
            # res_bbox = outputs['res_bbox']
            # res_idx = outputs['res_idx']
            # import ipdb; ipdb.set_trace()

            for i, (tgt, res, outbbox) in enumerate(zip(targets, results, outputs['pred_boxes'])):
                """
                pred vars:
                    K: number of bbox pred
                    score: Tensor(K),
                    label: list(len: K),
                    bbox: Tensor(K, 4)
                    idx: list(len: K)
                tgt: dict.

                """
                # compare gt and res (after postprocess)
                gt_bbox = tgt['boxes']
                gt_label = tgt['labels']
                gt_info = torch.cat((gt_bbox, gt_label.unsqueeze(-1)), 1)
                
                # img_h, img_w = tgt['orig_size'].unbind()
                # scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=0)
                # _res_bbox = res['boxes'] / scale_fct
                _res_bbox = outbbox
                _res_prob = res['scores']
                _res_label = res['labels']
                # print('_res_bbox:', _res_bbox.shape)
                res_info = torch.cat((_res_bbox, _res_prob.unsqueeze(-1), _res_label.unsqueeze(-1)), 1)
                # import ipdb;ipdb.set_trace()

                if 'gt_info' not in output_state_dict:
                    output_state_dict['gt_info'] = []
                output_state_dict['gt_info'].append(gt_info.cpu())

                if 'res_info' not in output_state_dict:
                    output_state_dict['res_info'] = []
                output_state_dict['res_info'].append(res_info.cpu())

            # # for debug only
            # import random
            # if random.random() > 0.7:
            #     print("Now let's break")
            #     break

        _cnt += 1
        if args.debug:
            if _cnt % 15 == 0:
                print("BREAK!"*5)
                break

    if args.save_results:
        import os.path as osp
        
        # output_state_dict['gt_info'] = torch.cat(output_state_dict['gt_info'])
        # output_state_dict['res_info'] = torch.cat(output_state_dict['res_info'])
        savepath = osp.join(args.output_dir, 'results-{}.pkl'.format(utils.get_rank()))
        print("Saving res to {}".format(savepath))
        torch.save(output_state_dict, savepath)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
    if panoptic_evaluator is not None:
        panoptic_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
        
    panoptic_res = None
    if panoptic_evaluator is not None:
        panoptic_res = panoptic_evaluator.summarize()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}
    if coco_evaluator is not None:
        if 'bbox' in postprocessors.keys():
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in postprocessors.keys():
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
    if panoptic_res is not None:
        stats['PQ_all'] = panoptic_res["All"]
        stats['PQ_th'] = panoptic_res["Things"]
        stats['PQ_st'] = panoptic_res["Stuff"]

    # import ipdb; ipdb.set_trace()

    return stats, coco_evaluator


@torch.no_grad()
def test(model, criterion, postprocessors, data_loader, base_ds, device, output_dir, wo_class_error=False, args=None, logger=None):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    # if not wo_class_error:
    #     metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    # coco_evaluator = CocoEvaluator(base_ds, iou_types)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    panoptic_evaluator = None
    if 'panoptic' in postprocessors.keys():
        panoptic_evaluator = PanopticEvaluator(
            data_loader.dataset.ann_file,
            data_loader.dataset.ann_folder,
            output_dir=os.path.join(output_dir, "panoptic_eval"),
        )

    final_res = []
    template_box = {}
    for samples, targets, templates in metric_logger.log_every(data_loader, 10, header, logger=logger):
        samples = samples.to(device)
        # print('sample:', samples.shape)
        # import pdb; pdb.set_trace()
        # targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        targets = [{k: to_device(v, device) for k, v in t.items()} for t in targets]

        outputs, _ = model(samples, templates)
        # loss_dict = criterion(outputs, targets)
        # weight_dict = criterion.weight_dict

        # # reduce losses over all GPUs for logging purposes
        # loss_dict_reduced = utils.reduce_dict(loss_dict)
        # loss_dict_reduced_scaled = {k: v * weight_dict[k]
        #                             for k, v in loss_dict_reduced.items() if k in weight_dict}
        # loss_dict_reduced_unscaled = {f'{k}_unscaled': v
        #                               for k, v in loss_dict_reduced.items()}
        # metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
        #                      **loss_dict_reduced_scaled,
        #                      **loss_dict_reduced_unscaled)
        # if 'class_error' in loss_dict_reduced:
        #     metric_logger.update(class_error=loss_dict_reduced['class_error'])

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes, not_to_xyxy=True)
        # [scores: [100], labels: [100], boxes: [100, 4]] x B
        if 'segm' in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}

        for image_id, outputs in res.items():

            # template_box[int(image_id)] = temp_pos
            # _scores = outputs['scores'].tolist()
            # _labels = outputs['labels'].tolist()
            # _boxes = outputs['boxes'].tolist()
            _scores = outputs['scores']
            _labels = outputs['labels']
            _boxes = outputs['boxes']
            # ------------------ NMS -----------------------
            # print('before:', _boxes[0])
            box = torch.zeros_like(_boxes)
            box[:, :2] = _boxes[:, :2] - (_boxes[:, 2:] / 2)
            box[:, 2:] = _boxes[:, :2] + (_boxes[:, 2:] / 2)
            # print('after:', box[0])
            # print(_boxes)
            # print('boxes:', boxes)
            keep = torchvision.ops.nms(box, _scores, 0.9)
            # print('keep:', _boxes)
            _boxes = _boxes[keep].tolist()
            # print(_boxes)
            # _boxes.tolist()
            _labels = _labels[keep].tolist()
            _scores = _scores[keep].tolist()
            # ----------------------------------------------
            for s, l, b in zip(_scores, _labels, _boxes):
                assert isinstance(l, int)
                itemdict = {
                        "image_id": int(image_id), 
                        "category_id": l, 
                        "bbox": b, 
                        "score": s,
                        }
                final_res.append(itemdict)

    if args.output_dir:
        import json
        with open(args.output_dir + f'/results{args.rank}.json', 'w') as f:
            json.dump(final_res, f)  
    return final_res

score_dict = {
    'airplane-3': 0.3,
    'airplane-0': 0.23, # 23
    'airplane-1': 0.35, # 35
    'airplane-2': 0.28,
    'bird-1': 0.25,
    # 'bird-0': 0.2,
    'bird-2': 0.3,
    # 'bird-3': 0.3,
    'person-3': 0.3,
    'person-1': 0.3, # 25
    'person-2': 0.4,
    'person-0': 0.23,
    'stock-3': 0.35,
    # 'stock-2': 0.23,
    'stock-1': 0.3,
    'stock-0': 0.22,
    # 'car-0': 0.15,
    'car-1': 0.43, # 23
    'car-2': 0.3, # 18
    # 'car-3': 0.25,
    'insect-3': 0.3,
    'insect-2': 0.3,
    # 'insect-1': 0.25,
    # 'insect-0': 0.2,
    'balloon-3': 0.13, # 15
    'balloon-2': 0.2,
    'balloon-1': 0.15, # 2
    # 'balloon-0': 0.25,
    'fish-3': 0.23,
    'fish-2': 0.23,
    # 'fish-1': 0.23,
    # 'fish-0': 0.23,
    'boat-3': 0.27,
    # 'boat-2': 0.006,
    'boat-1': 0.33,
    'boat-0': 0.2, # 2
    # 'ball-3': 0.18,
    # 'ball-0': 0.25,
    # 'ball-2': 0.15,
    # 'ball-1': 0.25,
    # 'ball-0': 0.009,
    'else': 0.25
}

score_dict_no = {
    'airplane-3': 0.28,
    'airplane-0': 0.18, # 23
    'airplane-1': 0.25, # 35
    'airplane-2': 0.25,
    'bird-1': 0.25,
    # 'bird-0': 0.2,
    'bird-2': 0.17,
    'bird-3': 0.2,
    'person-3': 0.25,
    'person-1': 0.3, # 25
    'person-2': 0.35,
    'person-0': 0.2,
    'stock-3': 0.25,
    # 'stock-2': 0.23,
    'stock-1': 0.25,
    'stock-0': 0.22,
    # 'car-0': 0.15,
    'car-1': 0.3, # 23
    'car-2': 0.2, # 18
    # 'car-3': 0.25,
    'insect-3': 0.3,
    'insect-2': 0.3,
    # 'insect-1': 0.25,
    # 'insect-0': 0.2,
    'balloon-3': 0.13, # 15
    'balloon-2': 0.2,
    'balloon-1': 0.12, # 2
    # 'balloon-0': 0.25,
    'fish-3': 0.21,
    'fish-2': 0.2,
    # 'fish-1': 0.23,
    'fish-0': 0.18,
    'boat-3': 0.25,
    'boat-2': 0.22,
    'boat-1': 0.22,
    'boat-0': 0.17, # 2
    'ball-3': 0.2,
    # 'ball-0': 0.25,
    # 'ball-2': 0.15,
    # 'ball-1': 0.25,
    'ball-0': 0.23,
    'else': 0.25
}

score_dict_4t = {
    'airplane-3': 0.35, # 79.8%
    'airplane-0': 0.28, # 23 -15.3
    'airplane-1': 0.35, # 35
    'airplane-2': 0.3, # 66.0%
    # 'bird-1': 0.25,
    'bird-0': 0.2,
    'bird-2': 0.3,
    'bird-3': 0.3, # 36.4%
    'person-3': 0.3,
    'person-1': 0.3, # 25
    'person-2': 0.42,
    'person-0': 0.23,
    'stock-3': 0.4, # 3.6%
    # 'stock-2': 0.23,
    'stock-1': 0.3,
    'stock-0': 0.22,
    'car-0': 0.2,
    'car-1': 0.35, # 23
    'car-2': 0.3, # 18
    'car-3': 0.2, # 74%
    'insect-3': 0.3,
    'insect-2': 0.3,
    'insect-1': 0.5,  #-37.1
    'insect-0': 0.15,  # 2%
    'balloon-3': 0.1, # 15
    'balloon-2': 0.2,
    'balloon-1': 0.15, # 2
    'balloon-0': 0.2,
    'fish-3': 0.2,
    'fish-2': 0.2,
    'fish-1': 0.25, # 70.1%
    'fish-0': 0.15, # 14.4%
    'boat-3': 0.3,
    'boat-2': 0.27,
    'boat-1': 0.33,
    'boat-0': 0.2, # 2
    # 'ball-3': 0.2,
    'ball-0': 0.3,
    # 'ball-2': 0.2,
    'ball-1': 0.18,
    # 'ball-0': 0.009,
    'else': 0.25
}

score_dict_ori = {
    'airplane-3': 0.28,
    'airplane-0': 0.2, # 23
    'airplane-1': 0.25, # 35
    'airplane-2': 0.25,
    'bird-1': 0.25,
    # 'bird-0': 0.2,
    'bird-2': 0.22,
    'bird-3': 0.2,
    'person-3': 0.28,
    'person-1': 0.3, # 25
    'person-2': 0.35,
    'person-0': 0.23,
    'stock-3': 0.28,
    # 'stock-2': 0.23,
    'stock-1': 0.28,
    'stock-0': 0.22,
    # 'car-0': 0.15,
    'car-1': 0.3, # 23
    'car-2': 0.2, # 18
    # 'car-3': 0.25,
    'insect-3': 0.3,
    'insect-2': 0.3,
    # 'insect-1': 0.25,
    # 'insect-0': 0.2,
    'balloon-3': 0.15, # 15
    'balloon-2': 0.2,
    'balloon-1': 0.17, # 2
    # 'balloon-0': 0.25,
    'fish-3': 0.21,
    'fish-2': 0.2,
    # 'fish-1': 0.23,
    'fish-0': 0.18,
    'boat-3': 0.25,
    # 'boat-2': 0.006,
    'boat-1': 0.3,
    'boat-0': 0.17, # 2
    'ball-3': 0.2,
    # 'ball-0': 0.25,
    # 'ball-2': 0.15,
    'ball-1': 0.24,
    'ball-0': 0.23,
    'else': 0.25
}

score_dict_1t = {
    'airplane-3': 0.28,
    'airplane-0': 0.21, # 23
    'airplane-1': 0.35, # 35
    'airplane-2': 0.25,
    'bird-1': 0.25,
    # 'bird-0': 0.2,
    'bird-2': 0.22,
    'bird-3': 0.25,
    'person-3': 0.28,
    'person-1': 0.3, # 25
    'person-2': 0.4,
    'person-0': 0.23,
    'stock-3': 0.28,
    # 'stock-2': 0.23,
    'stock-1': 0.28,
    'stock-0': 0.22,
    # 'car-0': 0.15,
    'car-1': 0.3, # 23
    'car-2': 0.2, # 18
    # 'car-3': 0.25,
    'insect-3': 0.3,
    'insect-2': 0.3,
    'insect-1': 0.3,
    # 'insect-0': 0.2,
    'balloon-3': 0.14, # 15
    'balloon-2': 0.2,
    'balloon-1': 0.16, # 2
    'balloon-0': 0.22,
    'fish-3': 0.21,
    'fish-2': 0.23,
    # 'fish-1': 0.23,
    'fish-0': 0.18,
    'boat-3': 0.25,
    # 'boat-2': 0.006,
    'boat-1': 0.28,
    'boat-0': 0.17, # 2
    'ball-3': 0.2,
    # 'ball-0': 0.25,
    # 'ball-2': 0.15,
    # 'ball-1': 0.25,
    'ball-0': 0.23,
    'else': 0.25
}

score_dict_ft = {
    'airplane-3': 0.3,
    'airplane-0': 0.23, # 23
    'airplane-1': 0.35, # 35
    'airplane-2': 0.28,
    'bird-1': 0.25,
    # 'bird-0': 0.2,
    'bird-2': 0.3,
    # 'bird-3': 0.3,
    'person-3': 0.3,
    'person-1': 0.3, # 25
    'person-2': 0.4,
    'person-0': 0.23,
    'stock-3': 0.35,
    # 'stock-2': 0.23,
    'stock-1': 0.3,
    'stock-0': 0.22,
    # 'car-0': 0.15,
    'car-1': 0.43, # 23
    'car-2': 0.3, # 18
    # 'car-3': 0.25,
    'insect-3': 0.3,
    'insect-2': 0.3,
    # 'insect-1': 0.3, ####25
    # 'insect-0': 0.2,
    'balloon-3': 0.13, # 15
    'balloon-2': 0.2,
    'balloon-1': 0.15, # 2
    # 'balloon-0': 0.25,
    'fish-3': 0.21, ####23
    'fish-2': 0.23,
    # 'fish-1': 0.23,
    # 'fish-0': 0.23,
    'boat-3': 0.27,
    # 'boat-2': 0.006,
    'boat-1': 0.33,
    'boat-0': 0.2, # 2
    # 'ball-3': 0.18,
    # 'ball-0': 0.25,
    # 'ball-2': 0.15,
    # 'ball-1': 0.23, ####25
    # 'ball-0': 0.009,
    'else': 0.25
}

@torch.no_grad()
def track_test(model, criterion, postprocessors, dataset, base_ds, device, output_dir, tracker, wo_class_error=False, args=None, logger=None):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    # if not wo_class_error:
    #     metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    # coco_evaluator = CocoEvaluator(base_ds, iou_types)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    save_path = 'results_all/results/'

    for seq, template in dataset:
        txt_name = save_path + str(seq) + '.txt'
        with open(txt_name, 'w') as f:
            f.close()
        tracker.reset()
        # import pdb;pdb.set_trace()
        if str(seq) in score_dict_4t.keys():
            # tracker.detection_person_thresh = score_dict_no[str(seq)]
            # tracker.detection_person_thresh = score_dict_no[str(seq)]
            # tracker.detection_person_thresh = score_dict[str(seq)]
            tracker.detection_person_thresh = score_dict_4t[str(seq)]
            # tracker.detection_person_thresh = score_dict_1t[str(seq)]
            # tracker.detection_person_thresh = 0.25
            # print(seq, tracker.detection_person_thresh)
        else:
            tracker.detection_person_thresh = score_dict['else']

        time_total = 0
        num_frames = 0

        print(seq)
        # metric_logger.add_meter('Track_seq', seq)
        start_frame = int(tracker.frame_range['start'] * len(seq))
        end_frame = int(tracker.frame_range['end'] * len(seq))
        seq_loader = DataLoader(torch.utils.data.Subset(seq, range(start_frame, end_frame)))
        # import pdb; pdb.set_trace()
        num_frames += len(seq_loader)

        start = time.time()

        for frame_data in metric_logger.log_every(seq_loader, 10, header, logger=logger):
            with torch.no_grad():
                # import pdb; pdb.set_trace()
                tracker.step(frame_data, template, postprocessors, seq.image_wh)

        results = tracker.get_results()

        for track_id in results.keys():
            track = results[track_id]
            for frame_id in track:
                x, y, w, h, s = track[frame_id]
                x = x - w/2
                y = y - h/2
                txt_name = save_path + str(seq) + '.txt'
                with open(txt_name, 'a') as f:
                    f.write(('%g,' * 6 + '-1,-1,-1,-1\n') % (frame_id, track_id, x,  # MOT format
                                                       y, w, h))

        time_total += time.time() - start
        print('Run time for {} use {}s'.format(seq, time_total))
        # import pdb; pdb.set_trace()
    final_res = []
    
    if args.output_dir:
        import json
        with open(args.output_dir + f'/results{args.rank}.json', 'w') as f:
            json.dump(final_res, f)  
    return final_res

class_dict = {"1": "person", "2": "bicycle", "3": "car", "4": "motorcycle", 
              "5": "airplane", "6": "bus","7": "train","8": "truck", 
              "9": "boat","10": "traffic light","11": "fire hydrant","13": "stop sign", 
              "14": "parking meter","15": "bench","16": "bird","17": "cat", 
              "18": "dog","19": "horse","20": "sheep","21": "cow", 
              "22": "elephant","23": "bear","24": "zebra","25": "giraffe", 
              "27": "backpack","28": "umbrella","31": "handbag","32": "tie", 
              "33": "suitcase", "34": "frisbee", "35": "skis", "36": "snowboard", 
              "37": "sports ball", "38": "kite", "39": "baseball bat", "40": "baseball glove", 
              "41": "skateboard", "42": "surfboard", "43": "tennis racket", "44": "bottle", 
              "46": "wine glass", "47": "cup", "48": "fork", "49": "knife", 
              "50": "spoon", "51": "bowl", "52": "banana", "53": "apple", 
              "54": "sandwich", "55": "orange", "56": "broccoli", "57": "carrot", 
              "58": "hot dog", "59": "pizza", "60": "donut", "61": "cake", 
              "62": "chair", "63": "couch", "64": "potted plant", "65": "bed", 
              "67": "dining table", "70": "toilet", "72": "tv", "73": "laptop", 
              "74": "mouse", "75": "remote", "76": "keyboard", "77": "cell phone", 
              "78": "microwave", "79": "oven", "80": "toaster", "81": "sink", 
              "82": "refrigerator", "84": "book", "85": "clock", "86": "vase", 
              "87": "scissors", "88": "teddy bear", "89": "hair drier", "90": "toothbrush"}

@torch.no_grad()
def ov_test(model, criterion, postprocessors, dataset, data_loader, base_ds, device, output_dir, wo_class_error=False, args=None, logger=None):
    model.eval()
    criterion.eval()
    # coco_evaluator = CocoEvaluator(base_ds, iou_types, useCats=useCats)

    metric_logger = utils.MetricLogger(delimiter="  ")
    # if not wo_class_error:
    #     metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    # coco_evaluator = CocoEvaluator(base_ds, iou_types)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    panoptic_evaluator = None
    if 'panoptic' in postprocessors.keys():
        panoptic_evaluator = PanopticEvaluator(
            data_loader.dataset.ann_file,
            data_loader.dataset.ann_folder,
            output_dir=os.path.join(output_dir, "panoptic_eval"),
        )

    final_res = []
    template_box = {}
    i = 1
    # import pdb; pdb.set_trace()
    for key in dataset.template_list.keys():
        class_results = []
        print('No.' + str(i), end=' ')
        i += 1
        print('Testing class ' + class_dict[str(key)])
        for samples, targets in metric_logger.log_every(data_loader, 10, header, logger=logger):
            samples = samples.to(device)
            targets = [{k: to_device(v, device) for k, v in t.items()} for t in targets]
            templates = [dataset.template_list[key]]

            # draw a template
            # from torchvision import transforms
            # unloader = transforms.ToPILImage()
            # image = templates[0][0].cpu().clone()  # clone the tensor
            # image = image.squeeze(0)  # remove the fake batch dimension
            # image = unloader(image)
            # name = 'ov_vis/test_' + str(key) + '.jpg'
            # image.save(name)
            # import pdb; pdb.set_trace()

            # import pdb; pdb.set_trace()
            outputs, _ = model(samples, templates)

            orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
            results = postprocessors['bbox'](outputs, orig_target_sizes, not_to_xyxy=True)
            # [scores: [100], labels: [100], boxes: [100, 4]] x B
            if 'segm' in postprocessors.keys():
                target_sizes = torch.stack([t["size"] for t in targets], dim=0)
                results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
            res = {target['image_id'].item(): output for target, output in zip(targets, results)}

            for image_id, outputs in res.items():
                _scores = outputs['scores']
                _labels = outputs['labels']
                _boxes = outputs['boxes']
                class_keep = _labels.bool()
                _scores = _scores[class_keep]
                _labels = _labels[class_keep]
                _boxes = _boxes[class_keep]
                # import pdb; pdb.set_trace()
                # ------------------ NMS -----------------------
                box = torch.zeros_like(_boxes)
                box[:, :2] = _boxes[:, :2] - (_boxes[:, 2:] / 2)
                box[:, 2:] = _boxes[:, :2] + (_boxes[:, 2:] / 2)
                keep = torchvision.ops.nms(box, _scores, 0.5)
                _boxes = _boxes[keep].tolist()
                _labels = _labels[keep].tolist()
                _scores = _scores[keep].tolist()
                # ----------------------------------------------
                # image_path = '../dataset/COCO/train2017/' + str(image_id).rjust(12, '0') + '.jpg'
                # img = cv2.imread(image_path, 1)
                # for i, box in enumerate(_boxes):
                #     if _scores[i] > 0.2 and _labels[i] == 1:
                #         cv2.putText(img, str(_labels[i])+':'+str(round(float(_scores[i]), 3)), (int(box[0]-box[2]/2), int(box[1]-box[3]/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                #         cv2.rectangle(img, (int(box[0]-box[2]/2), int(box[1]-box[3]/2)), (int(box[2]/2+box[0]), int(box[3]/2+box[1])), (0, 0, 255), 2)
                # save_path = 'ov_vis/vis/' + str(image_id).rjust(12, '0') + '.jpg'
                # cv2.imwrite(save_path, img)
                # import pdb; pdb.set_trace()
                # ----------------------------------------------
                for s, l, b in zip(_scores, _labels, _boxes):
                    assert isinstance(l, int)
                    itemdict = {
                            "image_id": int(image_id), 
                            "category_id": l*key, 
                            "bbox": b, 
                            "score": s,
                            }
                    class_results.append(itemdict)
                    final_res.append(itemdict)
                visual_result = False
                if visual_result:
                    image_path = '../dataset/COCO/val2017/' + str(image_id).rjust(12, '0') + '.jpg'
                    img = cv2.imread(image_path, 1)
                    for i, box in enumerate(_boxes):
                        if _scores[i] > 0.2:
                            cv2.rectangle(img, (int(box[0]-box[2]/2), int(box[1]-box[3]/2)), (int(box[2]/2+box[0]), int(box[3]/2+box[1])), (0, 255, 0), 2)
                    save_path = 'ov_vis/vis/' + str(image_id).rjust(12, '0') + '.jpg'
                    cv2.imwrite(save_path, img)
                    # from torchvision import transforms
                    # unloader = transforms.ToPILImage()
                    # image = dataset.template_list[key][0].cpu().clone()  # clone the tensor
                    # image = image.squeeze(0)  # remove the fake batch dimension
                    # image = unloader(image)
                    # name = 'ov_vis/example_' + str(key) + '.jpg'
                    # image.save(name)

        if args.output_dir:
            import json
            with open(args.output_dir + f'/results_class{key}.json', 'w') as f:
                json.dump(class_results, f, indent=2)
    # if args.output_dir:
    #     import json
    #     with open(args.output_dir + f'/results_class_all.json', 'w') as f:
    #         json.dump(final_res, f, indent=2)
    return final_res

@torch.no_grad()
def det_with_gtbox(model, criterion, postprocessors, data_loader, base_ds, device, output_dir, wo_class_error=False, args=None, logger=None):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    panoptic_evaluator = None
    if 'panoptic' in postprocessors.keys():
        panoptic_evaluator = PanopticEvaluator(
            data_loader.dataset.ann_file,
            data_loader.dataset.ann_folder,
            output_dir=os.path.join(output_dir, "panoptic_eval"),
        )

    final_res = []
    template_box = {}

    # load gt box
    import json
    det_results = {}
    with open('../gmot-main/data/COCO/annotations/gmot_test.json') as f:
        gmot_det = json.load(f)
    gmot_det = gmot_det['annotations']
    for item in gmot_det:
        image_id = item['image_id']
        box = item['bbox']
        if image_id not in det_results.keys():
            det_results[image_id] = []
            det_results[image_id].append([box[0]+box[2]/2, box[1]+box[3]/2, box[2], box[3]])
        else:
            det_results[image_id].append([box[0]+box[2]/2, box[1]+box[3]/2, box[2], box[3]])

    total_score = 0
    total_iou = 0
    total_number_iou = 0
    total_number_score = 0
    for samples, targets, templates in metric_logger.log_every(data_loader, 10, header, logger=logger):  
        samples = samples.to(device)
        targets = [{k: to_device(v, device) for k, v in t.items()} for t in targets]

        image_idx = targets[0]['image_id'].item()
        gt_boxes = det_results[image_idx]

        image_size = targets[0]['orig_size']
        gt_tensor = torch.tensor(gt_boxes).cuda()
        gt_tensor[:, 0] /= image_size[1]
        gt_tensor[:, 1] /= image_size[0]
        gt_tensor[:, 2] /= image_size[1]
        gt_tensor[:, 3] /= image_size[0]
        num_gt = len(gt_boxes)

        # import pdb; pdb.set_trace()
        outputs, _ = model(samples, templates, track_pos=gt_tensor)

        gt_outouts = {}
        for key in outputs.keys():
            if key in ['pred_logits', 'pred_boxes']:
                gt_outouts[key] = outputs[key][:, 600:, :]
        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        gt_results, box_index = postprocessors['bbox'](gt_outouts, orig_target_sizes, not_to_xyxy=False)
                
        track_scores = gt_results[0]['scores']
        track_boxes = gt_results[0]['boxes']
        track_labels = gt_results[0]['labels'].bool()
        track_scores = track_scores[track_labels]
        track_boxes = track_boxes[track_labels]
        box_index = box_index.squeeze(0)[track_labels]
        # import pdb; pdb.set_trace()

        
        _scores = track_scores.tolist()
        _labels = track_labels.tolist()
        _boxes = track_boxes.tolist()
        _order = box_index.tolist()
        # ----------------------------------------------
        temp_iou = {}
        temp_score = {}
        for s, l, b, i in zip(_scores, _labels, _boxes, _order):
            assert isinstance(l, int)
            # total_number += 1
            # total_score += s
            order = i % num_gt
            gt_box = gt_boxes[order]
            box = torch.tensor([gt_box[0], gt_box[1], gt_box[0]+gt_box[2], gt_box[1]+gt_box[3]])[None, :]
            pred_box = torch.tensor([b[0], b[1], b[2], b[3]])[None, :]
            iou = torchvision.ops.box_iou(box, pred_box).item()
            # total_iou += iou
            if order not in temp_iou.keys():
                temp_iou[order] = iou
                temp_score[order] = s
            else:
                # if s > temp_score[order]:
                #     temp_score[order] = s
                if iou > temp_iou[order]:
                    temp_iou[order] = iou
                    # temp_score[order] = s
                if s > temp_score[order]:
                    temp_score[order] = s
                # elif iou == temp_iou[order]:
                #     if s > temp_score[order]:
                #         temp_score[order] = s
            # import pdb; pdb.set_trace()
            itemdict = {
                    "image_id": int(image_idx), 
                    "category_id": l, 
                    "bbox": b, 
                    "score": s,
                    "gt": gt_box,
                    }
            final_res.append(itemdict)
        # import pdb; pdb.set_trace()
        for key in temp_iou.keys():
            total_iou += temp_iou[key]
        for key in temp_score.keys():
            total_score += temp_score[key]
        total_number_iou += len(temp_iou.keys())
        total_number_score += len(temp_score.keys())

    avg_score = total_score / total_number_score
    avg_iou = total_iou / total_number_iou
    print('avg_score:{}, avg_iou:{}'.format(avg_score, avg_iou))

    if args.output_dir:
        import json
        with open(args.output_dir + f'/results{args.rank}.json', 'w') as f:
            json.dump(final_res, f)  
    return final_res

@torch.no_grad()
def test_panda(model, criterion, postprocessors, data_loader, base_ds, device, output_dir, wo_class_error=False, args=None, logger=None):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    # if not wo_class_error:
    #     metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    # coco_evaluator = CocoEvaluator(base_ds, iou_types)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    panoptic_evaluator = None
    if 'panoptic' in postprocessors.keys():
        panoptic_evaluator = PanopticEvaluator(
            data_loader.dataset.ann_file,
            data_loader.dataset.ann_folder,
            output_dir=os.path.join(output_dir, "panoptic_eval"),
        )

    final_res = []
    template_box = {}
    for sample, targets, templates, image in metric_logger.log_every(data_loader, 10, header, logger=logger):
        total_scores = []
        total_labels = []
        total_bboxes = []
        batch_list = []
        sample = sample.to(device)
        targets = [{k: to_device(v, device) for k, v in t.items()} for t in targets]
        # import pdb; pdb.set_trace()
        image = image[0]
        size = image.size
        scales = [[2000, 1200], [4000, 2400], [6000, 3600], [10000, 6000]]
        
        for scale in scales:
            if scale[0] <= 2000:
                overlap = 0.8
            elif scale[0] > 2000 and scale[0] <= 7000:
                overlap = 0.6
            else:
                overlap = 0.5
            y_length = int(size[1] // (scale[1] * overlap))
            # import pdb; pdb.set_trace()
            for j in range(y_length):
                x_length = int(size[0] // (scale[0] * overlap))
                # import pdb; pdb.set_trace()
                y_begin = scale[1] * overlap * j
                if scale[1] <= 2000 and y_begin > size[1] / 2:
                    break
                elif scale[1] <= 5000 and y_begin > size[1] / 4 * 3:
                    break
                y_end = y_begin + scale[1]
                if y_end > image.size[1]:
                    y_end = image.size[1]
                    j = y_length
                for k in range(x_length):
                    x_begin = scale[0] * overlap * k
                    x_end = scale[0] + x_begin
                    if x_end > image.size[0]:
                        x_end = image.size[0]
                    box = (x_begin, y_begin, x_end, y_end)
                    image_seg = image.crop(box)

                    scale_x = image_seg.size[0]
                    scale_y = image_seg.size[1]

                    image_seg, _ = T.resize(image_seg, target=None, size=1920, max_size=1920)
                    transform = T.Compose([
                        # T.RandomResize([800], max_size=1333),
                        T.ToTensor(),
                        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    ])
                    image_pred, _ = transform(image_seg, None)
                    batch_list.append(image_pred.cuda())
                    # if len(batch_list) == 8 or (j == y_length-1 and k == x_length-1):
                    templates *= len(batch_list)
                    # import pdb; pdb.set_trace()
                    outputs, _ = model(batch_list, templates)
                    batch_list = []
                    # orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
                    # import pdb; pdb.set_trace()
                    orig_target_sizes = torch.tensor([[scale_y, scale_x]]).cuda()
                    results = postprocessors['bbox'](outputs, orig_target_sizes, not_to_xyxy=False)

                    res = {target['image_id'].item(): output for target, output in zip(targets, results)}

                    for image_id, outputs in res.items():
                        _scores = outputs['scores']
                        _labels = outputs['labels']
                        _boxes = outputs['boxes']
                        # ------------------ keep person -----------------------
                        keep = _labels.bool()
                        _boxes = _boxes[keep]
                        _scores = _scores[keep]
                        _labels = _labels[keep]
                        # ------------------ NMS -----------------------
                        box = torch.zeros_like(_boxes)
                        box[:, :2] = _boxes[:, :2] - (_boxes[:, 2:] / 2)
                        box[:, 2:] = _boxes[:, :2] + (_boxes[:, 2:] / 2)
                        keep = torchvision.ops.nms(box, _scores, 0.8)
                        _boxes = _boxes[keep].tolist()
                        _labels = _labels[keep].tolist()
                        _scores = _scores[keep].tolist()
                        # ----------------------------------------------
                        
                        for s, l, box in zip(_scores, _labels, _boxes):
                            # import pdb; pdb.set_trace()
                            if box[0] <= 15 or box[1] <= 15 or box[2] >= scale_x-15 or box[3] >= scale_y-15:
                                if box[0] <= 15 and x_begin != 0:
                                    continue
                                elif box[1] <= 15 and y_begin != 0:
                                    continue
                                elif box[2] >= scale_x-15 and x_end != image.size[1]:
                                    continue
                                elif box[3] >= scale_y-15 and y_end != image.size[0]:
                                    continue

                            box[0] = box[0] + x_begin
                            box[1] = box[1] + y_begin
                            box[2] = box[2] + x_begin
                            box[3] = box[3] + y_begin
                            assert isinstance(l, int)
                            total_scores.append(s)
                            total_labels.append(l)
                            total_bboxes.append(box)
        total_bboxes = torch.tensor(total_bboxes).cuda()
        total_scores = torch.tensor(total_scores).cuda()
        # import pdb; pdb.set_trace()
        keep = torchvision.ops.nms(total_bboxes, total_scores, 0.8)
        total_labels = torch.tensor(total_labels).cuda()
        # total_scores = torch.tensor(total_scores)

        total_bboxes = total_bboxes[keep].tolist()
        total_labels = total_labels[keep].tolist()
        total_scores = total_scores[keep].tolist()
        image_id = targets[0]['image_id']
        for s, l, b in zip(total_scores, total_labels, total_bboxes):
            assert isinstance(l, int)
            itemdict = {
                    "image_id": int(image_id), 
                    "category_id": l, 
                    "bbox": b, 
                    "score": s,
                    }
            final_res.append(itemdict)
    if args.output_dir:
        import json
        with open(args.output_dir + f'/results{args.rank}.json', 'w') as f:
            json.dump(final_res, f)  
    
    
    return final_res
