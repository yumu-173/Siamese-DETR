# ------------------------------------------------------------------------
# DINO
# Copyright (c) 2022 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# DN-DETR
# Copyright (c) 2022 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]


import torch
import torchvision
import random
import numpy as np
from copy import deepcopy

from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized, inverse_sigmoid)
# from .DABDETR import sigmoid_focal_loss
from util import box_ops
import torch.nn.functional as F

def random_crop():
    y = torch.rand((1000, 1))
    x = torch.rand((1000, 1))
    y = torch.clamp(y, 0, 0.8)
    x = torch.clamp(x, 0, 0.9)

    h = torch.rand((1000, 1)) / 5
    w = torch.rand((1000, 1)) / 10

    image_crop = torch.cat((x, y, x+h, y+w), dim=1).cuda()
    # print('image_crop:', image_crop.shape)
    return image_crop
    # print('image:', image_crop)

def prepare_for_sample_dn(dn_args, training, num_queries, hidden_dim, query_label):
    if training:
        targets, dn_number, label_noise_ratio, box_noise_scale = dn_args
        # positive and negative dn queries
        dn_number = dn_number * 2
        # limit dn_number < query_num/2
        targets_list = []
        for t in targets:
            if len(t['labels']) > num_queries/4:
                # print('********* in *********')
                rand_choise = np.random.randint(0, len(t['labels'])-1, int(num_queries/4))
                t['labels_dn'] = t['labels'][rand_choise]
                t['boxes_dn'] = t['boxes'][rand_choise]
            else:
                t['labels_dn'] = t['labels']
                t['boxes_dn'] = t['boxes']
            targets_list.append(t)
        targets = targets_list

        known = [(torch.ones_like(t['labels_dn'])).cuda() for t in targets]
        batch_size = len(known)
        known_num = [sum(k) for k in known]

        if int(max(known_num)) == 0:
            dn_number = 1
        else:
            if dn_number >= 100:
                dn_number = dn_number // (int(max(known_num) * 2))
            elif dn_number < 1:
                dn_number = 1
        if dn_number == 0:
            dn_number = 1
        unmask_bbox = unmask_label = torch.cat(known)
        # labels = torch.cat([t['labels'] for t in targets])
        boxes = torch.cat([t['boxes_dn'] for t in targets])

        batch_idx = torch.cat([torch.full_like(t['labels_dn'].long(), i) for i, t in enumerate(targets)])
        
        query_idx = torch.randint(0, num_queries, (batch_idx.shape[0],)).to(batch_idx)
        label_dn = query_label[batch_idx, query_idx, :] # num_targets x C

        known_indice = torch.nonzero(unmask_label + unmask_bbox)
        known_indice = known_indice.view(-1)

        known_indice = known_indice.repeat(2 * dn_number, 1).view(-1) # 2* dn_number * num_target
        input_label_embed = label_dn.repeat(2 * dn_number, 1) # 2* dn_number * num_target x C
        # print('input_label_embed', input_label_embed.shape)
        known_bid = batch_idx.repeat(2 * dn_number, 1).view(-1) # 2* dn_number * num_target
        known_bboxs = boxes.repeat(2 * dn_number, 1) # 2* dn_number * num_target x 4
        # known_labels_expaned = known_labels.clone()
        known_bbox_expand = known_bboxs.clone()

        single_pad = int(max(known_num))

        pad_size = int(single_pad * 2 * dn_number)
        positive_idx = torch.tensor(range(len(boxes))).long().cuda().unsqueeze(0).repeat(dn_number, 1)
        positive_idx += (torch.tensor(range(dn_number)) * len(boxes) * 2).long().cuda().unsqueeze(1)
        positive_idx = positive_idx.flatten()
        negative_idx = positive_idx + len(boxes)
        if box_noise_scale > 0:
            known_bbox_ = torch.zeros_like(known_bboxs)
            known_bbox_[:, :2] = known_bboxs[:, :2] - known_bboxs[:, 2:] / 2
            known_bbox_[:, 2:] = known_bboxs[:, :2] + known_bboxs[:, 2:] / 2

            diff = torch.zeros_like(known_bboxs)
            diff[:, :2] = known_bboxs[:, 2:] / 2
            diff[:, 2:] = known_bboxs[:, 2:] / 2

            rand_sign = torch.randint_like(known_bboxs, low=0, high=2, dtype=torch.float32) * 2.0 - 1.0
            rand_part = torch.rand_like(known_bboxs)
            rand_part[negative_idx] += 1.0
            rand_part *= rand_sign
            known_bbox_ = known_bbox_ + torch.mul(rand_part,
                                                  diff).cuda() * box_noise_scale
            known_bbox_ = known_bbox_.clamp(min=0.0, max=1.0)
            known_bbox_expand[:, :2] = (known_bbox_[:, :2] + known_bbox_[:, 2:]) / 2
            known_bbox_expand[:, 2:] = known_bbox_[:, 2:] - known_bbox_[:, :2]
        
        # m = known_labels_expaned.long().to('cuda')
        # input_label_embed = label_enc(m)
        input_bbox_embed = inverse_sigmoid(known_bbox_expand)

        padding_label = torch.zeros(pad_size, hidden_dim).cuda()
        padding_bbox = torch.zeros(pad_size, 4).cuda()

        input_query_label = padding_label.repeat(batch_size, 1, 1)
        input_query_bbox = padding_bbox.repeat(batch_size, 1, 1)

        map_known_indice = torch.tensor([]).to('cuda')
        if len(known_num):
            map_known_indice = torch.cat([torch.tensor(range(num)) for num in known_num])  # [1,2, 1,2,3]
            map_known_indice = torch.cat([map_known_indice + single_pad * i for i in range(2 * dn_number)]).long()
            # print('map_known_indice:', map_known_indice.shape)
        if len(known_bid):
            input_query_label[(known_bid.long(), map_known_indice)] = input_label_embed
            input_query_bbox[(known_bid.long(), map_known_indice)] = input_bbox_embed
            # print('input_query_label', input_query_label)
            # print('input_query_bbox', input_query_bbox)
        # import pdb; pdb.set_trace()
        tgt_size = pad_size + num_queries
        attn_mask = torch.ones(tgt_size, tgt_size).to('cuda') < 0
        # match query cannot see the reconstruct
        attn_mask[pad_size:, :pad_size] = True
        # reconstruct cannot see each other
        for i in range(dn_number):
            if i == 0:
                attn_mask[single_pad * 2 * i:single_pad * 2 * (i + 1), single_pad * 2 * (i + 1):pad_size] = True
            if i == dn_number - 1:
                attn_mask[single_pad * 2 * i:single_pad * 2 * (i + 1), :single_pad * i * 2] = True
            else:
                attn_mask[single_pad * 2 * i:single_pad * 2 * (i + 1), single_pad * 2 * (i + 1):pad_size] = True
                attn_mask[single_pad * 2 * i:single_pad * 2 * (i + 1), :single_pad * 2 * i] = True

        dn_meta = {
            'pad_size': pad_size,
            'num_dn_group': dn_number,
        }

        # recover target with multi-template
        # for t1, t2 in zip(targets_copy, targets):
        #     # print(t1.keys())
        #     for key in t1.keys():
        #         t2[key] = t1[key]
    else:

        input_query_label = None
        input_query_bbox = None
        attn_mask = None
        dn_meta = None

    
    # print(len(targets))
    return input_query_label, input_query_bbox, attn_mask, dn_meta


def prepare_for_cdn(dn_args, training, num_queries, hidden_dim, label_enc, temp_pos, denoise_query):
    """
        A major difference of DINO from DN-DETR is that the author process pattern embedding pattern embedding in its detector
        forward function and use learnable tgt embedding, so we change this function a little bit.
        :param dn_args: targets, dn_number, label_noise_ratio, box_noise_scale
        :param training: if it is training or inference
        :param num_queries: number of queires
        :param num_classes: number of classes
        :param hidden_dim: transformer hidden dim
        :param label_enc: encode labels in dn
        :return:
        """
    if training:
        targets, dn_number, label_noise_ratio, box_noise_scale = dn_args
        # positive and negative dn queries
        dn_number = dn_number * 2
        
        # limit dn_number < query_num/2
        targets_list = []
        for t in targets:
            if len(t['labels']) > num_queries/4:
                # print('********* in *********')
                rand_choise = np.random.randint(0, len(t['labels'])-1, int(num_queries/4))
                t['labels_dn'] = t['labels'][rand_choise]
                t['boxes_dn'] = t['boxes'][rand_choise]
            else:
                t['labels_dn'] = t['labels']
                t['boxes_dn'] = t['boxes']
            targets_list.append(t)
        targets = targets_list
        
        known = [(torch.ones_like(t['labels'])).cuda() for t in targets]
        batch_size = len(known)
        # print('batch_szie:', batch_size)
        known_num = [sum(k) for k in known]
        if int(max(known_num)) == 0:
            dn_number = 1
        else:
            if dn_number >= 100:
                dn_number = dn_number // (int(max(known_num) * 2))
            elif dn_number < 1:
                dn_number = 1
        if dn_number == 0:
            dn_number = 1
        unmask_bbox = unmask_label = torch.cat(known)
        labels = torch.cat([t['labels'] for t in targets])
        boxes = torch.cat([t['boxes'] for t in targets])
        # print("boxes:", boxes)
        if denoise_query:
            # negative query -----------------------------------------------------------------------------------------------
            image_crop = random_crop()
            # boxs = torch.cat([t['boxes'] for t in targets])
            boxs_ = torch.zeros_like(boxes)
            boxs_[:, :2] = boxes[:, :2] - boxes[:, 2:] / 2
            boxs_[:, 2:] = boxes[:, :2] + boxes[:, 2:] / 2
            # print("boxs:", boxs_.shape)
            # print("image_crop:", image_crop[0])
            if boxes.shape[0] != 0:
                IOU = torchvision.ops.box_iou(image_crop, boxs_)
                # print('IOU:', IOU)
                # print('IOU SHAPE:', IOU.shape)
                max_IOU = torch.max(IOU, dim=1)
                max_iou = max_IOU[0]
                # max_iou = torch.nonzero(max_IOU[0])
                # print('iou:', max_iou)
                iou_thr = random.randint(1, 3)
                negative_id = 0
                for i, item in enumerate(max_iou):
                    if item > 0 and item < 0.1 and iou_thr == 1:
                        negative_id = i
                        break
                    elif item > 0.1 and item < 0.2 and iou_thr == 2:
                        negative_id = i
                        break
                    elif item > 0.2 and item < 0.3 and iou_thr == 3:
                        negative_id = i
                        break
                negative_emb = image_crop[negative_id]
            else:
                negative_emb = image_crop[0]
            # print('iou:', max_iou[negative_id])
            # print('negative_emb:', negative_emb)
            # positive query -----------------------------------------------------------------------------------------------
            # print('tempid:', temp_id)
            box_list = []
            label_list = []
            # print('len target:', targets)
            for i, t in enumerate(targets):
                # print('i:', i)
                label = torch.zeros(1, dtype=torch.int64).repeat(len(t['labels'])).cuda()
                # print('temp_pos:', temp_pos[i])
                box = torch.Tensor([temp_pos[i][0]/t['orig_size'][0], temp_pos[i][1]/t['orig_size'][1], temp_pos[i][2]/t['orig_size'][0], temp_pos[i][3]/t['orig_size'][1]]) \
                                        [None, :].repeat(len(t['boxes']), 1).cuda()
                # print('box:', box[0])
                box_list.append(box)
                label_list.append(label)
            boxes = torch.cat(box_list, dim=0)
            labels = torch.cat(label_list, dim=0)
            # print('boxes1:', boxes)
            # print('labels', labels.shape)
            # print('labels', labels)
            # --------------------------------------------------------------------------------------------------------------
        
        batch_idx = torch.cat([torch.full_like(t['labels'].long(), i) for i, t in enumerate(targets)])
        # print('batch_idx:', batch_idx.shape)
            

        known_indice = torch.nonzero(unmask_label + unmask_bbox)
        known_indice = known_indice.view(-1)

        known_indice = known_indice.repeat(2 * dn_number, 1).view(-1)
        known_labels = labels.repeat(2 * dn_number, 1).view(-1)
        known_bid = batch_idx.repeat(2 * dn_number, 1).view(-1)
        known_bboxs = boxes.repeat(2 * dn_number, 1)
        known_labels_expaned = known_labels.clone()
        known_bbox_expand = known_bboxs.clone()

        if label_noise_ratio > 0:
            p = torch.rand_like(known_labels_expaned.float())
            chosen_indice = torch.nonzero(p < (label_noise_ratio * 0.5)).view(-1)  # half of bbox prob
            # new_label = torch.randint_like(chosen_indice, 0, num_classes)  # randomly put a new one here
            new_label = torch.ones_like(chosen_indice)
            known_labels_expaned.scatter_(0, chosen_indice, new_label)
        single_pad = int(max(known_num))

        pad_size = int(single_pad * 2 * dn_number)
        positive_idx = torch.tensor(range(len(boxes))).long().cuda().unsqueeze(0).repeat(dn_number, 1)
        positive_idx += (torch.tensor(range(dn_number)) * len(boxes) * 2).long().cuda().unsqueeze(1)
        positive_idx = positive_idx.flatten()
        negative_idx = positive_idx + len(boxes)
        if box_noise_scale > 0:
            known_bbox_ = torch.zeros_like(known_bboxs)
            known_bbox_[:, :2] = known_bboxs[:, :2] - known_bboxs[:, 2:] / 2
            known_bbox_[:, 2:] = known_bboxs[:, :2] + known_bboxs[:, 2:] / 2

            diff = torch.zeros_like(known_bboxs)
            diff[:, :2] = known_bboxs[:, 2:] / 2
            diff[:, 2:] = known_bboxs[:, 2:] / 2

            rand_sign = torch.randint_like(known_bboxs, low=0, high=2, dtype=torch.float32) * 2.0 - 1.0
            rand_part = torch.rand_like(known_bboxs)
            rand_part[negative_idx] += 1.0
            rand_part *= rand_sign
            known_bbox_ = known_bbox_ + torch.mul(rand_part,
                                                  diff).cuda() * box_noise_scale
            known_bbox_ = known_bbox_.clamp(min=0.0, max=1.0)
            known_bbox_expand[:, :2] = (known_bbox_[:, :2] + known_bbox_[:, 2:]) / 2
            known_bbox_expand[:, 2:] = known_bbox_[:, 2:] - known_bbox_[:, :2]

        # change box *****************************************************************
        if denoise_query:
            known_bbox_expand[chosen_indice] = negative_emb
        # ****************************************************************************
        m = known_labels_expaned.long().to('cuda')
        input_label_embed = label_enc(m)
        print('input_label_embed', input_label_embed.shape)
        exit(0)
        input_bbox_embed = inverse_sigmoid(known_bbox_expand)

        padding_label = torch.zeros(pad_size, hidden_dim).cuda()
        padding_bbox = torch.zeros(pad_size, 4).cuda()

        input_query_label = padding_label.repeat(batch_size, 1, 1)
        input_query_bbox = padding_bbox.repeat(batch_size, 1, 1)

        map_known_indice = torch.tensor([]).to('cuda')
        if len(known_num):
            map_known_indice = torch.cat([torch.tensor(range(num)) for num in known_num])  # [1,2, 1,2,3]
            map_known_indice = torch.cat([map_known_indice + single_pad * i for i in range(2 * dn_number)]).long()
        if len(known_bid):
            input_query_label[(known_bid.long(), map_known_indice)] = input_label_embed
            input_query_bbox[(known_bid.long(), map_known_indice)] = input_bbox_embed

        tgt_size = pad_size + num_queries
        attn_mask = torch.ones(tgt_size, tgt_size).to('cuda') < 0
        # match query cannot see the reconstruct
        attn_mask[pad_size:, :pad_size] = True
        # reconstruct cannot see each other
        for i in range(dn_number):
            if i == 0:
                attn_mask[single_pad * 2 * i:single_pad * 2 * (i + 1), single_pad * 2 * (i + 1):pad_size] = True
            if i == dn_number - 1:
                attn_mask[single_pad * 2 * i:single_pad * 2 * (i + 1), :single_pad * i * 2] = True
            else:
                attn_mask[single_pad * 2 * i:single_pad * 2 * (i + 1), single_pad * 2 * (i + 1):pad_size] = True
                attn_mask[single_pad * 2 * i:single_pad * 2 * (i + 1), :single_pad * 2 * i] = True

        dn_meta = {
            'pad_size': pad_size,
            'num_dn_group': dn_number,
        }
    else:

        input_query_label = None
        input_query_bbox = None
        attn_mask = None
        dn_meta = None

    return input_query_label, input_query_bbox, attn_mask, dn_meta


def dn_post_process(outputs_class, outputs_coord, dn_meta, aux_loss, _set_aux_loss):
    """
        post process of dn after output from the transformer
        put the dn part in the dn_meta
    """
    if dn_meta and dn_meta['pad_size'] > 0:
        output_known_class = outputs_class[:, :, :dn_meta['pad_size'], :]
        output_known_coord = outputs_coord[:, :, :dn_meta['pad_size'], :]
        outputs_class = outputs_class[:, :, dn_meta['pad_size']:, :]
        outputs_coord = outputs_coord[:, :, dn_meta['pad_size']:, :]
        out = {'pred_logits': output_known_class[-1], 'pred_boxes': output_known_coord[-1]}
        if aux_loss:
            out['aux_outputs'] = _set_aux_loss(output_known_class, output_known_coord)
        dn_meta['output_known_lbs_bboxes'] = out
    return outputs_class, outputs_coord


