# ------------------------------------------------------------------------
# DINO
# Copyright (c) 2022 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Conditional DETR model and criterion classes.
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
import copy
import math
import random
from typing import List
import torch
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import torch.nn.init as init
from torch import nn
from torchvision.ops.boxes import nms
import numpy as np

from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized, inverse_sigmoid)

from .backbone import build_backbone
from .matcher import build_matcher
from .segmentation import (DETRsegm, PostProcessPanoptic, PostProcessSegm,
                           dice_loss)
from .deformable_transformer import build_deformable_transformer
from .utils import sigmoid_focal_loss, MLP

from ..registry import MODULE_BUILD_FUNCS
from .dn_components import prepare_for_cdn,dn_post_process

class ClassLinear(nn.Module):
    def __init__(self, out_features: int, bias: bool = True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.out_features = out_features
        self.pdist = nn.PairwiseDistance(p=2)
        if bias:
            self.bias = Parameter(torch.zeros(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)

    def forward(self, hs, template_feature):
        template_feature = torch.mean(template_feature, dim=1, keepdim=True)
        # import pdb; pdb.set_trace()
        template_feature = template_feature.repeat(1, hs.shape[1], 1) # B x N x D
        pd = self.pdist(hs, template_feature)[:, :, None] # B x N x 1
        pd = torch.cat([pd, -pd], dim=-1)
        pd = pd + self.bias.view(1, 1, self.out_features)
        return pd

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )

class DINO(nn.Module):
    """ This is the Cross-Attention Detector module that performs object detection """
    def __init__(self, backbone, transformer, num_classes, num_queries, 
                    aux_loss=False, iter_update=False,
                    query_dim=2, 
                    random_refpoints_xy=False,
                    fix_refpoints_hw=-1,
                    num_feature_levels=1,
                    nheads=8,
                    # two stage
                    two_stage_type='no', # ['no', 'standard']
                    two_stage_add_query_num=0,
                    dec_pred_class_embed_share=True,
                    dec_pred_bbox_embed_share=True,
                    two_stage_class_embed_share=True,
                    two_stage_bbox_embed_share=True,
                    decoder_sa_type = 'sa',
                    num_patterns = 0,
                    dn_number = 100,
                    dn_box_noise_scale = 0.4,
                    dn_label_noise_ratio = 0.5,
                    dn_labelbook_size = 100,
                    temp_weight=False,
                    denoise_query=False,
                    template_lvl=4,
                    number_template=1,
                    keep_template_look=False
                    ):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         Conditional DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.

            fix_refpoints_hw: -1(default): learn w and h for each box seperately
                                >0 : given fixed number
                                -2 : learn a shared w and h
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim = transformer.d_model
        self.num_feature_levels = num_feature_levels
        self.nheads = nheads
        self.label_enc = nn.Embedding(dn_labelbook_size + 1, hidden_dim)
        self.share_weight = temp_weight
        self.denoise_query = denoise_query
        self.template_lvl = template_lvl
        self.number_template = number_template
        self.keep_template_look = keep_template_look

        # setting query dim
        self.query_dim = query_dim
        assert query_dim == 4
        self.random_refpoints_xy = random_refpoints_xy
        self.fix_refpoints_hw = fix_refpoints_hw

        # for dn training
        self.num_patterns = num_patterns
        self.dn_number = dn_number
        self.dn_box_noise_scale = dn_box_noise_scale
        self.dn_label_noise_ratio = dn_label_noise_ratio
        self.dn_labelbook_size = dn_labelbook_size

        # prepare input projection layers
        if num_feature_levels > 1:
            num_backbone_outs = len(backbone.num_channels)
            # ---------------------------------------------------------------------------------------------------------
            print('num_backbone_outs', num_backbone_outs)
            # ---------------------------------------------------------------------------------------------------------
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone.num_channels[_]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
            print('share weight:', self.share_weight)
            print('*****************************')
            if self.share_weight:
                print('share weight')
            else:
                temp_input_proj_list = []
                if self.template_lvl <= 3:
                    for _ in range(self.template_lvl):
                        # import pdb; pdb.set_trace()
                        in_channels = backbone.num_channels[_]
                        print('in_channels:', in_channels)
                        # import pdb; pdb.set_trace()
                        temp_input_proj_list.append(nn.Sequential(
                            nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                            nn.GroupNorm(32, hidden_dim),
                        ))
                    for _ in range(num_feature_levels - self.template_lvl):
                        temp_input_proj_list.append(nn.Sequential(
                            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                            nn.GroupNorm(32, hidden_dim),
                        ))
                    # print('temp_input_proj_list', temp_input_proj_list)
                    # import pdb; pdb.set_trace()
                elif self.template_lvl == 4:
                    for _ in range(num_backbone_outs):
                        # import pdb; pdb.set_trace()
                        in_channels = backbone.num_channels[_]
                        # print('in_channels:', in_channels)
                        # import pdb; pdb.set_trace()
                        temp_input_proj_list.append(nn.Sequential(
                            nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                            nn.GroupNorm(32, hidden_dim),
                        ))
                    for _ in range(num_feature_levels - num_backbone_outs):
                        temp_input_proj_list.append(nn.Sequential(
                            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                            nn.GroupNorm(32, hidden_dim),
                        ))
                        in_channels = hidden_dim
                # print('temp_input_proj_list', temp_input_proj_list)
                # import pdb; pdb.set_trace()
                self.temp_input_proj = nn.ModuleList(temp_input_proj_list)
        else:
            assert two_stage_type == 'no', "two_stage_type should be no if num_feature_levels=1 !!!"
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(backbone.num_channels[-1], hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                )])
            if self.share_weight == False:
                self.temp_input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(backbone.num_channels[-1], hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                )])

        self.backbone = backbone
        # self.template_backbone = template_backbone
        self.aux_loss = aux_loss
        self.box_pred_damping = box_pred_damping = None

        self.iter_update = iter_update
        assert iter_update, "Why not iter_update?"

        # prepare pred layers
        self.dec_pred_class_embed_share = dec_pred_class_embed_share
        self.dec_pred_bbox_embed_share = dec_pred_bbox_embed_share
        # prepare class & box embed
        if self.keep_template_look:
            print('class_embed is class_Linear')
            _class_embed = ClassLinear(num_classes, bias=True)
        else:
            print('class_embed is nn.Linear')
            _class_embed = nn.Linear(hidden_dim, num_classes)
        _bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        # init the two embed layers
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        _class_embed.bias.data = torch.ones(self.num_classes) * bias_value
        nn.init.constant_(_bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(_bbox_embed.layers[-1].bias.data, 0)

        if dec_pred_bbox_embed_share:
            box_embed_layerlist = [_bbox_embed for i in range(transformer.num_decoder_layers)]
        else:
            box_embed_layerlist = [copy.deepcopy(_bbox_embed) for i in range(transformer.num_decoder_layers)]
        if dec_pred_class_embed_share:
            class_embed_layerlist = [_class_embed for i in range(transformer.num_decoder_layers)]
        else:
            class_embed_layerlist = [copy.deepcopy(_class_embed) for i in range(transformer.num_decoder_layers)]
        self.bbox_embed = nn.ModuleList(box_embed_layerlist)
        self.class_embed = nn.ModuleList(class_embed_layerlist)
        self.transformer.decoder.bbox_embed = self.bbox_embed
        self.transformer.decoder.class_embed = self.class_embed

        # two stage
        self.two_stage_type = two_stage_type
        self.two_stage_add_query_num = two_stage_add_query_num
        assert two_stage_type in ['no', 'standard'], "unknown param {} of two_stage_type".format(two_stage_type)
        if two_stage_type != 'no':
            if two_stage_bbox_embed_share:
                assert dec_pred_class_embed_share and dec_pred_bbox_embed_share
                self.transformer.enc_out_bbox_embed = _bbox_embed
            else:
                self.transformer.enc_out_bbox_embed = copy.deepcopy(_bbox_embed)
    
            if two_stage_class_embed_share:
                assert dec_pred_class_embed_share and dec_pred_bbox_embed_share
                self.transformer.enc_out_class_embed = _class_embed
            else:
                self.transformer.enc_out_class_embed = copy.deepcopy(_class_embed)
    
            # self.refpoint_embed = None
            if self.two_stage_add_query_num > 0:
                self.init_ref_points(two_stage_add_query_num)

        self.decoder_sa_type = decoder_sa_type
        assert decoder_sa_type in ['sa', 'ca_label', 'ca_content']
        # self.replace_sa_with_double_ca = replace_sa_with_double_ca
        if decoder_sa_type == 'ca_label':
            self.label_embedding = nn.Embedding(num_classes, hidden_dim)
            for layer in self.transformer.decoder.layers:
                layer.label_embedding = self.label_embedding
        else:
            for layer in self.transformer.decoder.layers:
                layer.label_embedding = None
            self.label_embedding = None

        self._reset_parameters()

    def _reset_parameters(self):
        # init input_proj
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

    def init_ref_points(self, use_num_queries):
        raise RuntimeError('The following code has some errors need to be fixed!')
        self.refpoint_embed = nn.Embedding(use_num_queries, self.query_dim)

        if self.random_refpoints_xy:
            # import ipdb; ipdb.set_trace()
            self.refpoint_embed.weight.data[:, :2].uniform_(0,1)
            self.refpoint_embed.weight.data[:, :2] = inverse_sigmoid(self.refpoint_embed.weight.data[:, :2])
            self.refpoint_embed.weight.data[:, :2].requires_grad = False

        
        if self.fix_refpoints_hw > 0:
            print("fix_refpoints_hw: {}".format(self.fix_refpoints_hw))
            assert self.random_refpoints_xy
            self.refpoint_embed.weight.data[:, 2:] = self.fix_refpoints_hw
            self.refpoint_embed.weight.data[:, 2:] = inverse_sigmoid(self.refpoint_embed.weight.data[:, 2:])
            self.refpoint_embed.weight.data[:, 2:].requires_grad = False
        elif int(self.fix_refpoints_hw) == -1:
            pass
        elif int(self.fix_refpoints_hw) == -2:
            print('learn a shared h and w')
            assert self.random_refpoints_xy
            self.refpoint_embed = nn.Embedding(use_num_queries, 2)
            self.refpoint_embed.weight.data[:, :2].uniform_(0,1)
            self.refpoint_embed.weight.data[:, :2] = inverse_sigmoid(self.refpoint_embed.weight.data[:, :2])
            self.refpoint_embed.weight.data[:, :2].requires_grad = False
            self.hw_embed = nn.Embedding(1, 1)
        else:
            raise NotImplementedError('Unknown fix_refpoints_hw {}'.format(self.fix_refpoints_hw))

    def forward(self, samples: NestedTensor, templates: NestedTensor, targets:List=None, track_pos=None, num_temp=1):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x num_classes]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, width, height). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        # import pdb; pdb.set_trace()
        # templates = nested_tensor_from_tensor_list(templates).to('cuda')
        template = []
        for template_list in templates:
            template.extend(template_list[:num_temp])
        # import pdb; pdb.set_trace()
        templates = nested_tensor_from_tensor_list(template).to('cuda')
        # temp_features, temp_poss = self.template_backbone(templates)
        # print('---template_begin---')
        # print(templates.shape)
        
        temp_features, temp_poss = self.backbone(templates)
        
        # print('---template_over---')
        # import pdb; pdb.set_trace()

        # from thop import profile
        # flops, params = profile(self.backbone, (templates, ))
        # print('flops: %.2f M, params: %.2f M' % (flops / 1e6, params / 1e6))
        # exit(0)

        features, poss = self.backbone(samples)
        # import pdb; pdb.set_trace()
        # --------------------------------------------------------------------------------------------------------------
        # print('samples:', samples.shape)
        # print('templates:', templates.shape)
        # print('targets:', targets)
        # print(temp_features[0])
        # print(temp_features[1]['tensors'])
        # print(temp_features[2].shape)
        # print(temp_features[3].shape)
        #
        # print(features[0].shape)
        # print(features[1].shape)
        # print(features[2].shape)
        # print(features[3].shape)
        # tgt = []
        # for i, feature in enumerate(temp_features):
        #     # print(feature.shape)
        #     src, mask = feature.decompose()
        #     bs, C, _, _ = src.shape
        #     conv1 = nn.Conv2d(in_channels=C, out_channels=256,
        #                       kernel_size=1, stride=1, padding=0).cuda()
        #     feat = conv1(src)
        #     pooling = nn.AdaptiveAvgPool2d(1).cuda()
        #     # print(feat.shape)
        #     feat = pooling(feat).reshape(bs, 256, 1).transpose(2, 1)
        #     # print(feat.shape)
        #     feat = feat.repeat(1, 900, 1)
        #     # print(feat.shape)
        #     tgt.append(feat)
        # for item in tgt:
        #     print(item.shape)
        # A = torch.ones(2, 152, 256)  # 2x3的张量（矩阵）
        # print("A:\n", A, "\nA.shape:\n", A.shape, "\n")
        #
        # B = 2 * torch.ones(2, 900, 256)  # 4x3的张量（矩阵）
        # print("B:\n", B, "\nB.shape:\n", B.shape, "\n")
        #
        # C = torch.cat([A, B], dim=1)  # 按维数0（行）拼接
        # print("C:\n", C, "\nC.shape:\n", C.shape, "\n")

        # --------------------------------------------------------------------------------------------------------------


        srcs = []
        masks = []
        temp_srcs = []
        temp_masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            assert mask is not None
        # share weight for template feature ----------------------------------------------------------------------------
        new_temp_features = temp_features[-1*self.template_lvl:]
        # import pdb; pdb.set_trace()
        for i in range(len(new_temp_features), len(features)):
            # print(i)
            temp_features[i] = temp_features[i-1]
        # print(len(temp_features))
        if self.share_weight:
            # print('**************** True *******************')
            # print(self.share_weight)
            for l, feat in enumerate(temp_features):
                src, mask = feat.decompose()
                temp_srcs.append(self.input_proj[l](src))
                temp_masks.append(mask)
                assert temp_masks is not None
        else:
            # print('**************** False ******************')
            for l, feat in enumerate(temp_features):
                src, mask = feat.decompose()
                # print(src.shape)
                temp_srcs.append(self.temp_input_proj[l](src))
                temp_masks.append(mask)
                assert temp_masks is not None
        # --------------------------------------------------------------------------------------------------------------
        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = samples.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                poss.append(pos_l)
                
        # import pdb; pdb.set_trace()
        if self.num_feature_levels > len(temp_srcs):
            _len_srcs = len(temp_srcs)
            # print('_len_srcs', _len_srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    temp_src = self.temp_input_proj[l](temp_features[-1].tensors)
                else:
                    temp_src = self.temp_input_proj[l](temp_srcs[-1])
                m = templates.mask
                temp_mask = F.interpolate(m[None].float(), size=temp_src.shape[-2:]).to(torch.bool)[0]
                temp_pos_l = self.backbone[1](NestedTensor(temp_src, temp_mask)).to(temp_src.dtype)
                temp_srcs.append(temp_src)
                temp_masks.append(temp_mask)
                temp_poss.append(temp_pos_l)
        # import pdb; pdb.set_trace()
        # if self.dn_number > 0 and targets is not None:
        #     input_query_label, input_query_bbox, attn_mask, dn_meta =\
        #         prepare_for_cdn(dn_args=(targets, self.dn_number, self.dn_label_noise_ratio, self.dn_box_noise_scale),
        #                         training=self.training,num_queries=self.num_queries,
        #                         hidden_dim=self.hidden_dim,label_enc=self.label_enc,temp_pos=temp_pos,denoise_query=self.denoise_query)
        # else:
            # assert targets is None
        input_query_bbox = input_query_label = attn_mask = dn_meta = None

        # --------------------------------------------------------------------------------------------------------------
        # print('input_query_label:', input_query_label.shape)
        # print('temp srcs0:', temp_srcs[0].shape)
        # print('temp srcs1:', temp_srcs[1].shape)
        # print('temp srcs2:', temp_srcs[2].shape)
        # print('temp srcs3:', temp_srcs[3].shape)
        # print('srcs0:', srcs[0].shape)
        # print('srcs1:', srcs[1].shape)
        # print('srcs2:', srcs[2].shape)
        # print('srcs3:', srcs[3].shape)
        # --------------------------------------------------------------------------------------------------------------
        # hs, reference, hs_enc, ref_enc, init_box_proposal = self.transformer(srcs, masks, input_query_bbox, poss, input_query_label, temp_srcs, temp_masks, attn_mask)
        hs, reference, hs_enc, ref_enc, init_box_proposal, dn_meta, template_feature = self.transformer(srcs, masks, poss, temp_srcs, temp_masks, targets, track_pos, num_temp)

        # split output into batch
        if self.two_stage_type == 'standard':
            raise NotImplementedError
        self.number_template = num_temp
        split_hs = []
        for hs_item in hs:
            hs_split = []
            split_num = hs_item.shape[1] // self.number_template
            for i in range(len(hs_item)):
                hs_temp = torch.split(hs_item[i], split_num, dim = 0)
                hs_temp = torch.stack(hs_temp)
                hs_split.append(hs_temp)
            hs_item = torch.cat(hs_split, dim=0)
            # import pdb; pdb.set_trace()
            split_hs.append(hs_item)
        hs = split_hs
        split_reference = []
        for reference_item in reference:
            ref_split = []
            split_num = reference_item.shape[1] // self.number_template
            for i in range(len(reference_item)):
                ref_temp = torch.split(reference_item[i], split_num, dim = 0)
                ref_temp = torch.stack(ref_temp)
                ref_split.append(ref_temp)
            reference_item = torch.cat(ref_split, dim=0)
            # import pdb; pdb.set_trace()
            split_reference.append(reference_item)
        reference = split_reference
        # import pdb; pdb.set_trace()
        # In case num object=0
        hs[0] += self.label_enc.weight[0, 0] * 0.0

        # deformable-detr-like anchor update
        # reference_before_sigmoid = inverse_sigmoid(reference[:-1]) # n_dec, bs, nq, 4
        outputs_coord_list = []
        # import pdb; pdb.set_trace()
        for dec_lid, (layer_ref_sig, layer_bbox_embed, layer_hs) in enumerate(zip(reference[:-1], self.bbox_embed, hs)):
            layer_delta_unsig = layer_bbox_embed(layer_hs)
            layer_outputs_unsig = layer_delta_unsig + inverse_sigmoid(layer_ref_sig)
            layer_outputs_unsig = layer_outputs_unsig.sigmoid()
            outputs_coord_list.append(layer_outputs_unsig)
        outputs_coord_list = torch.stack(outputs_coord_list)


        # outputs_class = self.class_embed(hs)
        if self.keep_template_look:
            outputs_class = torch.stack([layer_cls_embed(layer_hs, template_feature) for
                                            layer_cls_embed, layer_hs in zip(self.class_embed, hs)])
        else:
            outputs_class = torch.stack([layer_cls_embed(layer_hs) for
                                            layer_cls_embed, layer_hs in zip(self.class_embed, hs)])
        # import pdb; pdb.set_trace()
        if self.dn_number > 0 and dn_meta is not None:
            outputs_class, outputs_coord_list = \
                dn_post_process(outputs_class, outputs_coord_list,
                                dn_meta, self.aux_loss, self._set_aux_loss)
        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord_list[-1]}
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord_list)


        # for encoder output
        if hs_enc is not None:
            # prepare intermediate outputs
            interm_coord = ref_enc[-1]
            interm_class = self.transformer.enc_out_class_embed(hs_enc[-1])
            out['interm_outputs'] = {'pred_logits': interm_class, 'pred_boxes': interm_coord}
            out['interm_outputs_for_matching_pre'] = {'pred_logits': interm_class, 'pred_boxes': init_box_proposal}

            # prepare enc outputs
            # import ipdb; ipdb.set_trace()
            if hs_enc.shape[0] > 1:
                enc_outputs_coord = []
                enc_outputs_class = []
                for layer_id, (layer_box_embed, layer_class_embed, layer_hs_enc, layer_ref_enc) in enumerate(zip(self.enc_bbox_embed, self.enc_class_embed, hs_enc[:-1], ref_enc[:-1])):
                    layer_enc_delta_unsig = layer_box_embed(layer_hs_enc)
                    layer_enc_outputs_coord_unsig = layer_enc_delta_unsig + inverse_sigmoid(layer_ref_enc)
                    layer_enc_outputs_coord = layer_enc_outputs_coord_unsig.sigmoid()

                    layer_enc_outputs_class = layer_class_embed(layer_hs_enc)
                    enc_outputs_coord.append(layer_enc_outputs_coord)
                    enc_outputs_class.append(layer_enc_outputs_class)

                # enc_delta_unsig = self.enc_bbox_embed(hs_enc[:-1])
                # enc_outputs_unsig = enc_delta_unsig + ref_enc[:-1]
                # enc_outputs_coord = enc_outputs_unsig.sigmoid()
                # enc_outputs_class = self.enc_class_embed(hs_enc[:-1])
                out['enc_outputs'] = [
                    {'pred_logits': a, 'pred_boxes': b} for a, b in zip(enc_outputs_class, enc_outputs_coord)
                ]

        out['dn_meta'] = dn_meta
        return out, targets

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]


class SetCriterion(nn.Module):
    """ This class computes the loss for Conditional DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, matcher, weight_dict, focal_alpha, losses,):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            focal_alpha: alpha in Focal Loss
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_alpha = focal_alpha

    def loss_labels(self, outputs, targets, indices, num_boxes, loss_name, log=True):
        """Classification loss (Binary focal loss)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']
        # import pdb; pdb.set_trace()
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)]) 
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2]+1],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        target_classes_onehot = target_classes_onehot[:,:,:-1]
        # import pdb; pdb.set_trace()
        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha, gamma=2) * src_logits.shape[1]
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes, loss_name):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        # import pdb; pdb.set_trace()
        # print(loss_name)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes, loss_name):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        # import pdb; pdb.set_trace()

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes

        # calculate the x,y and h,w loss
        with torch.no_grad():
            losses['loss_xy'] = loss_bbox[..., :2].sum() / num_boxes
            losses['loss_hw'] = loss_bbox[..., 2:].sum() / num_boxes


        return losses

    def loss_masks(self, outputs, targets, indices, num_boxes, loss_name):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]
        masks = [t["masks"] for t in targets]
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        # upsample predictions to the target size
        src_masks = interpolate(src_masks[:, None], size=target_masks.shape[-2:],
                                mode="bilinear", align_corners=False)
        src_masks = src_masks[:, 0].flatten(1)

        target_masks = target_masks.flatten(1)
        target_masks = target_masks.view(src_masks.shape)
        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, loss_name='dn', **kwargs):
        # print('loss_name', loss_name)
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'masks': self.loss_masks,
            # 'dn_labels': self.loss_dn_labels,
            # 'dn_boxes': self.loss_dn_boxes
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, loss_name=loss_name, **kwargs)

    def forward(self, outputs, targets, return_indices=False):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
            
             return_indices: used for vis. if True, the layer0-5 indices will be returned as well.

        """
        # import pdb; pdb.set_trace()
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}
        device = next(iter(outputs.values())).device
        indices = self.matcher(outputs_without_aux, targets)

        if return_indices:
            indices0_copy = indices
            indices_list = []

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}

        # prepare for dn loss
        dn_meta = outputs['dn_meta']

        if self.training and dn_meta and 'output_known_lbs_bboxes' in dn_meta:
            output_known_lbs_bboxes, single_pad, scalar = self.prep_for_dn(dn_meta)

            dn_pos_idx = []
            dn_neg_idx = []
            dn_target = []
            # import pdb; pdb.set_trace()
            # dn loss limit
            for i in range(len(targets)):
                dn_target_item = {
                    'labels': targets[i]['labels_dn'],
                    'boxes': targets[i]['boxes_dn']
                }
                dn_target.append(dn_target_item)
            for i in range(len(dn_target)):
                if len(dn_target[i]['labels']) > 0:
                    t = torch.range(0, len(dn_target[i]['labels']) - 1).long().cuda()
                    t = t.unsqueeze(0).repeat(scalar, 1)
                    tgt_idx = t.flatten()
                    output_idx = (torch.tensor(range(scalar)) * single_pad).long().cuda().unsqueeze(1) + t
                    output_idx = output_idx.flatten()
                else:
                    output_idx = tgt_idx = torch.tensor([]).long().cuda()

                dn_pos_idx.append((output_idx, tgt_idx))
                dn_neg_idx.append((output_idx + single_pad // 2, tgt_idx))


            output_known_lbs_bboxes = dn_meta['output_known_lbs_bboxes']
            # print('output_known_lbs_bboxes', output_known_lbs_bboxes.keys())
            # print('pred_boxes', output_known_lbs_bboxes['pred_boxes'].shape)
            # print('pred_logits', output_known_lbs_bboxes['pred_logits'].shape)
            # print('aux_outputs',output_known_lbs_bboxes['aux_outputs'][0])
            # exit(0)
            l_dict = {}
            for loss in self.losses:
                kwargs = {}
                if 'labels' in loss:
                    kwargs = {'log': False}
                l_dict.update(self.get_loss(loss, output_known_lbs_bboxes, dn_target, dn_pos_idx, num_boxes*scalar,**kwargs))

            l_dict = {k + f'_dn': v for k, v in l_dict.items()}
            losses.update(l_dict)
        else:
            l_dict = dict()
            l_dict['loss_bbox_dn'] = torch.as_tensor(0.).to('cuda')
            l_dict['loss_giou_dn'] = torch.as_tensor(0.).to('cuda')
            l_dict['loss_ce_dn'] = torch.as_tensor(0.).to('cuda')
            l_dict['loss_xy_dn'] = torch.as_tensor(0.).to('cuda')
            l_dict['loss_hw_dn'] = torch.as_tensor(0.).to('cuda')
            l_dict['cardinality_error_dn'] = torch.as_tensor(0.).to('cuda')
            losses.update(l_dict)


        for loss in self.losses:
            # print(loss)
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes, loss_name='pred'))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for idx, aux_outputs in enumerate(outputs['aux_outputs']):
                # import pdb; pdb.set_trace()
                indices = self.matcher(aux_outputs, targets)
                if return_indices:
                    indices_list.append(indices)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{idx}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

                if self.training and dn_meta and 'output_known_lbs_bboxes' in dn_meta:
                    aux_outputs_known = output_known_lbs_bboxes['aux_outputs'][idx]
                    # import pdb; pdb.set_trace()
                    l_dict={}
                    for loss in self.losses:
                        kwargs = {}
                        if 'labels' in loss:
                            kwargs = {'log': False}

                        l_dict.update(self.get_loss(loss, aux_outputs_known, dn_target, dn_pos_idx, num_boxes*scalar,
                                                                 **kwargs))

                    l_dict = {k + f'_dn_{idx}': v for k, v in l_dict.items()}
                    losses.update(l_dict)
                else:
                    l_dict = dict()
                    l_dict['loss_bbox_dn']=torch.as_tensor(0.).to('cuda')
                    l_dict['loss_giou_dn']=torch.as_tensor(0.).to('cuda')
                    l_dict['loss_ce_dn']=torch.as_tensor(0.).to('cuda')
                    l_dict['loss_xy_dn'] = torch.as_tensor(0.).to('cuda')
                    l_dict['loss_hw_dn'] = torch.as_tensor(0.).to('cuda')
                    l_dict['cardinality_error_dn'] = torch.as_tensor(0.).to('cuda')
                    l_dict = {k + f'_{idx}': v for k, v in l_dict.items()}
                    losses.update(l_dict)


        # interm_outputs loss
        if 'interm_outputs' in outputs:
            interm_outputs = outputs['interm_outputs']
            indices = self.matcher(interm_outputs, targets)
            if return_indices:
                indices_list.append(indices)
            for loss in self.losses:
                if loss == 'masks':
                    # Intermediate masks losses are too costly to compute, we ignore them.
                    continue
                kwargs = {}
                if loss == 'labels':
                    # Logging is enabled only for the last layer
                    kwargs = {'log': False}
                l_dict = self.get_loss(loss, interm_outputs, targets, indices, num_boxes, **kwargs)
                l_dict = {k + f'_interm': v for k, v in l_dict.items()}
                losses.update(l_dict)

        # enc output loss
        if 'enc_outputs' in outputs:
            for i, enc_outputs in enumerate(outputs['enc_outputs']):
                indices = self.matcher(enc_outputs, targets)
                if return_indices:
                    indices_list.append(indices)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, enc_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_enc_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        if return_indices:
            indices_list.append(indices0_copy)
            return losses, indices_list

        return losses, targets

    def prep_for_dn(self,dn_meta):
        output_known_lbs_bboxes = dn_meta['output_known_lbs_bboxes']
        num_dn_groups, pad_size = dn_meta['num_dn_group'],dn_meta['pad_size']
        assert pad_size % num_dn_groups == 0
        single_pad=pad_size//num_dn_groups

        return output_known_lbs_bboxes, single_pad, num_dn_groups


class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""
    def __init__(self, num_select=100, nms_iou_threshold=-1) -> None:
        super().__init__()
        self.num_select = num_select
        self.nms_iou_threshold = nms_iou_threshold

    @torch.no_grad()
    def forward(self, outputs, target_sizes, not_to_xyxy=False, test=False):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        # num_select = self.num_select
        num_select = outputs['pred_logits'].shape[1]
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']
        # import pdb; pdb.set_trace()
        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = out_logits.sigmoid()
        # topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), num_select, dim=1)
        topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), num_select, dim=1)
        scores = topk_values
        topk_boxes = topk_indexes // out_logits.shape[2]
        labels = topk_indexes % out_logits.shape[2]
        if not_to_xyxy:
            boxes = out_bbox
        else:
            boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)

        if test:
            assert not not_to_xyxy
            boxes[:,:,2:] = boxes[:,:,2:] - boxes[:,:,:2]
        boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1,1,4))
        # import pdb; pdb.set_trace()
        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]
        # import pdb; pdb.set_trace()

        if self.nms_iou_threshold > 0:
            item_indices = [nms(b, s, iou_threshold=self.nms_iou_threshold) for b,s in zip(boxes, scores)]
            # import ipdb; ipdb.set_trace()
            results = [{'scores': s[i], 'labels': l[i], 'boxes': b[i]} for s, l, b, i in zip(scores, labels, boxes, item_indices)]
        else:
            results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]
        # print(num_select)
        # return results, topk_boxes
        # import pdb; pdb.set_trace()
        if num_select != self.num_select:
            return results, topk_boxes
        else:
            return results

@MODULE_BUILD_FUNCS.registe_with_name(module_name='dino')
def build_dino(args):
    # the `num_classes` naming here is somewhat misleading.
    # it indeed corresponds to `max_obj_id + 1`, where max_obj_id
    # is the maximum id for a class in your dataset. For example,
    # COCO has a max_obj_id of 90, so we pass `num_classes` to be 91.
    # As another example, for a dataset that has a single class with id 1,
    # you should pass `num_classes` to be 2 (max_obj_id + 1).
    # For more details on this, check the following discussion
    # https://github.com/facebookresearch/detr/issues/108#issuecomment-650269223
    # num_classes = 20 if args.dataset_file != 'coco' else 91
    # if args.dataset_file == "coco_panoptic":
    #     # for panoptic, we just add a num_classes that is large enough to hold
    #     # max_obj_id + 1, but the exact value doesn't really matter
    #     num_classes = 250
    # if args.dataset_file == 'o365':
    #     num_classes = 366
    # if args.dataset_file == 'vanke':
    #     num_classes = 51
    num_classes = args.num_classes
    device = torch.device(args.device)

    backbone = build_backbone(args)
    # template_backbone = build_backbone(args)

    transformer = build_deformable_transformer(args)

    try:
        match_unstable_error = args.match_unstable_error
        dn_labelbook_size = args.dn_labelbook_size
    except:
        match_unstable_error = True
        dn_labelbook_size = num_classes

    try:
        dec_pred_class_embed_share = args.dec_pred_class_embed_share
    except:
        dec_pred_class_embed_share = True
    try:
        dec_pred_bbox_embed_share = args.dec_pred_bbox_embed_share
    except:
        dec_pred_bbox_embed_share = True


    model = DINO(
        backbone,
        # template_backbone,
        transformer,
        num_classes=num_classes,
        num_queries=args.num_queries,
        aux_loss=True,
        iter_update=True,
        query_dim=4,
        random_refpoints_xy=args.random_refpoints_xy,
        fix_refpoints_hw=args.fix_refpoints_hw,
        num_feature_levels=args.num_feature_levels,
        nheads=args.nheads,
        dec_pred_class_embed_share=dec_pred_class_embed_share,
        dec_pred_bbox_embed_share=dec_pred_bbox_embed_share,
        # two stage
        two_stage_type=args.two_stage_type,
        # box_share
        two_stage_bbox_embed_share=args.two_stage_bbox_embed_share,
        two_stage_class_embed_share=args.two_stage_class_embed_share,
        decoder_sa_type=args.decoder_sa_type,
        num_patterns=args.num_patterns,
        dn_number = args.dn_number if args.use_dn else 0,
        dn_box_noise_scale = args.dn_box_noise_scale,
        dn_label_noise_ratio = args.dn_label_noise_ratio,
        dn_labelbook_size = dn_labelbook_size,
        temp_weight = args.temp_weight,
        denoise_query = args.denoise_query,
        template_lvl = args.template_lvl,
        number_template = args.number_template,
        keep_template_look = args.keep_template_look,
    )
    if args.masks:
        model = DETRsegm(model, freeze_detr=(args.frozen_weights is not None))
    matcher = build_matcher(args)

    # prepare weight dict
    weight_dict = {'loss_ce': args.cls_loss_coef, 'loss_bbox': args.bbox_loss_coef}
    weight_dict['loss_giou'] = args.giou_loss_coef
    clean_weight_dict_wo_dn = copy.deepcopy(weight_dict)

    
    # for DN training
    if args.use_dn:
        weight_dict['loss_ce_dn'] = args.cls_loss_coef
        weight_dict['loss_bbox_dn'] = args.bbox_loss_coef
        weight_dict['loss_giou_dn'] = args.giou_loss_coef

    if args.masks:
        weight_dict["loss_mask"] = args.mask_loss_coef
        weight_dict["loss_dice"] = args.dice_loss_coef
    clean_weight_dict = copy.deepcopy(weight_dict)

    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in clean_weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    if args.two_stage_type != 'no':
        interm_weight_dict = {}
        try:
            no_interm_box_loss = args.no_interm_box_loss
        except:
            no_interm_box_loss = False
        _coeff_weight_dict = {
            'loss_ce': 1.0,
            'loss_bbox': 1.0 if not no_interm_box_loss else 0.0,
            'loss_giou': 1.0 if not no_interm_box_loss else 0.0,
        }
        try:
            interm_loss_coef = args.interm_loss_coef
        except:
            interm_loss_coef = 1.0
        interm_weight_dict.update({k + f'_interm': v * interm_loss_coef * _coeff_weight_dict[k] for k, v in clean_weight_dict_wo_dn.items()})
        weight_dict.update(interm_weight_dict)

    losses = ['labels', 'boxes', 'cardinality']
    if args.masks:
        losses += ["masks"]
    criterion = SetCriterion(num_classes, matcher=matcher, weight_dict=weight_dict,
                             focal_alpha=args.focal_alpha, losses=losses
                            )
    criterion.to(device)
    postprocessors = {'bbox': PostProcess(num_select=args.num_select, nms_iou_threshold=args.nms_iou_threshold)}
    if args.masks:
        postprocessors['segm'] = PostProcessSegm()
        if args.dataset_file == "coco_panoptic":
            is_thing_map = {i: i <= 90 for i in range(201)}
            postprocessors["panoptic"] = PostProcessPanoptic(is_thing_map, threshold=0.85)

    return model, criterion, postprocessors
