# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
COCO dataset which returns image_id for evaluation.

Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py
"""
if __name__=="__main__":
    # for debug only
    import os, sys
    sys.path.append(os.path.dirname(sys.path[0]))

import json
from pathlib import Path
import random
import os
import numpy as np
import cv2

import torch
import torch.utils.data
import torchvision
from pycocotools import mask as coco_mask

from datasets.data_util import preparing_dataset
import datasets.transforms as T
from util.box_ops import box_cxcywh_to_xyxy, box_iou
from copy import deepcopy
from collections import Counter
import torchvision.transforms as standard_transforms
import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

from .gmot_wrapper import GMOT40Wrapper

__all__ = ['build']

######################################################
# some hookers for training
class label2compat():
    def __init__(self) -> None:
        self.category_map_str = {"1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9, "10": 10, "11": 11, "13": 12, "14": 13, "15": 14, "16": 15, "17": 16, "18": 17, "19": 18, "20": 19, "21": 20, "22": 21, "23": 22, "24": 23, "25": 24, "27": 25, "28": 26, "31": 27, "32": 28, "33": 29, "34": 30, "35": 31, "36": 32, "37": 33, "38": 34, "39": 35, "40": 36, "41": 37, "42": 38, "43": 39, "44": 40, "46": 41, "47": 42, "48": 43, "49": 44, "50": 45, "51": 46, "52": 47, "53": 48, "54": 49, "55": 50, "56": 51, "57": 52, "58": 53, "59": 54, "60": 55, "61": 56, "62": 57, "63": 58, "64": 59, "65": 60, "67": 61, "70": 62, "72": 63, "73": 64, "74": 65, "75": 66, "76": 67, "77": 68, "78": 69, "79": 70, "80": 71, "81": 72, "82": 73, "84": 74, "85": 75, "86": 76, "87": 77, "88": 78, "89": 79, "90": 80}
        self.category_map = {int(k):v for k,v in self.category_map_str.items()}

    def __call__(self, target, img=None):
        labels = target['labels']
        res = torch.zeros(labels.shape, dtype=labels.dtype)
        for idx, item in enumerate(labels):
            res[idx] = self.category_map[item.item()] - 1
        target['label_compat'] = res
        if img is not None:
            return target, img
        else:
            return target

class label_compat2onehot():
    def __init__(self, num_class=80, num_output_objs=1):
        self.num_class = num_class
        self.num_output_objs = num_output_objs
        if num_output_objs != 1:
            raise DeprecationWarning("num_output_objs!=1, which is only used for comparison")

    def __call__(self, target, img=None):
        labels = target['label_compat']
        place_dict = {k:0 for k in range(self.num_class)}
        if self.num_output_objs == 1:
            res = torch.zeros(self.num_class)
            for i in labels:
                itm = i.item()
                res[itm] = 1.0
        else:
            # compat with baseline
            res = torch.zeros(self.num_class, self.num_output_objs)
            for i in labels:
                itm = i.item()
                res[itm][place_dict[itm]] = 1.0
                place_dict[itm] += 1
        target['label_compat_onehot'] = res
        if img is not None:
            return target, img
        else:
            return target

class box_label_catter():
    def __init__(self):
        pass

    def __call__(self, target, img=None):
        labels = target['label_compat']
        boxes = target['boxes']
        box_label = torch.cat((boxes, labels.unsqueeze(-1)), 1)
        target['box_label'] = box_label
        if img is not None:
            return target, img
        else:
            return target

def label2onehot(label, num_classes):
    """
    label: Tensor(K)
    """
    res = torch.zeros(num_classes)
    for i in label:
        itm = int(i.item())
        res[itm] = 1.0
    return res

class RandomSelectBoxlabels():
    def __init__(self, num_classes, leave_one_out=False, blank_prob=0.8,
                    prob_first_item = 0.0,
                    prob_random_item = 0.0,
                    prob_last_item = 0.8,
                    prob_stop_sign = 0.2
                ) -> None:
        self.num_classes = num_classes
        self.leave_one_out = leave_one_out
        self.blank_prob = blank_prob

        self.set_state(prob_first_item, prob_random_item, prob_last_item, prob_stop_sign)
        

    def get_state(self):
        return [self.prob_first_item, self.prob_random_item, self.prob_last_item, self.prob_stop_sign]

    def set_state(self, prob_first_item, prob_random_item, prob_last_item, prob_stop_sign):
        sum_prob = prob_first_item + prob_random_item + prob_last_item + prob_stop_sign
        assert sum_prob - 1 < 1e-6, \
            f"Sum up all prob = {sum_prob}. prob_first_item:{prob_first_item}" \
            + f"prob_random_item:{prob_random_item}, prob_last_item:{prob_last_item}" \
            + f"prob_stop_sign:{prob_stop_sign}"

        self.prob_first_item = prob_first_item
        self.prob_random_item = prob_random_item
        self.prob_last_item = prob_last_item
        self.prob_stop_sign = prob_stop_sign
        

    def sample_for_pred_first_item(self, box_label: torch.FloatTensor):
        box_label_known = torch.Tensor(0,5)
        box_label_unknown = box_label
        return box_label_known, box_label_unknown

    def sample_for_pred_random_item(self, box_label: torch.FloatTensor):
        n_select = int(random.random() * box_label.shape[0])
        box_label = box_label[torch.randperm(box_label.shape[0])]
        box_label_known = box_label[:n_select]
        box_label_unknown = box_label[n_select:]
        return box_label_known, box_label_unknown

    def sample_for_pred_last_item(self, box_label: torch.FloatTensor):
        box_label_perm = box_label[torch.randperm(box_label.shape[0])]
        known_label_list = []
        box_label_known = []
        box_label_unknown = []
        for item in box_label_perm:
            label_i = item[4].item()
            if label_i in known_label_list:
                box_label_known.append(item)
            else:
                # first item
                box_label_unknown.append(item)
                known_label_list.append(label_i)
        box_label_known = torch.stack(box_label_known) if len(box_label_known) > 0 else torch.Tensor(0,5)
        box_label_unknown = torch.stack(box_label_unknown) if len(box_label_unknown) > 0 else torch.Tensor(0,5)
        return box_label_known, box_label_unknown

    def sample_for_pred_stop_sign(self, box_label: torch.FloatTensor):
        box_label_unknown = torch.Tensor(0,5)
        box_label_known = box_label
        return box_label_known, box_label_unknown

    def __call__(self, target, img=None):
        box_label = target['box_label'] # K, 5

        dice_number = random.random()

        if dice_number < self.prob_first_item:
            box_label_known, box_label_unknown = self.sample_for_pred_first_item(box_label)
        elif dice_number < self.prob_first_item + self.prob_random_item:
            box_label_known, box_label_unknown = self.sample_for_pred_random_item(box_label)
        elif dice_number < self.prob_first_item + self.prob_random_item + self.prob_last_item:
            box_label_known, box_label_unknown = self.sample_for_pred_last_item(box_label)
        else:
            box_label_known, box_label_unknown = self.sample_for_pred_stop_sign(box_label)

        target['label_onehot_known'] = label2onehot(box_label_known[:,-1], self.num_classes)
        target['label_onehot_unknown'] = label2onehot(box_label_unknown[:, -1], self.num_classes)
        target['box_label_known'] = box_label_known
        target['box_label_unknown'] = box_label_unknown

        return target, img


class RandomDrop():
    def __init__(self, p=0.2) -> None:
        self.p = p

    def __call__(self, target, img=None):
        known_box = target['box_label_known']
        num_known_box = known_box.size(0)
        idxs = torch.rand(num_known_box)
        # indices = torch.randperm(num_known_box)[:int((1-self).p*num_known_box + 0.5 + random.random())]
        target['box_label_known'] = known_box[idxs > self.p]
        return target, img


class BboxPertuber():
    def __init__(self, max_ratio = 0.02, generate_samples = 1000) -> None:
        self.max_ratio = max_ratio
        self.generate_samples = generate_samples
        self.samples = self.generate_pertube_samples()
        self.idx = 0

    def generate_pertube_samples(self):
        import torch
        samples = (torch.rand(self.generate_samples, 5) - 0.5) * 2 * self.max_ratio
        return samples

    def __call__(self, target, img):
        known_box = target['box_label_known'] # Tensor(K,5), K known bbox
        K = known_box.shape[0]
        known_box_pertube = torch.zeros(K, 6) # 4:bbox, 1:prob, 1:label
        if K == 0:
            pass
        else:
            if self.idx + K > self.generate_samples:
                self.idx = 0
            delta = self.samples[self.idx: self.idx + K, :]
            known_box_pertube[:, :4] = known_box[:, :4] + delta[:, :4]
            iou = (torch.diag(box_iou(box_cxcywh_to_xyxy(known_box[:, :4]), box_cxcywh_to_xyxy(known_box_pertube[:, :4]))[0])) * (1 + delta[:, -1])
            known_box_pertube[:, 4].copy_(iou)
            known_box_pertube[:, -1].copy_(known_box[:, -1])

        target['box_label_known_pertube'] = known_box_pertube
        return target, img


class RandomCutout():
    def __init__(self, factor=0.5) -> None:
        self.factor = factor

    def __call__(self, target, img=None):
        unknown_box = target['box_label_unknown']           # Ku, 5
        known_box = target['box_label_known_pertube']       # Kk, 6
        Ku = unknown_box.size(0)

        known_box_add = torch.zeros(Ku, 6) # Ku, 6
        known_box_add[:, :5] = unknown_box
        known_box_add[:, 5].uniform_(0.5, 1) 
        

        known_box_add[:, :2] += known_box_add[:, 2:4] * (torch.rand(Ku, 2) - 0.5) / 2
        known_box_add[:, 2:4] /= 2

        target['box_label_known_pertube'] = torch.cat((known_box, known_box_add))
        return target, img




class RandomSelectBoxes():
    def __init__(self, num_class=80) -> None:
        Warning("This is such a slow function and will be deprecated soon!!!")
        self.num_class = num_class

    def __call__(self, target, img=None):
        boxes = target['boxes']
        labels = target['label_compat']

        # transform to list of tensors
        boxs_list = [[] for i in range(self.num_class)]
        for idx, item in enumerate(boxes):
            label = labels[idx].item()
            boxs_list[label].append(item)
        boxs_list_tensor = [torch.stack(i) if len(i) > 0 else torch.Tensor(0,4) for i in boxs_list]

        # random selection
        box_known = []
        box_unknown = []
        for idx, item in enumerate(boxs_list_tensor):
            ncnt = item.shape[0]
            nselect = int(random.random() * ncnt) # close in both sides, much faster than random.randint
            # import ipdb; ipdb.set_trace()
            item = item[torch.randperm(ncnt)]
            # random.shuffle(item)
            box_known.append(item[:nselect])
            box_unknown.append(item[nselect:])
        # import ipdb; ipdb.set_trace()
        # box_known_tensor = [torch.stack(i) if len(i) > 0 else torch.Tensor(0,4) for i in box_known]
        # box_unknown_tensor = [torch.stack(i) if len(i) > 0 else torch.Tensor(0,4) for i in box_unknown]
        # print('box_unknown_tensor:', box_unknown_tensor)
        target['known_box'] = box_known
        target['unknown_box'] = box_unknown
        return target, img






        


# class BoxCatter():
#     def __init__(self) -> None:
#         pass

#     def __call__(self, target, img):
#         """
#         known_box_cat:
#             - Tensor(k*5), 
#                 * Tensor[:, :4]: bbox,  
#                 * Tensor[:, -1]: label
#         """
#         known_box = target['known_box']
#         boxes_list = []
#         for idx, boxes in enumerate(known_box):
#             nbox = boxes.shape[0]
#             boxes_idx = torch.cat([boxes, torch.Tensor([idx] * nbox).unsqueeze(1)], 1)
#             boxes_list.append(boxes_idx)
#         known_box_cat = torch.cat(boxes_list, 0)
#         target['known_box_cat'] = known_box_cat
#         return target, img
        

class MaskCrop():
    def __init__(self) -> None:
        pass

    def __call__(self, target, img):
        known_box = target['known_box']
        h,w = img.shape[1:] # h,w
        # imgsize = target['orig_size'] # h,w
        # import ipdb; ipdb.set_trace()
        scale = torch.Tensor([w, h, w, h])

        # _cnt = 0
        for boxes in known_box:
            if boxes.shape[0] == 0:
                continue
            box_xyxy = box_cxcywh_to_xyxy(boxes) * scale
            for box in box_xyxy:
                x1, y1, x2, y2 = [int(i) for i in box.tolist()]
                img[:, y1:y2, x1:x2] = 0
                # _cnt += 1
        # print("_cnt:", _cnt)
        return target, img


        

dataset_hook_register = {
    'label2compat': label2compat,
    'label_compat2onehot': label_compat2onehot,
    'box_label_catter': box_label_catter,
    'RandomSelectBoxlabels': RandomSelectBoxlabels,
    'RandomSelectBoxes': RandomSelectBoxes,
    'MaskCrop': MaskCrop,
    'BboxPertuber': BboxPertuber,
}
                
##################################################################################
class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, ann_file, transforms, return_masks, image_set, aux_target_hacks=None,
                 num_imgs=None, number_template=None):
        super(CocoDetection, self).__init__(img_folder, ann_file)
        # import pdb; pdb.set_trace()
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks, image_set)
        self.aux_target_hacks = aux_target_hacks
        self.num_imgs = num_imgs
        self.class_dict = {}
        # if image_set in ['train', 'val']:
        #     self.class_weight = {}
        if image_set in ['test', 'test_ov', 'train', 'val']:
            self.template_list = {}
        # if image_set == 'train':
        #     self.ids = sorted(self.ids)
        self.image_set = image_set
        self.number_template = number_template
        self.count1 = 0

        if self.num_imgs is not None:
            self.ids = sorted(self.ids)
            self.ids = self.ids[0:self.num_imgs]

        ids1 = []
        for i in self.ids:
            target = self._load_target(i)
            if len(target) > 0:
                ids1.append(i)
            # if len(target) == 0:
            #     self.ids.remove(i)
        self.ids = ids1

    def change_hack_attr(self, hackclassname, attrkv_dict):
        target_class = dataset_hook_register[hackclassname]
        for item in self.aux_target_hacks:
            if isinstance(item, target_class):
                for k,v in attrkv_dict.items():
                    setattr(item, k, v)

    def get_hack(self, hackclassname):
        target_class = dataset_hook_register[hackclassname]
        for item in self.aux_target_hacks:
            if isinstance(item, target_class):
                return item

    def get_class_id_to_img_id(self):
        
        for i in self.ids:
            target = self._load_target(i)
            for item in target:
                category_id = item['category_id']
                # if category_id > 1000:
                #     import pdb; pdb.set_trace()
                if category_id not in self.class_dict.keys():
                    self.class_dict[category_id] = [i]
                else:
                    self.class_dict[category_id].append(i)
        for category in self.class_dict.keys():
            self.class_dict[category] = list(set(self.class_dict[category]))

    def get_sequence_id_to_img_id(self):
        for i in self.ids:
            path = self.coco.loadImgs(i)[0]["file_name"]
            sequence_name = path.split('/')[1]
            image_name = path.split('/')[-1]
            if sequence_name not in self.class_dict.keys():
                self.class_dict[sequence_name] = {'image':[], 'template':[]}
                self.class_dict[sequence_name]['image'].append(i)
            else:
                self.class_dict[sequence_name]['image'].append(i)
            
            if image_name == '000000.jpg':
                self.class_dict[sequence_name]['template_img'] = i
                self.class_dict[sequence_name]['template'] = self._load_target(i)
            # print(path)
            # exit(0)
        
    def get_ov_template(self):
        template_number = 3
        # import pdb; pdb.set_trace()
        for key in self.class_dict.keys():
            templates = []
            # image_id = self.class_dict[key][0]
            for image_id in self.class_dict[key]:
                idx = self.ids.index(image_id)
                img, target = super(CocoDetection, self).__getitem__(idx)
                # import pdb; pdb.set_trace()
                for item in target:
                    if item['category_id'] == key and item['bbox'][2] > 25 and item['bbox'][3] > 25:
                        box = deepcopy(item['bbox'])
                        box[2] += box[0]
                        box[3] += box[1]
                        template = img.crop(box)
                        template_path = 'ov_vis/templates/template_' + str(key) + '_' + str(image_id) +  '.jpg'
                        template.save(template_path)
                        # import pdb; pdb.set_trace()
                        template, _ = T.resize(template, target=None, size=400, max_size=400)
                        tran_template = T.Compose([ 
                            T.ToTensor(),
                            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                        ])
                        template, _ = tran_template(template, target)
                        # from torchvision import transforms
                        # unloader = transforms.ToPILImage()
                        # image = template.cpu().clone()  # clone the tensor
                        # image = image.squeeze(0)  # remove the fake batch dimension
                        # image = unloader(image)
                        # name = 'ov_vis/template/example_' + str(key) + '_' + str(image_id) + '.jpg'
                        # image.save(name)
                        templates.append(template)
                        # self.template_list[key] = [template]
                        # import pdb; pdb.set_trace()
                        break
                    else:
                        pass
                if len(templates) == template_number:
                    self.template_list[key] = [templates[-1]]
                    break

    def get_class_weight(self):
        category_list = []
        for i in self.ids:
            target = self._load_target(i)
            category_list.extend([x['category_id'] for x in target])
        counter = Counter(category_list)

        # draw class hist in coco
        # x_value = [int(x) for x in counter.keys()]
        # y_value = [counter[x]/len(category_list) for x in counter.keys()]
        # plt.bar(x_value, y_value, width=0.8, bottom=None)
        # plt.savefig('template/coco_class_hist.png')
        # exit(0)
        # import pdb; pdb.set_trace()
        for i in counter:
            self.class_weight[i] = 1/(counter[i]/len(category_list))

    def __len__(self):
        # return 100
        return len(self.ids)

    def __getitem__(self, idx):
        if self.image_set == 'test':
            return self.getitem_test(idx)
        elif self.image_set == 'panda':
            return self.getitem_panda(idx)
        elif self.image_set == 'test_ov':
            return self.getitem_test_ov(idx)
        elif self.image_set == 'train_cur':
            # use template in current image
            return self.get_item_ablation2(idx)
        elif self.number_template == 1:
            # no background class
            return self.get_item_ablation(idx)
        else:
            return self.getitem_train(idx)
    
    def get_item_ablation(self, idx):
        try:
            img, target = super(CocoDetection, self).__getitem__(idx)
            while len(target) <= 0:
                idx += 1
                img, target = super(CocoDetection, self).__getitem__(idx)
        except:
            print("Error idx: {}".format(idx))
            idx += 1
            img, target = super(CocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        # print('template image id:', image_id)
        for item in target:
            item['template_id'] = 0
        target = {'image_id': image_id, 'annotations': target}
        # --------------------------------------------------------------------------------------------------------------
        img, target = self.prepare(img, target)
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        # --------------------------------------------------------------------------------------------------------------
        # import pdb; pdb.set_trace()
        num = len(target['labels'])
        while num == 0:
            try:
                img, target = super(CocoDetection, self).__getitem__(idx)
                while len(target) <= 0:
                    idx += 1
                    img, target = super(CocoDetection, self).__getitem__(idx)
            except:
                print("Error idx: {}".format(idx))
                idx += 1
                img, target = super(CocoDetection, self).__getitem__(idx)
            image_id = self.ids[idx]
            # print('template image id:', image_id)
            for item in target:
                item['template_id'] = 0
            target = {'image_id': image_id, 'annotations': target}
            # --------------------------------------------------------------------------------------------------------------
            img, target = self.prepare(img, target)
            if self._transforms is not None:
                img, target = self._transforms(img, target)
            # --------------------------------------------------------------------------------------------------------------
            num = len(target['labels'])
        template_list = []
        temp_cls_list = []
        if num == 1:
            temp_idx = 0
        else:
            temp_idx = np.random.randint(0, num - 1)
        template_class = target['labels'][temp_idx].item()
        temp_cls_list.append(template_class)
        new_img_id = random.choice(self.class_dict[template_class])
        new_idx = self.ids.index(new_img_id)
        new_idx = self.ids.index(new_img_id)
        new_img, new_img_target = super(CocoDetection, self).__getitem__(new_idx)
        template_anno = list(filter(lambda item: item['category_id'] == template_class, new_img_target))
        
        if len(template_anno) > 1:
            box = deepcopy(template_anno[np.random.randint(0, max(len(template_anno) - 1, 0))]['bbox'])
        else:
            box = deepcopy(template_anno[0]['bbox'])
        box[2] = box[0] + max(1, box[2])
        box[3] = box[1] + max(box[3], 1)
        template = new_img.crop(box)
        template_list.append(template)
        
        return_template_list = []
        tran_template = T.Compose([ 
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        for template in template_list:    
            template, _ = T.resize(template, target=None, size=400, max_size=400)
            template, _ = tran_template(template, target)
            return_template_list.append(template)
        if self.aux_target_hacks is not None:
            for hack_runner in self.aux_target_hacks:
                target, img = hack_runner(target, img=img)

        # split target into bs*template_num
        new_targets = []
        for num in range(self.number_template):
            new_t = {}
            # import pdb; pdb.set_trace()
            keep = target['labels']==template_class
            # import pdb; pdb.set_trace()
            for key in target.keys():
                if key in ['boxes', 'labels', 'iscrowd', 'area', 'template_id']:
                    new_t[key] = target[key][keep]
                    # targets[key] = targets[key][keep]
                else:
                    new_t[key] = target[key]
            new_targets.append(new_t)
        for tgt in new_targets:
            tgt['labels'] = torch.ones_like(tgt['labels'])
        target = new_targets
        # import pdb; pdb.set_trace()

        return img, target, return_template_list, self.number_template, temp_cls_list

    def get_item_ablation2(self, idx):
        try:
            img, target = super(CocoDetection, self).__getitem__(idx)
            while len(target) <= 0:
                idx += 1
                img, target = super(CocoDetection, self).__getitem__(idx)
        except:
            print("Error idx: {}".format(idx))
            idx += 1
            img, target = super(CocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        # print('template image id:', image_id)
        for item in target:
            item['template_id'] = 0
        target = {'image_id': image_id, 'annotations': target}
        # --------------------------------------------------------------------------------------------------------------
        img, target = self.prepare(img, target)
        image = img
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        # --------------------------------------------------------------------------------------------------------------
        # import pdb; pdb.set_trace()
        num = len(target['labels'])
        while num == 0:
            idx += 1
            img, target = super(CocoDetection, self).__getitem__(idx)
            image_id = self.ids[idx]
            # print('template image id:', image_id)
            for item in target:
                item['template_id'] = 0
            target = {'image_id': image_id, 'annotations': target}
            # --------------------------------------------------------------------------------------------------------------
            img, target = self.prepare(img, target)
            if self._transforms is not None:
                img, target = self._transforms(img, target)
            # --------------------------------------------------------------------------------------------------------------
            num = len(target['labels'])
        template_list = []
        temp_cls_list = []
        if num == 1:
            temp_idx = 0
        else:
            temp_idx = np.random.randint(0, num - 1)
        template_class = target['labels'][temp_idx].item()
        temp_cls_list.append(template_class)
        template_anno = deepcopy(target['boxes']).tolist()
        if len(template_anno) > 1:
            box = deepcopy(template_anno[np.random.randint(0, max(len(template_anno) - 1, 0))])
        else:
            box = deepcopy(template_anno[0])
        size = target['size'].tolist()
        # import pdb; pdb.set_trace()
        box[0] *= size[1]
        box[1] *= size[0]
        box[2] *= size[1]
        box[3] *= size[0]

        box[0] -= box[2]/2
        box[1] -= box[3]/2
        box[2] += box[0]
        box[3] += box[1]
        # import pdb; pdb.set_trace()
        template = image.crop(box)
        template_list.append(template)
        
        return_template_list = []
        tran_template = T.Compose([ 
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        for template in template_list:    
            template, _ = T.resize(template, target=None, size=400, max_size=400)
            template, _ = tran_template(template, target)
            return_template_list.append(template)
        if self.aux_target_hacks is not None:
            for hack_runner in self.aux_target_hacks:
                target, img = hack_runner(target, img=img)

        # split target into bs*template_num
        new_targets = []
        for num in range(self.number_template):
            new_t = {}
            # import pdb; pdb.set_trace()
            keep = target['template_id']==num
            # import pdb; pdb.set_trace()
            for key in target.keys():
                if key in ['boxes', 'labels', 'iscrowd', 'area', 'template_id']:
                    new_t[key] = target[key][keep]
                    # targets[key] = targets[key][keep]
                else:
                    new_t[key] = target[key]
            new_targets.append(new_t)
        target = new_targets
        new_label = []
        for label in target[0]['labels']:
            if label == template_class:
                new_label.append(1)
            else:
                new_label.append(0)
        target[0]['labels'] = torch.tensor(new_label)
        # import pdb; pdb.set_trace()
        return img, target, return_template_list, self.number_template, temp_cls_list

    def getitem_test_ov(self, idx):
        try:
            img, target = super(CocoDetection, self).__getitem__(idx)
        except:
            print("Error idx: {}".format(idx))
            idx += 1
            img, target = super(CocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}

        img, target = self.prepare(img, target)
        if self._transforms is not None:
            img, target = self._transforms(img, target)

        if self.aux_target_hacks is not None:
            for hack_runner in self.aux_target_hacks:
                target, img = hack_runner(target, img=img)
        
        return img, target

    def getitem_test(self, idx):
        try:
            img, target = super(CocoDetection, self).__getitem__(idx)
        except:
            print("Error idx: {}".format(idx))
            idx += 1
            img, target = super(CocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        # print('template image id:', image_id)
        target = {'image_id': image_id, 'annotations': target}

        for key in self.class_dict.keys():
            if self.class_dict[key]['image'].count(image_id):
                # print('sequence:', key)
                if key not in self.template_list.keys():
                    # random.seed(2)
                    # template_idx = random.randint(0, len(self.class_dict[key]['template'])-1)
                    # template group 1
                    # template = Image.open('template/gmot1/' + key +'.jpg')

                    # template group 2
                    # template = Image.open('template/gmot2/' + key +'.jpg')
                    
                    # template group 3
                    # template = Image.open('template/gmot3/' + key +'.jpg')

                    # template group track
                    template = Image.open('template/gmot_track/' + key +'.jpg')

                    # save template id
                    # with open('template/group3.txt', 'a') as f:
                    #     f.write(key + ':' + str(template_idx) + '\n')

                    # box = self.class_dict[key]['template'][template_idx]['bbox']
                    # box[2] = box[0] + max(1, box[2])
                    # box[3] = box[1] + max(box[3], 1)
                    # # print(box)
                    # temp_image_id = self.class_dict[key]['template_img']
                    # template_img = self._load_image(temp_image_id)
                    # template = template_img.crop(box)
                    # template.save('template/gmot3/'+key+'.jpg')

                    self.template_list[key] = template
                else:
                    template = self.template_list[key]


        img, target = self.prepare(img, target)
        if self._transforms is not None:
            img, target = self._transforms(img, target)

            template, _ = T.resize(template, target=None, size=400, max_size=400)
            tran_template = T.Compose([ 
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
            template, _ = tran_template(template, target)

        if self.aux_target_hacks is not None:
            for hack_runner in self.aux_target_hacks:
                target, img = hack_runner(target, img=img)

        # print('sample:', img.shape)
        # print('template:', template.shape)
        return img, target, [template]
    
    def getitem_train(self, idx):
        """
        Output:
            - target: dict of multiple items
                - boxes: Tensor[num_box, 4]. \
                    Init type: x0,y0,x1,y1. unnormalized data.
                    Final type: cx,cy,w,h. normalized data. 
        """
        try:
            img, target = super(CocoDetection, self).__getitem__(idx)
            while len(target) <= 0:
                idx += 1
                img, target = super(CocoDetection, self).__getitem__(idx)
        except:
            print("Error idx: {}".format(idx))
            idx += 1
            img, target = super(CocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        # print('template image id:', image_id)
        target = {'image_id': image_id, 'annotations': target}
        # import pdb; pdb.set_trace()
        # --------------------------------------------------------------------------------------------------------------
        # print(img)
        # print('getitem-target:', target['image_id'])
        # print(target['annotations'])
        # --------------------------------------------------------------------------------------------------------------
        # template
        num = len(target['annotations'])
        # print('*****num:{}*****'.format(num))
        template_list = []
        targets_list = []
        temp_cls_list = []
        # print(num)

        if self.number_template == 1:
            num_template = 1
            if num == 1:
                temp_idx = 0
            else:
                # category_list = [x['category_id'] for x in target['annotations']]
                # choise_list = torch.tensor(list(set(category_list)))
                # category_list = torch.tensor(category_list)
                # weights = torch.tensor([self.class_weight[x] for x in choise_list.tolist()])
                # template_class = choise_list[torch.multinomial(weights, self.number_template, replacement=False)].item()
                # if template_class == 1:
                #     self.count1 += 1
                #     print(self.count1)
                # category_mask = (category_list == template_class).int()
                # same_category_template_num = sum(category_mask)
                # nth, index_list = torch.topk(category_mask, same_category_template_num)
                # temp_idx = random.choice(index_list.tolist())

                temp_idx = np.random.randint(0, num - 1)
                choise_num = 0
                while min(target['annotations'][temp_idx]['bbox'][2], target['annotations'][temp_idx]['bbox'][3]) <= 1 and choise_num < 5:
                    temp_idx = np.random.randint(0, num - 1)
                    choise_num += 1
            template_class = target['annotations'][temp_idx]['category_id']
            box = deepcopy(target['annotations'][temp_idx]['bbox'])
            # print(box)
            
            box[2] = box[0] + max(1, box[2])
            box[3] = box[1] + max(box[3], 1)
            template = img.crop(box)
            template_list.append(template)
            # print('template:', template.size)

            # find an image with the same class object
            new_img_id = random.choice(self.class_dict[template_class])
            new_idx = self.ids.index(new_img_id)
            # load new image and target
            try:
                img, target = super(CocoDetection, self).__getitem__(new_idx)
            except:
                print("Error idx: {}".format(idx))
                idx += 1
                img, target = super(CocoDetection, self).__getitem__(new_idx)
            image_id = new_img_id

            # print('sample image id:', image_id)
            target_2class = deepcopy(target)
            for item in target_2class:
                if item['category_id'] == template_class:
                    item['category_id'] = 1
                    item['template_id'] = 0
                else:
                    item['category_id'] = 0
                    item['template_id'] = 0
                targets_list.append(item)
            # target = {'image_id': image_id, 'annotations': target_2class} 

        # use more than 1 template
        if self.number_template > 1:
            temp_ann = []
            while len(temp_ann) < self.number_template:
                temp_ann += deepcopy(target['annotations'])
            # temp_ann = target['annotations']
            num_template = min(self.number_template, len(temp_ann))
            # category_list = [x['category_id'] for x in temp_ann]
            # choise_list = torch.tensor(list(set(category_list)))
            # category_list = torch.tensor(category_list)
            # weights = torch.tensor([self.class_weight[x] for x in choise_list.tolist()])
            # if len(choise_list) > 1:
            #     template_class_list = torch.multinomial(weights, num_template, replacement=False).tolist()
            # else:
            #     template_class_list = torch.multinomial(weights, num_template, replacement=True).tolist()
            
            # load another template
            for _ in range(num_template):

                # choose other template category with higher chance
                # template_class = choise_list[template_class_list[_]].item()
                # category_mask = (category_list == template_class).int()
                # same_category_template_num = sum(category_mask)
                # nth, index_list = torch.topk(category_mask, same_category_template_num)
                # temp_idx = random.choice(index_list.tolist())
                temp_idx = random.randint(0, len(temp_ann)-1)
                choise_num = 0
                while min(temp_ann[temp_idx]['bbox'][2], temp_ann[temp_idx]['bbox'][3]) <= 1 and choise_num < 5:
                    # temp_idx = random.choice(index_list.tolist())
                    temp_idx = random.randint(0, len(temp_ann)-1)
                    choise_num += 1
                template_class = temp_ann[temp_idx]['category_id']
                # if template_class > 1000:
                #     import pdb; pdb.set_trace()
                temp_cls_list.append(template_class)
                # print('temp_idx:{},template_class:{}'.format(temp_idx, template_class))
                new_img_id = random.choice(self.class_dict[template_class])
                new_idx = self.ids.index(new_img_id)
                new_img, new_img_target = super(CocoDetection, self).__getitem__(new_idx)
                template_anno = list(filter(lambda item: item['category_id'] == template_class, new_img_target))
                if len(template_anno) > 1:
                    box = deepcopy(template_anno[np.random.randint(0, max(len(template_anno) - 1, 0))]['bbox'])
                    # box = template_anno[np.random.randint(0, max(len(template_anno) - 1, 0))]['bbox']
                else:
                    box = deepcopy(template_anno[0]['bbox'])
                    # box = template_anno[0]['bbox']
                box[2] = box[0] + max(1, box[2])
                box[3] = box[1] + max(box[3], 1)
                template = new_img.crop(box)
                # template_show = False
                # if template_show:
                #     template_path = 'template/coco/template_' + str(_) + '.jpg'
                #     template.save(template_path)
                template_list.append(template)
                # import pdb; pdb.set_trace()
                # change target
                target_2class = deepcopy(target['annotations'])
                for item in target_2class:
                    if item['category_id'] == template_class:
                        item['category_id'] = 1
                        item['template_id'] = _
                    else:
                        item['category_id'] = 0
                        item['template_id'] = _
                    targets_list.append(item)

            # print('targets_list:', len(targets_list))
        # import pdb; pdb.set_trace()
        target = {'image_id': image_id, 'annotations': targets_list}
        # --------------------------------------------------------------------------------------------------------------
        # import pdb; pdb.set_trace()
        img, target = self.prepare(img, target)
        # import pdb; pdb.set_trace()
        # print('target:', target)
        # exit(0)
        if self._transforms is not None:
            img, target = self._transforms(img, target)
            # print('target2:', len(target['boxes']))
            # ----------------------------------------------------------------------------------------------------------   
            return_template_list = []
            tran_template = T.Compose([ 
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
            for template in template_list:    
                template, _ = T.resize(template, target=None, size=400, max_size=400)
                # print(template)
                template, _ = tran_template(template, target)
                # print('template:', template.shape)
                return_template_list.append(template)
            # ----------------------------------------------------------------------------------------------------------
            # print(img.shape)
            # print(template.shape)
            # ----------------------------------------------------------------------------------------------------------
        # print('target3:', len(target['boxes']))
        # convert to needed format
        if self.aux_target_hacks is not None:
            for hack_runner in self.aux_target_hacks:
                target, img = hack_runner(target, img=img)

        # split target into bs*template_num
        new_targets = []
        for num in range(num_template):
            new_t = {}
            # import pdb; pdb.set_trace()
            keep = target['template_id']==num
            # import pdb; pdb.set_trace()
            for key in target.keys():
                if key in ['boxes', 'labels', 'iscrowd', 'area', 'template_id']:
                    new_t[key] = target[key][keep]
                    # targets[key] = targets[key][keep]
                else:
                    new_t[key] = target[key]
            new_targets.append(new_t)
        target = new_targets
        # # 如果只有一个gt,看看这里的target长度是不是1
        # import pdb; pdb.set_trace()        from PIL import ImageDraw
        # def tensor_to_img(x):
        #     mean = torch.tensor([0.485, 0.456, 0.406]).view(1,1,3)
        #     std = torch.tensor([0.229, 0.224, 0.225]).view(1,1,3)
        #     x = ((x.permute(1,2,0) * std) + mean) * 255
        #     x = np.array(x.clamp(0, 255), np.uint8)
        #     x = cv2.cvtColor(x,cv2.COLOR_RGB2BGR)
        #     return x
        
        # for _ in range(num_template):
        #     image = tensor_to_img(img)
        #     image_path = 'ov_vis/train_vis/' + '_' + str(image_id) +  '.jpg'
        #     boxes = target[_]['boxes']
        #     boxes[:, 0] *= target[_]['size'][1]
        #     boxes[:, 1] *= target[_]['size'][0]
        #     boxes[:, 2] *= target[_]['size'][1]
        #     boxes[:, 3] *= target[_]['size'][0]

        #     for i, box in enumerate(boxes):
        #         if target[_]['labels'][i] == 1:
        #             cv2.rectangle(image, (int(box[0]-box[2]/2), int(box[1]-box[3]/2)), (int(box[2]/2+box[0]), int(box[3]/2+box[1])), (0,255,0), 2)
        #         # cv2.putText(image, str(int(target[_]['labels'][i])), (int(box[0]-box[2]/2), int(box[1]-box[3]/2)), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
        #         else:
        #             cv2.rectangle(image, (int(box[0]-box[2]/2), int(box[1]-box[3]/2)), (int(box[2]/2+box[0]), int(box[3]/2+box[1])), (0,0,255), 2)
        #     cv2.imwrite(image_path, image)
        #     # import pdb; pdb.set_trace()
        #     temp_image = tensor_to_img(return_template_list[_])
        #     name = 'ov_vis/train_template/example_' + '_' + str(image_id) + '.jpg'
        #     cv2.imwrite(name, temp_image)
        # import pdb; pdb.set_trace()
        return img, target, return_template_list, num_template, temp_cls_list

    def getitem_panda(self, idx):
        try:
            img, target = super(CocoDetection, self).__getitem__(idx)
        except:
            print("Error idx: {}".format(idx))
            idx += 1
            img, target = super(CocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        # print('template image id:', image_id)
        target = {'image_id': image_id, 'annotations': target}

        template_anno = random.choice(target['annotations'])
        box = deepcopy(template_anno['bbox'])
        while box[2] < 0 or box[3] < 0:
            template_anno = random.choice(target['annotations'])
            box = deepcopy(template_anno['bbox'])
        box[2] += box[0]
        box[3] += box[1]
        template = img.crop(box)

        image, target = self.prepare(img, target)
        if self._transforms is not None:
            image, target = self._transforms(img, target)

            template, _ = T.resize(template, target=None, size=400, max_size=400)
            tran_template = T.Compose([ 
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
            # img, _ = tran_template(img, target)
            template, _ = tran_template(template, target)
            # import pdb; pdb.set_trace()

        if self.aux_target_hacks is not None:
            for hack_runner in self.aux_target_hacks:
                target, img = hack_runner(target, img=img)

        return image, target, [template], img

def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertCocoPolysToMask(object): 
    def __init__(self, return_masks=False, image_set='train'):
        self.return_masks = return_masks
        self.image_set = image_set

    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        boxes = [obj["bbox"] for obj in anno]

        if self.image_set in ['train', 'train_ov', 'val_ov', 'val', 'train_cur', 'train_coco_lasot_got', 
                              'val_coco_lasot_got', 'train_o365', 'val_o365']:
            template = [obj["template_id"] for obj in anno]
            template = torch.tensor(template, dtype=torch.int64)
        # print('template_id', template)
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        if self.image_set in ['train', 'train_ov', 'val_ov', 'val', 'train_cur', 'train_coco_lasot_got', 
                              'val_coco_lasot_got', 'train_o365', 'val_o365']:
            template = template[keep]
        # import pdb; pdb.set_trace()
        if self.return_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id
        if self.image_set in ['train', 'train_ov', 'val_ov', 'val', 'train_cur', 
                              'train_coco_lasot_got', 'val_coco_lasot_got', 'train_o365', 'val_o365']:
            target["template_id"] = template
        if keypoints is not None:
            target["keypoints"] = keypoints

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        return image, target


def make_coco_transforms(image_set, fix_size=False, strong_aug=False, args=None):

    # ------------------------------------------------------------------------------------------------------------------
    # print('transform')
    # ------------------------------------------------------------------------------------------------------------------

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # config the params for data aug
    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
    max_size = 1333
    scales2_resize = [400, 500, 600]
    scales2_crop = [384, 600]
    
    # update args from config files
    scales = getattr(args, 'data_aug_scales', scales)
    max_size = getattr(args, 'data_aug_max_size', max_size)
    scales2_resize = getattr(args, 'data_aug_scales2_resize', scales2_resize)
    scales2_crop = getattr(args, 'data_aug_scales2_crop', scales2_crop)

    # resize them
    data_aug_scale_overlap = getattr(args, 'data_aug_scale_overlap', None)
    if data_aug_scale_overlap is not None and data_aug_scale_overlap > 0:
        data_aug_scale_overlap = float(data_aug_scale_overlap)
        scales = [int(i*data_aug_scale_overlap) for i in scales]
        max_size = int(max_size*data_aug_scale_overlap)
        scales2_resize = [int(i*data_aug_scale_overlap) for i in scales2_resize]
        scales2_crop = [int(i*data_aug_scale_overlap) for i in scales2_crop]
    # else:
    #     scales = getattr(args, 'data_aug_scales', scales)
    #     max_size = getattr(args, 'data_aug_max_size', max_size)
    #     scales2_resize = getattr(args, 'data_aug_scales2_resize', scales2_resize)
    #     scales2_crop = getattr(args, 'data_aug_scales2_crop', scales2_crop)

    datadict_for_print = {
        'scales': scales,
        'max_size': max_size,
        'scales2_resize': scales2_resize,
        'scales2_crop': scales2_crop
    }
    print("data_aug_params:", json.dumps(datadict_for_print, indent=2))
        

    if image_set == 'train':
        if fix_size:
            return T.Compose([
                T.RandomHorizontalFlip(),
                T.RandomResize([(max_size, max(scales))]),
                # T.RandomResize([(512, 512)]),
                normalize,
            ])
        
        # if os.environ.get('IPDB_DEBUG_SHILONG') == 'INFO':
        #     import datasets.sltransform as SLT
        #     return T.Compose([
        #         T.RandomHorizontalFlip(),
        #         T.RandomSelect(
        #             T.RandomResize(scales, max_size=1333),
        #             T.Compose([
        #                 T.RandomResize([400, 500, 600]),
        #                 T.RandomSizeCrop(384, 600),
        #                 T.RandomResize(scales, max_size=1333),
        #             ])
        #         ),
        #         SLT.RandomCropDebug(),
        #         SLT.LightingNoise(),
        #         SLT.AdjustBrightness(2),
        #         SLT.AdjustContrast(2),
        #         SLT.Albumentations(),
        #         normalize,
        #     ])

        # if strong_aug:
        #     import datasets.sltransform as SLT
        #     return T.Compose([
        #         T.RandomHorizontalFlip(),
        #         T.RandomSelect(
        #             T.RandomResize(scales, max_size=max_size),
        #             T.Compose([
        #                 T.RandomResize(scales2_resize),
        #                 T.RandomSizeCrop(*scales2_crop),
        #                 T.RandomResize(scales, max_size=max_size),
        #             ])
        #         ),
        #         T.RandomSelect(
        #             SLT.RandomSelectMulti([
        #                 SLT.RandomCrop(),
        #                 SLT.LightingNoise(),
        #                 SLT.AdjustBrightness(2),
        #                 SLT.AdjustContrast(2),
        #             ]),                   
        #             SLT.Albumentations(),
        #             p=0.05
        #         ),
        #         normalize,
        #     ])

        if strong_aug:
            import datasets.sltransform as SLT
            
            return T.Compose([
                T.RandomHorizontalFlip(),
                T.RandomSelect(
                    T.RandomResize(scales, max_size=max_size),
                    T.Compose([
                        T.RandomResize(scales2_resize),
                        T.RandomSizeCrop(*scales2_crop),
                        T.RandomResize(scales, max_size=max_size),
                    ])
                ),
                SLT.RandomSelectMulti([
                    SLT.RandomCrop(),
                    # SLT.Rotate(10),
                    SLT.LightingNoise(),
                    SLT.AdjustBrightness(2),
                    SLT.AdjustContrast(2),
                ]),              
                # # for debug only  
                # SLT.RandomCrop(),
                # SLT.LightingNoise(),
                # SLT.AdjustBrightness(2),
                # SLT.AdjustContrast(2),
                # SLT.Rotate(10),
                normalize,
            ])
        
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomSelect(
                T.RandomResize(scales, max_size=max_size),
                T.Compose([
                    T.RandomResize(scales2_resize),
                    T.RandomSizeCrop(*scales2_crop),
                    T.RandomResize(scales, max_size=max_size),
                ])
            ),
            normalize,
        ])

    if image_set in ['val', 'eval_debug', 'train_adj', 'test', 'train_ov', 'val_ov', 'test_ov', 'train_cur', 
                     'panda', 'train_coco_lasot_got', 'val_coco_lasot_got', 'train_o365', 'val_o365']:

        if os.environ.get("GFLOPS_DEBUG_SHILONG", False) == 'INFO':
            print("Under debug mode for flops calculation only!!!!!!!!!!!!!!!!")
            return T.Compose([
                T.ResizeDebug((1280, 800)),
                normalize,
            ])   

        return T.Compose([
            T.RandomResize([max(scales)], max_size=max_size),
            normalize,
        ])



    raise ValueError(f'unknown {image_set}')


def get_aux_target_hacks_list(image_set, args):
    if args.modelname in ['q2bs_mask', 'q2bs']:
        aux_target_hacks_list = [
            label2compat(), 
            label_compat2onehot(), 
            RandomSelectBoxes(num_class=args.num_classes)
        ]
        if args.masked_data and image_set == 'train':
            # aux_target_hacks_list.append()
            aux_target_hacks_list.append(MaskCrop())
    elif args.modelname in ['q2bm_v2', 'q2bs_ce', 'q2op', 'q2ofocal', 'q2opclip', 'q2ocqonly']:
        aux_target_hacks_list = [
            label2compat(),
            label_compat2onehot(),
            box_label_catter(),
            RandomSelectBoxlabels(num_classes=args.num_classes,
                                    prob_first_item=args.prob_first_item,
                                    prob_random_item=args.prob_random_item,
                                    prob_last_item=args.prob_last_item,
                                    prob_stop_sign=args.prob_stop_sign,
                                    ),
            BboxPertuber(max_ratio=0.02, generate_samples=1000),
        ]
    elif args.modelname in ['q2omask', 'q2osa']:
        if args.coco_aug:
            aux_target_hacks_list = [
                label2compat(),
                label_compat2onehot(),
                box_label_catter(),
                RandomSelectBoxlabels(num_classes=args.num_classes,
                                        prob_first_item=args.prob_first_item,
                                        prob_random_item=args.prob_random_item,
                                        prob_last_item=args.prob_last_item,
                                        prob_stop_sign=args.prob_stop_sign,
                                        ),
                RandomDrop(p=0.2),
                BboxPertuber(max_ratio=0.02, generate_samples=1000),
                RandomCutout(factor=0.5)
            ]
        else:
            aux_target_hacks_list = [
                label2compat(),
                label_compat2onehot(),
                box_label_catter(),
                RandomSelectBoxlabels(num_classes=args.num_classes,
                                        prob_first_item=args.prob_first_item,
                                        prob_random_item=args.prob_random_item,
                                        prob_last_item=args.prob_last_item,
                                        prob_stop_sign=args.prob_stop_sign,
                                        ),
                BboxPertuber(max_ratio=0.02, generate_samples=1000),
            ]
    else:
        aux_target_hacks_list = None

    return aux_target_hacks_list


def build(image_set, args):
    root = Path(args.coco_path)
    # assert root.exists(), f'provided COCO path {root} does not exist'
    mode = 'instances'
    gmot_root = '../gmot-main/data/COCO/'
    gmot_root = Path(gmot_root)
    panda_root = '../dataset/panda/'
    panda_root = Path(panda_root)
    ov_root = 'ov_data/'
    ov_root = Path(ov_root)
    if args.train_with_coco_lasot_got:
        lasot_got_coco_root = args.coco_lasot_got_path
        lasot_got_coco_root = Path(lasot_got_coco_root)
        val_coco_lasot_got = args.coco_lasot_got_path + '/COCO'
        val_coco_lasot_got = Path(val_coco_lasot_got)
    else:
        lasot_got_coco_root = Path(root)
        val_coco_lasot_got = Path(root)
    PATHS = {
        "train": (root / 'train2017', root / "annotations" / f'{mode}_train2017.json'),
        "train_ov": (root / 'train2017', ov_root / "annotations" / f'{mode}_ov_train2017.json'),
        "train_cur": (root / 'train2017', root / "annotations" / f'{mode}_train2017.json'),
        "train_adj": (root, root / "annotations" / f'fsc_adj.json'),
        "test":(gmot_root, gmot_root / 'annotations' / 'gmot_test.json'),
        "test_track":(gmot_root, gmot_root / 'annotations' / 'gmot_test.json'),
        # "test_ov":(root / 'train2017', root / "annotations" / f'{mode}_train2017.json'),
        "test_ov": (root / 'val2017', root / "annotations" / f'{mode}_val2017.json'),
        "train_reg": (root / "train2017", root / "annotations" / f'{mode}_train2017.json'),
        "val": (root / 'val2017', root / "annotations" / f'{mode}_val2017.json'),
        "val_ov": (root / 'val2017', ov_root / "annotations" / f'{mode}_ov_val2017.json'),
        "eval_debug": (root / "val2017", root / "annotations" / f'{mode}_val2017.json'),
        "panda": (panda_root, panda_root / "annotations" / f'01_University_Canteen.json'),
        "train_coco_lasot_got": (lasot_got_coco_root, lasot_got_coco_root / f'{mode}_coco_lasot_got_train.json'),
        "val_coco_lasot_got": (val_coco_lasot_got / 'val2017', val_coco_lasot_got / "annotations" / f'{mode}_val2017.json'),
        "train_o365": (root / 'train', root / f'cov_o365_train.json'),
        "val_o365": (root / 'val', root / f'cov_o365_val.json'),
        # "test": (root / "test2017", root / "annotations" / 'image_info_test-dev2017.json' ),
    }

    # add some hooks to datasets
    aux_target_hacks_list = get_aux_target_hacks_list(image_set, args)
    img_folder, ann_file = PATHS[image_set]
    print('Dataset name:')
    print(img_folder, ann_file)
    # exit(0)
    # import pdb; pdb.set_trace()
    
    # copy to local path
    if os.environ.get('DATA_COPY_SHILONG') == 'INFO':
        preparing_dataset(dict(img_folder=img_folder, ann_file=ann_file), image_set, args)

    try:
        strong_aug = args.strong_aug
    except:
        strong_aug = False
    if image_set == 'test_track':
        dataset = GMOT40Wrapper()
        # import pdb; pdb.set_trace()
    elif image_set == 'test_ov':
        dataset = CocoDetection(img_folder, ann_file,
                transforms=make_coco_transforms(image_set, fix_size=args.fix_size, strong_aug=strong_aug, args=args), 
                return_masks=args.masks,
                image_set=image_set,
                aux_target_hacks=aux_target_hacks_list,
                number_template=args.number_template,
                # num_imgs=100,
            )
    else:
        dataset = CocoDetection(img_folder, ann_file,
                transforms=make_coco_transforms(image_set, fix_size=args.fix_size, strong_aug=strong_aug, args=args), 
                return_masks=args.masks,
                image_set=image_set,
                aux_target_hacks=aux_target_hacks_list,
                number_template=args.number_template,
                # num_imgs=100,
            )
    if image_set == 'test':
        dataset.get_sequence_id_to_img_id()
        # print(dataset.class_dict)
    elif image_set == 'test_track':
        pass
    elif image_set == 'train':
        dataset.get_class_id_to_img_id()
        # dataset.get_class_weight()
        dataset.get_ov_template()
        # import pdb; pdb.set_trace()
    elif image_set == 'val':
        dataset.get_class_id_to_img_id()
        # dataset.get_class_weight()
    else:
        dataset.get_class_id_to_img_id()
        # dataset.get_class_weight()
        print(dataset.class_dict.keys())
    return dataset



if __name__ == "__main__":
    # # aux_target_hacks_list = []
    # dataset = CocoDetection('/comp_robot/cv_public_dataset/COCO2017/train2017', 
    #         "/comp_robot/cv_public_dataset/COCO2017/annotations/instances_train2017.json", 
    #         transforms=make_coco_transforms('train'), 
    #         return_masks=False,
    #     )

    # Objects365 Val example
    dataset_o365 = CocoDetection(
            '/comp_robot/cv_public_dataset/Objects365/train/', 
            "/comp_robot/cv_public_dataset/Objects365/slannos/anno_preprocess_shilong_train_v2.json", 
            transforms=None,
            return_masks=False,
        )
    print('len(dataset_o365):', len(dataset_o365))

    # import pdb; pdb.set_trace()

    # ['/raid/liushilong/data/Objects365/train/patch16/objects365_v2_00908726.jpg', '/raid/liushilong/data/Objects365/train/patch6/objects365_v1_00320532.jpg', '/raid/liushilong/data/Objects365/train/patch6/objects365_v1_00320534.jpg']
