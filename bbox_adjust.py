import torch
import torchvision
import torch.distributed as dist
from scipy.optimize import linear_sum_assignment
import json
import util.misc as utils

alpha = 0.5
lamda = 1 - 0.7
score = 0.4

def intersect(box_a, box_b):
    """ We resize both tensors to [A,B,2] without new malloc:
    [A,2] -> [A,1,2] -> [A,B,2]
    [B,2] -> [1,B,2] -> [A,B,2]
    Then we compute the area of intersect between box_a and box_b.
    Args:
      box_a: (tensor) bounding boxes, Shape: [A,4].
      box_b: (tensor) bounding boxes, Shape: [B,4].
    Return:
      (tensor) intersection area, Shape: [A,B].
    """
    A = box_a.size(0)
    B = box_b.size(0)
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                       box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    # print('max_xy:', max_xy)
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                       box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    # print('min_xy:', min_xy)
    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, :, 0] * inter[:, :, 1]
    # inter[:, :, 0] is the width of intersection and inter[:, :, 1] is height


def jaccard(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.  Here we operate on
    ground truth boxes and default boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: (tensor) Ground truth bounding boxes, Shape: [A,4]
        box_b: (tensor) Prior boxes from priorbox layers, Shape: [B,4]
    Return:
        jaccard overlap: (tensor) Shape: [A, B]
    """
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
    area_b = ((box_b[:, 2]-box_b[:, 0]) *
              (box_b[:, 3]-box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]
    union = area_a + area_b - inter
    # print('union:', union)
    # print('inter:', inter)
    return inter / union  # [A,B]


def bbox_adjustment(test_res, coco_path):
    # pred data
    anno_path = coco_path + '/annotations/train_fsc.json'
    with open(anno_path) as infile:
        target = json.load(infile)
    gt_anno = target['annotations']
    gt_box = {}
    for item in gt_anno:
        if item['image_id'] not in gt_box.keys():
            gt_box[item['image_id']] = {'id': [], 'bboxes': [], 'scores': []}
            gt_box[item['image_id']]['id'].append(item['id'])
            gt_box[item['image_id']]['bboxes'].append(item['bbox'])
            gt_box[item['image_id']]['scores'].append(item['score'])
        else:
            gt_box[item['image_id']]['id'].append(item['id'])
            gt_box[item['image_id']]['bboxes'].append(item['bbox'])
            gt_box[item['image_id']]['scores'].append(item['score'])
    # print(gt_box.keys())

    pred_box = {}
    for item in test_res:
        if item['image_id'] not in pred_box.keys() and item['score'] > score:
            pred_box[item['image_id']] = {'scores': [], 'bboxes': []}
            pred_box[item['image_id']]['bboxes'].append(item['bbox'])
            pred_box[item['image_id']]['scores'].append(item['score'])
        elif item['score']> score:
            pred_box[item['image_id']]['bboxes'].append(item['bbox'])
            pred_box[item['image_id']]['scores'].append(item['score'])
    
    # init adj json

    # box adjustment
    print('Start adjustment !')
    change_list = {'id': [], 'bbox': [], 'score': []}
    for key in gt_box.keys():
        gt = gt_box[key]['bboxes']
        if key in pred_box.keys():
            pred = pred_box[key]['bboxes']
            pred_score = pred_box[key]['scores']
        else:
            continue
        
        pred = torch.tensor(pred)
        pred_ = torch.ones_like(pred)
        pred_[:, :2] = pred[:, :2] - pred[:, 2:]/2
        pred_[:, 2:] = pred[:, :2] + pred[:, 2:]/2
        gt = torch.tensor(gt)
        gt[:, 2:] = gt[:, :2] + gt[:, 2:]
        # print('pred', pred.shape)
        # print('gt', gt.shape)

        IOU = jaccard(gt, pred_)
        iou = 1. - IOU
        iou[iou > lamda] = lamda + 1e-5
        iou = iou.cpu().numpy()
        row_indices, col_indices = linear_sum_assignment(iou)
        pred = pred.tolist()
        for row, col in zip(row_indices, col_indices):
            if iou[row, col] < lamda:
                # gt_box[key]['bboxes'][row] = alpha * gt_box[key]['bboxes'][row] + (1-alpha) * pred[col]
                anno_id = gt_box[key]['id'][row]
                anno_score = gt_box[key]['scores'][row]
                for i, item in enumerate(gt_anno):
                    if item['id'] == anno_id and pred_score[col] > anno_score:
                        # import pdb; pdb.set_trace()
                        pred[col] = [pred[col][0]-pred[col][2]/2, pred[col][1]-pred[col][3]/2, pred[col][2], pred[col][3]]
                        # import pdb; pdb.set_trace()
                        change_list['bbox'].append([alpha*a + (1 - alpha)*b for a,b in zip(item['bbox'], pred[col])])
                        change_list['id'].append(anno_id)
                        change_list['score'].append(pred_score[col])
                        # item['area'] = item['bbox'][2] * item['bbox'][3]
                        # item['score'] = pred_score[col]
                        # gt_anno[i] = item
    change_list['bbox'] = torch.tensor(change_list['bbox']).cuda()
    change_list['id'] = torch.tensor(change_list['id']).cuda()
    change_list['score'] = torch.tensor(change_list['score']).cuda()
    
    world_size = utils.get_world_size()
    
    torch.distributed.barrier()
    change_list_gather = utils.all_gather(change_list) # List[Dict]
    print('Rank:{}'.format(utils.get_rank()), change_list.keys())
    if utils.is_main_process():
        for i in range(world_size):
            change_list_gather[i]['id'] = change_list_gather[i]['id'].tolist()
            
        # # 创建一个空列表用于存放所有进程的字典数据
        # gather_list = [{} for _ in range(world_size)]

        # import copy
        # # 对字典中的每个键值对进行all_gather操作
        # for key in change_list:
            
        #     # 创建一个空列表用于存放当前键对应的值
        #     value_list = [torch.zeros_like(change_list[key]) for _ in range(world_size)]
        #     # import pdb; pdb.set_trace()
        #     # 调用all_gather函数将当前键对应的值收集到value_list中，并广播到所有进程
        #     dist.all_gather(value_list, change_list[key])
        #     # 将value_list中的值按照键添加到gather_list中的相应位置
        #     for i in range(world_size):
        #         gather_list[i][key] = value_list[i]
        # print('Gather over !')
 
        # 如果是主卡，则保存gather_list中的结果
        print('This is rank 0, start write adj.json')

        anno_total = {}
        anno_total['info'] = target['info']
        anno_total['categories'] = target['categories']
        anno_total['licenses'] = target['licenses']
        anno_total['images'] = target['images']
        anno_list = []
        

        for item in target['annotations']:
            flag = 0
            item_id = item['id']
            for i in range(world_size):
                idx_count = change_list_gather[i]['id'].count(item_id)
                if idx_count > 0:
                    idx = change_list_gather[i]['id'].index(item_id)
                    print('find:', idx)
                    item['bbox'] = change_list_gather[i]['bbox'][idx].cpu().tolist()
                    item['area'] = item['bbox'][2] * item['bbox'][3]
                    item['score'] = change_list_gather[i]['score'][idx].cpu().item()
                    anno_list.append(item)
                    flag = 1
                # for j, change_id in enumerate(gather_list[i]['id']):
                #     if change_id == item_id:
                #         print(change_id)
                #         item['bbox'] = gather_list[i]['bbox'][j].cpu().tolist()
                #         item['area'] = item['bbox'][2] * item['bbox'][3]
                #         item['score'] = gather_list[i]['score'][j].cpu().item()
                #         anno_list.append(item)
            if flag == 0:
                anno_list.append(item)
        print('anno_list:', len(anno_list))
        anno_total['annotations'] = anno_list

        total_json = json.dumps(anno_total, indent=2)
        json_name = coco_path + '/annotations/fsc_adj.json'
        with open(json_name, "w", encoding='utf-8') as f:
            f.write(total_json)
            print("write over !")