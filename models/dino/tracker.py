from collections import deque

import cv2
import json
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
# from torchreid import metrics
from torchvision.ops.boxes import clip_boxes_to_image, nms
import torchvision
from copy import deepcopy

from .utils_track import (bbox_overlaps, get_center, get_height, get_width, make_pos,
                    warp_pos)


class Tracker:
    """The main tracking file, here is where magic happens."""
    # only track pedestrian
    cl = 1

    def __init__(self, obj_detect, reid_network, tracker_cfg):
        self.obj_detect = obj_detect
        self.reid_network = reid_network
        self.detection_person_thresh = tracker_cfg.detection_person_thresh
        self.regression_person_thresh = tracker_cfg.regression_person_thresh
        self.regression_iou_thresh = tracker_cfg.regression_iou_thresh
        self.detection_nms_thresh = tracker_cfg.detection_nms_thresh
        self.regression_nms_thresh = tracker_cfg.regression_nms_thresh
        self.public_detections = tracker_cfg.public_detections
        self.inactive_patience = tracker_cfg.inactive_patience
        self.do_reid = tracker_cfg.do_reid
        self.max_features_num = tracker_cfg.max_features_num
        self.reid_sim_threshold = tracker_cfg.reid_sim_threshold
        self.reid_iou_threshold = tracker_cfg.reid_iou_threshold
        self.do_align = tracker_cfg.do_align
        self.motion_model_cfg = tracker_cfg.motion_model

        self.warp_mode = getattr(cv2, tracker_cfg.warp_mode)
        self.number_of_iterations = tracker_cfg.number_of_iterations
        self.termination_eps = tracker_cfg.termination_eps

        self.tracks = []
        self.inactive_tracks = []
        self.track_num = 0
        self.im_index = 0
        self.results = {}

        self.frame_range = tracker_cfg.frame_range
        self.num_queries = tracker_cfg.num_queries

        self.load_results =tracker_cfg.load_results
        if self.load_results:
            self.det_results = {}
            with open('logs/DINO/R50-MS4-1/results_gmotdet_name.json') as f:
                gmot_det = json.load(f)
            for item in gmot_det:
                seq_name = item['image_id'].split('/')[-3]
                frame_id = int(item['image_id'].split('/')[-1][:-4])
                box = item['bbox']
                if seq_name not in self.det_results.keys():
                    self.det_results[seq_name] = {}
                if frame_id not in self.det_results[seq_name].keys():
                    self.det_results[seq_name][frame_id] = []
                    self.det_results[seq_name][frame_id].append([box[0]+box[2]/2, box[1]+box[3]/2, box[2], box[3]])
                else:
                    self.det_results[seq_name][frame_id].append([box[0]+box[2]/2, box[1]+box[3]/2, box[2], box[3]])

    def reset(self, hard=True):
        self.tracks = []
        self.inactive_tracks = []

        if hard:
            self.track_num = 0
            self.results = {}
            self.im_index = 0

    def tracks_to_inactive(self, tracks):
        self.tracks = [t for t in self.tracks if t not in tracks]
        for t in tracks:
            t.pos = t.last_pos[-1]
        self.inactive_tracks += tracks

    # def add(self, new_det_pos, new_det_scores, new_det_features):
    def add(self, new_det_pos, new_det_scores):
        """Initializes new Track objects and saves them."""
        num_new = new_det_pos.size(0)
        for i in range(num_new):
            self.tracks.append(Track(
                new_det_pos[i].view(1, -1),
                new_det_scores[i],
                self.track_num + i,
                # new_det_features[i].view(1, -1),
                self.inactive_patience,
                self.max_features_num,
                self.motion_model_cfg['n_steps'] if self.motion_model_cfg['n_steps'] > 0 else 1
            ))
        self.track_num += num_new

    def regress_tracks(self, blob, boxes, scores, box_index):
        """Regress the position of the tracks and also checks their scores."""
        pos_num = len(self.tracks)
        # regress
        # import pdb; pdb.set_trace()
        # boxes, scores = self.obj_detect.predict_boxes(pos)
        pos = clip_boxes_to_image(boxes, blob['img'].shape[-2:])
        # import pdb; pdb.set_trace()
        keep = nms(pos, scores, 0.8)
        pos = pos[keep]
        scores = scores[keep]
        boxes_index = box_index[keep]

        boxes_order = torch.zeros((pos_num, 4)).cuda()
        scores_order = torch.zeros(pos_num).cuda()
        for box, score, idx in zip(pos, scores, boxes_index):
            iou_thresh = self.regression_iou_thresh
            order = idx % pos_num
            # print(order)
            t = self.tracks[order]
            track_pos = deepcopy(t.pos)
            track_pos[:,0] = track_pos[:,0] - track_pos[:,2]/2
            track_pos[:,1] = track_pos[:,1] - track_pos[:,3]/2
            track_pos[:,2] = track_pos[:,0] + track_pos[:,2]
            track_pos[:,3] = track_pos[:,1] + track_pos[:,3]
            # import pdb; pdb.set_trace()
            iou = torchvision.ops.box_iou(track_pos, box[None, :]).view(-1).item()
            if iou <= self.regression_iou_thresh:
                scores_order[order] = 0
            elif iou > iou_thresh:
                iou_thresh = iou
                scores_order[order] = self.regression_person_thresh + 1e-5
                boxes_order[order] = box

        pos = boxes_order
        pos[:, 2] = pos[:, 2] - pos[:, 0]
        pos[:, 3] = pos[:, 3] - pos[:, 1]
        pos[:, 0] = pos[:, 0] + pos[:, 2]/2
        pos[:, 1] = pos[:, 1] + pos[:, 3]/2
        scores = scores_order

        # path = blob['img_path'][0]
        # img = cv2.imread(path, 1)
        # for bbox in pos:
        #     cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 1)
        # # for bbox in self.get_pos():
        # #     cv2.rectangle(img, (int(bbox[0]-bbox[2]/2), int(bbox[1]-bbox[3]/2)), (int(bbox[0]+bbox[2]/2), int(bbox[1]+bbox[3]/2)), (0, 0, 255), 3)
        # img_name = 'track_vis/' + path.split('/')[-3] + '_' + path.split('/')[-1]
        # cv2.imwrite(img_name, img)
        # import pdb; pdb.set_trace()

        s = []
        for i in range(len(self.tracks) - 1, -1, -1):
            t = self.tracks[i]
            t.score = scores[i]
            if scores[i] <= self.regression_person_thresh:
                self.tracks_to_inactive([t])
            else:
                s.append(scores[i])
                # t.prev_pos = t.pos
                t.pos = pos[i].view(1, -1)

        return torch.Tensor(s[::-1]).cuda()

    def get_pos(self):
        """Get the positions of all active tracks."""
        if len(self.tracks) == 1:
            pos = self.tracks[0].pos
        elif len(self.tracks) > 1:
            pos = torch.cat([t.pos for t in self.tracks], 0)
        else:
            pos = torch.zeros(0).cuda()
        return pos

    def get_features(self):
        """Get the features of all active tracks."""
        if len(self.tracks) == 1:
            features = self.tracks[0].features
        elif len(self.tracks) > 1:
            features = torch.cat([t.features for t in self.tracks], 0)
        else:
            features = torch.zeros(0).cuda()
        return features

    def get_inactive_features(self):
        """Get the features of all inactive tracks."""
        if len(self.inactive_tracks) == 1:
            features = self.inactive_tracks[0].features
        elif len(self.inactive_tracks) > 1:
            features = torch.cat([t.features for t in self.inactive_tracks], 0)
        else:
            features = torch.zeros(0).cuda()
        return features

    def reid(self, blob, new_det_pos, new_det_scores):
        """Tries to ReID inactive tracks with new detections."""
        # new_det_features = [torch.zeros(0).cuda() for _ in range(len(new_det_pos))]

        if self.do_reid:
            # new_det_features = self.get_appearances(blob, new_det_pos)

            if len(self.inactive_tracks) >= 1:
                # calculate appearance distances
                dist_mat, pos = [], []
                for t in self.inactive_tracks:
                #     dist_mat.append(torch.cat([t.test_features(feat.view(1, -1))
                #                                for feat in new_det_features], dim=1))
                    pos.append(t.pos)
                if len(pos) > 1:
                    # dist_mat = torch.cat(dist_mat, 0)
                    pos = torch.cat(pos, 0)
                else:
                    # dist_mat = dist_mat[0]
                    pos = pos[0]

                # calculate IoU distances
                if self.reid_iou_threshold:
                    iou = bbox_overlaps(pos, new_det_pos)
                    iou_mask = torch.ge(iou, self.reid_iou_threshold)
                    iou_neg_mask = ~iou_mask
                    # make all impossible assignments to the same add big value
                    # dist_mat = dist_mat * iou_mask.float() + iou_neg_mask.float() * 1000
                    dist_mat = 1 - iou + iou_neg_mask.float() * 1000
                    # import pdb; pdb.set_trace()
                dist_mat = dist_mat.cpu().numpy()

                row_ind, col_ind = linear_sum_assignment(dist_mat)

                assigned = []
                remove_inactive = []
                for r, c in zip(row_ind, col_ind):
                    # import pdb; pdb.set_trace()
                    if dist_mat[r, c] <= self.reid_sim_threshold:
                        t = self.inactive_tracks[r]
                        self.tracks.append(t)
                        t.count_inactive = 0
                        t.pos = new_det_pos[c].view(1, -1)
                        t.reset_last_pos()
                        # t.add_features(new_det_features[c].view(1, -1))
                        assigned.append(c)
                        remove_inactive.append(t)

                for t in remove_inactive:
                    self.inactive_tracks.remove(t)

                keep = torch.Tensor([i for i in range(new_det_pos.size(0)) if i not in assigned]).long().cuda()
                if keep.nelement() > 0:
                    new_det_pos = new_det_pos[keep]
                    new_det_scores = new_det_scores[keep]
                    # new_det_features = new_det_features[keep]
                else:
                    new_det_pos = torch.zeros(0).cuda()
                    new_det_scores = torch.zeros(0).cuda()
                    # new_det_features = torch.zeros(0).cuda()
        
        return new_det_pos, new_det_scores
        # return new_det_pos, new_det_scores, new_det_features

    def get_appearances(self, blob, pos):
        """Uses the siamese CNN to get the features for all active tracks."""
        crops = []
        for r in pos:
            # import pdb; pdb.set_trace()
            x0 = max(int(r[0]-r[2]/2), 0)
            y0 = max(int(r[1]-r[3]/2), 0)
            x1 = min(int(r[2]/2+r[0]), blob['img'].shape[3]-1)
            y1 = min(int(r[3]/2+r[1]), blob['img'].shape[2]-1)
            if x0 == x1:
                if x0 != 0:
                    x0 -= 1
                else:
                    x1 += 1
            if y0 == y1:
                if y0 != 0:
                    y0 -= 1
                else:
                    y1 += 1
            crop = blob['img'][0, :, y0:y1, x0:x1].permute(1, 2, 0)
            # import pdb; pdb.set_trace()
            crops.append(crop.mul(255).numpy().astype(np.uint8))

        new_features = self.reid_network(crops)

        return new_features

    def add_features(self, new_features):
        """Adds new appearance features to active tracks."""
        for t, f in zip(self.tracks, new_features):
            t.add_features(f.view(1, -1))

    def align(self, blob):
        """Aligns the positions of active and inactive tracks depending on camera motion."""
        if self.im_index > 0:
            im1 = np.transpose(self.last_image.cpu().numpy(), (1, 2, 0))
            im2 = np.transpose(blob['img'][0].cpu().numpy(), (1, 2, 0))
            im1_gray = cv2.cvtColor(im1, cv2.COLOR_RGB2GRAY)
            im2_gray = cv2.cvtColor(im2, cv2.COLOR_RGB2GRAY)
            warp_matrix = np.eye(2, 3, dtype=np.float32)
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, self.number_of_iterations,  self.termination_eps)
            cc, warp_matrix = cv2.findTransformECC(im1_gray, im2_gray, warp_matrix, self.warp_mode, criteria)
            warp_matrix = torch.from_numpy(warp_matrix)

            for t in self.tracks:
                t.pos = warp_pos(t.pos, warp_matrix)
                # t.pos = clip_boxes(Variable(pos), blob['im_info'][0][:2]).data

            if self.do_reid:
                for t in self.inactive_tracks:
                    t.pos = warp_pos(t.pos, warp_matrix)

            if self.motion_model_cfg['enabled']:
                for t in self.tracks:
                    for i in range(len(t.last_pos)):
                        t.last_pos[i] = warp_pos(t.last_pos[i], warp_matrix)

    def motion_step(self, track):
        """Updates the given track's position by one step based on track.last_v"""
        if self.motion_model_cfg['center_only']:
            center_new = get_center(track.pos) + track.last_v
            track.pos = make_pos(*center_new, get_width(track.pos), get_height(track.pos))
        else:
            track.pos = track.pos + track.last_v

    def motion(self):
        """Applies a simple linear motion model that considers the last n_steps steps."""
        for t in self.tracks:
            last_pos = list(t.last_pos)

            # avg velocity between each pair of consecutive positions in t.last_pos
            if self.motion_model_cfg['center_only']:
                vs = [get_center(p2) - get_center(p1) for p1, p2 in zip(last_pos, last_pos[1:])]
            else:
                vs = [p2 - p1 for p1, p2 in zip(last_pos, last_pos[1:])]

            t.last_v = torch.stack(vs).mean(dim=0)
            self.motion_step(t)

        if self.do_reid:
            for t in self.inactive_tracks:
                if t.last_v.nelement() > 0:
                    self.motion_step(t)

    def step(self, blob, template, postprocessors, size):
        w = size[0]
        h = size[1]
        """This function should be called every timestep to perform tracking with a blob
        containing the image information.
        """
        for t in self.tracks:
            # add current position to last_pos list
            t.last_pos.append(t.pos.clone())

        ###########################
        # Look for new detections #
        ###########################
        sample = blob['img'].to('cuda')
        # import pdb; pdb.set_trace()
        # self.obj_detect.load_image(blob['img'])

        if self.public_detections:
            raise RuntimeError('The following code has some errors need to be fixed!')
        else:
            if len(self.tracks):
                poses = self.get_pos()
                pos = deepcopy(poses)
                pos[:, 0] /= w
                pos[:, 1] /= h
                pos[:, 2] /= w
                pos[:, 3] /= h
                outputs, _ = self.obj_detect(sample, [[template]], track_pos=pos)
            else:
                outputs, _ = self.obj_detect(sample, [[template]])
            orig_target_sizes = torch.tensor([[h, w]]).cuda()
            if len(self.tracks):
                det_outputs = {}
                track_outouts = {}
                for key in outputs.keys():
                    if key in ['pred_logits', 'pred_boxes']:
                        det_outputs[key] = outputs[key][:, :self.num_queries, :]
                        track_outouts[key] = outputs[key][:, self.num_queries:, :]
                det_results = postprocessors['bbox'](det_outputs, orig_target_sizes, not_to_xyxy=True)
                track_results, box_index = postprocessors['bbox'](track_outouts, orig_target_sizes, not_to_xyxy=False)
                
                track_scores = track_results[0]['scores']
                track_boxes = track_results[0]['boxes']
                track_labels = track_results[0]['labels'].bool()
                track_scores = track_scores[track_labels]
                track_boxes = track_boxes[track_labels]
                box_index = box_index.squeeze(0)[track_labels]
                # import pdb; pdb.set_trace()
            else:
                det_results = postprocessors['bbox'](outputs, orig_target_sizes, not_to_xyxy=True)
            # import pdb; pdb.set_trace()
            scores = det_results[0]['scores']
            boxes = det_results[0]['boxes']
            labels = det_results[0]['labels'].bool()
            scores = scores[labels]
            boxes = boxes[labels]
            

            if self.load_results:
                image_name = blob['img_path'][0]
                # import pdb; pdb.set_trace()
                seq_name = image_name.split('/')[-3]
                frame_id = int(image_name.split('/')[-1][:-4])
                if frame_id in self.det_results[seq_name].keys():
                    boxes = torch.tensor(self.det_results[seq_name][frame_id]).cuda()
                    scores = torch.ones(boxes.shape[0]).cuda()
                else:
                    scores = torch.zeros(boxes.shape[0]).cuda()
                # import pdb; pdb.set_trace()

        if boxes.nelement() > 0:
            boxes = clip_boxes_to_image(boxes, blob['img'].shape[-2:])

            # Filter out tracks that have too low person score
            inds = torch.gt(scores, self.detection_person_thresh).nonzero(as_tuple=False).view(-1)
        else:
            inds = torch.zeros(0).cuda()

        if inds.nelement() > 0:
            det_pos = boxes[inds]

            det_scores = scores[inds]
        else:
            det_pos = torch.zeros(0).cuda()
            det_scores = torch.zeros(0).cuda()
        path = blob['img_path'][0]
        img = cv2.imread(path, 1)
        ##################
        # Predict tracks #
        ##################
        if len(self.tracks):
            # align
            if self.do_align:
                self.align(blob)

            # apply motion model
            if self.motion_model_cfg['enabled']:
                self.motion()
                self.tracks = [t for t in self.tracks if t.has_positive_area()]

            # regress
            # import pdb; pdb.set_trace()
            person_scores = self.regress_tracks(blob, track_boxes, track_scores, box_index)

            if len(self.tracks):
                # create nms input
                # nms here if tracks overlap
                poses = self.get_pos()
                pos = deepcopy(poses)
                pos[:, 0] = pos[:, 0] - pos[:, 2]/2
                pos[:, 1] = pos[:, 1] - pos[:, 3]/2
                pos[:, 2] = pos[:, 2] + pos[:, 0]
                pos[:, 3] = pos[:, 3] + pos[:, 1]
                keep = nms(pos, person_scores, self.regression_nms_thresh)

                self.tracks_to_inactive([self.tracks[i] for i in list(range(len(self.tracks))) if i not in keep])

                # if keep.nelement() > 0 and self.do_reid:
                #         new_features = self.get_appearances(blob, self.get_pos())
                #         self.add_features(new_features)

        #####################
        # Create new tracks #
        #####################

        # !!! Here NMS is used to filter out detections that are already covered by tracks. This is
        # !!! done by iterating through the active tracks one by one, assigning them a bigger score
        # !!! than 1 (maximum score for detections) and then filtering the detections with NMS.
        # !!! In the paper this is done by calculating the overlap with existing tracks, but the
        # !!! result stays the same.
        if det_pos.nelement() > 0:
            det_poses = deepcopy(det_pos)
            det_poses[:, 0] = det_pos[:, 0] - det_pos[:, 2]/2
            det_poses[:, 1] = det_pos[:, 1] - det_pos[:, 3]/2
            det_poses[:, 2] = det_pos[:, 0] + det_pos[:, 2]/2
            det_poses[:, 3] = det_pos[:, 1] + det_pos[:, 3]/2
            keep = nms(det_poses, det_scores, self.detection_nms_thresh)
            det_pos = det_pos[keep]
            det_scores = det_scores[keep]

            # check with every track in a single run (problem if tracks delete each other)
            
            for t in self.tracks:
                
                nms_track_poses = torch.cat([t.pos, det_pos])
                nms_track_scores = torch.cat([torch.tensor([1.0]).to(det_scores.device), det_scores])
                # nms_track_scores = torch.cat([torch.tensor([t.score]).to(det_scores.device), det_scores])
                nms_track_pos = deepcopy(nms_track_poses)
                nms_track_pos[:, 0] = nms_track_pos[:, 0] - nms_track_pos[:, 2]/2
                nms_track_pos[:, 1] = nms_track_pos[:, 1] - nms_track_pos[:, 3]/2
                nms_track_pos[:, 2] = nms_track_pos[:, 2] + nms_track_pos[:, 0]
                nms_track_pos[:, 3] = nms_track_pos[:, 3] + nms_track_pos[:, 1]
                keep = nms(nms_track_pos, nms_track_scores, self.detection_nms_thresh)
                keep = keep[torch.ge(keep, 1)] - 1

                det_pos = det_pos[keep]
                det_scores = det_scores[keep]
                if keep.nelement() == 0:
                    break

        if det_pos.nelement() > 0:
            new_det_pos = det_pos
            new_det_scores = det_scores

            # try to reidentify tracks
            # import pdb; pdb.set_trace()
            new_det_pos, new_det_scores = self.reid(blob, new_det_pos, new_det_scores)
            # import pdb; pdb.set_trace()
            # new_det_pos, new_det_scores, new_det_features = self.reid(blob, new_det_pos, new_det_scores)

            # add new
            if new_det_pos.nelement() > 0:
                # self.add(new_det_pos, new_det_scores, new_det_features)
                self.add(new_det_pos, new_det_scores)

        ####################
        # Generate Results #
        ####################
        # import pdb; pdb.set_trace()
        for t in self.tracks:
            if t.id not in self.results.keys():
                self.results[t.id] = {}
            self.results[t.id][self.im_index] = np.concatenate([
                t.pos[0].cpu().numpy(),
                np.array([t.score.cpu()])])

        for t in self.inactive_tracks:
            t.count_inactive += 1

        self.inactive_tracks = [
            t for t in self.inactive_tracks if t.has_positive_area() and t.count_inactive <= self.inactive_patience
        ]

        self.im_index += 1
        self.last_image = blob['img'][0]

        path = blob['img_path'][0]
        img = cv2.imread(path, 1)
        # for bbox in self.get_pos():
        #     cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 1)
        for i, bbox in enumerate(self.get_pos()):
            cv2.putText(img, str(self.tracks[i].id), (int(bbox[0]-bbox[2]/2), int(bbox[1]-bbox[3]/2)), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)
            cv2.rectangle(img, (int(bbox[0]-bbox[2]/2), int(bbox[1]-bbox[3]/2)), (int(bbox[0]+bbox[2]/2), int(bbox[1]+bbox[3]/2)), (0, 0, 255), 1)
        
        import os
        seq = 'track_vis/'+path.split('/')[-3]
        if os.path.exists(seq):
            pass
        else:
            os.mkdir(seq)
        img_name = seq + '/' + path.split('/')[-1]
        cv2.imwrite(img_name, img)
        # import pdb; pdb.set_trace()

    def get_results(self):
        return self.results


class Track(object):
    """This class contains all necessary for every individual track."""

    # def __init__(self, pos, score, track_id, features, inactive_patience, max_features_num, mm_steps):
    def __init__(self, pos, score, track_id, inactive_patience, max_features_num, mm_steps):
        self.id = track_id
        self.pos = pos
        self.score = score
        # self.features = deque([features])
        self.ims = deque([])
        self.count_inactive = 0
        self.inactive_patience = inactive_patience
        self.max_features_num = max_features_num
        self.last_pos = deque([pos.clone()], maxlen=mm_steps + 1)
        self.last_v = torch.Tensor([])
        self.gt_id = None

    def has_positive_area(self):
        return self.pos[0, 2] > self.pos[0, 0] and self.pos[0, 3] > self.pos[0, 1]

    def add_features(self, features):
        """Adds new appearance features to the object."""
        self.features.append(features)
        if len(self.features) > self.max_features_num:
            self.features.popleft()

    def test_features(self, test_features):
        """Compares test_features to features of this Track object"""
        if len(self.features) > 1:
            features = torch.cat(list(self.features), dim=0)
        else:
            features = self.features[0]
        features = features.mean(0, keepdim=True)
        dist = metrics.compute_distance_matrix(features, test_features)
        # dist = F.pairwise_distance(features, test_features, keepdim=True)
        return dist

    def reset_last_pos(self):
        self.last_pos.clear()
        self.last_pos.append(self.pos.clone())
