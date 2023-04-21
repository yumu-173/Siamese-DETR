_base_ = ['coco_transformer.py']

num_classes = 2
template_lvl = 4
number_template = 1

lr = 0.0001
param_dict_type = 'default'
lr_backbone = 1e-05
lr_backbone_names = ['backbone.0']
lr_linear_proj_names = ['reference_points', 'sampling_offsets']
lr_linear_proj_mult = 0.1
ddetr_lr_param = False
# batch_size = 8
weight_decay = 0.0001
epochs = 12
lr_drop = 11
save_checkpoint_interval = 1
clip_max_norm = 0.1
onecyclelr = False
multi_step_lr = False
lr_drop_list = [11, 20, 30]


modelname = 'dino'
frozen_weights = None
backbone = 'resnet50'
use_checkpoint = False

dilation = False
position_embedding = 'sine'
pe_temperatureH = 20
pe_temperatureW = 20
return_interm_indices = [1, 2, 3]
backbone_freeze_keywords = None
enc_layers = 6
dec_layers = 6
unic_layers = 0
pre_norm = False
dim_feedforward = 2048
hidden_dim = 256
dropout = 0.0
nheads = 8
num_queries = 600
query_dim = 4
num_patterns = 0
pdetr3_bbox_embed_diff_each_layer = False
pdetr3_refHW = -1
random_refpoints_xy = False
fix_refpoints_hw = -1
dabdetr_yolo_like_anchor_update = False
dabdetr_deformable_encoder = False
dabdetr_deformable_decoder = False
use_deformable_box_attn = False
box_attn_type = 'roi_align'
dec_layer_number = None
num_feature_levels = 4
enc_n_points = 4
dec_n_points = 4
decoder_layer_noise = False
dln_xy_noise = 0.2
dln_hw_noise = 0.2
add_channel_attention = False
add_pos_value = False
two_stage_type = 'no'
two_stage_pat_embed = 0
two_stage_add_query_num = 0
two_stage_bbox_embed_share = False
two_stage_class_embed_share = False
two_stage_learn_wh = False
two_stage_default_hw = 0.05
two_stage_keep_all_tokens = False
num_select = 600
transformer_activation = 'relu'
batch_norm_type = 'FrozenBatchNorm2d'
masks = False
aux_loss = True
set_cost_class = 2.0
set_cost_bbox = 5.0
set_cost_giou = 2.0
cls_loss_coef = 1.0
mask_loss_coef = 1.0
dice_loss_coef = 1.0
bbox_loss_coef = 5.0
giou_loss_coef = 2.0
enc_loss_coef = 1.0
interm_loss_coef = 1.0
no_interm_box_loss = False
focal_alpha = 0.25

decoder_sa_type = 'sa' # ['sa', 'ca_label', 'ca_content']
matcher_type = 'HungarianMatcher' # or SimpleMinsumMatcher
decoder_module_seq = ['sa', 'ca', 'ffn']
nms_iou_threshold = -1

dec_pred_bbox_embed_share = True
dec_pred_class_embed_share = True

# for dn
use_dn = True
dn_number = 100
dn_box_noise_scale = 0.4
dn_label_noise_ratio = 0.5
embed_init_tgt = True
dn_labelbook_size = 1

match_unstable_error = True

# for ema
use_ema = False
ema_decay = 0.9997
ema_epoch = 0

use_detached_boxes_dec_out = False

# tracktor
interpolate = False
# [False, 'debug', 'pretty']
# compile video with: `ffmpeg -f image2 -framerate 15 -i %06d.jpg -vcodec libx264 -y movie.mp4 -vf scale=320:-1`
write_images = False
# load tracking results if available and only evaluate
load_results = False
# dataset (look into tracker/datasets/factory.py)
dataset = 'mot17_train_FRCNN'
# start and end percentage of frames to run, e.g., [0.0, 0.5] for train and [0.75, 1.0] for val split.
frame_range = {
  'start': 0.0,
  'end': 1.0
}
# FRCNN score threshold for detections
detection_person_thresh = 0.21
# FRCNN score threshold for keeping the track alive
regression_person_thresh = 0.1
# NMS threshold for detection
detection_nms_thresh = 0.8
# NMS theshold while tracking
regression_nms_thresh = 0.8
# motion model settings
motion_model = {
    'enabled': False, 
    # average velocity over last n_steps steps
    'n_steps': 5,
    # if true, only model the movement of the bounding box center. If false, width and height are also modeled.
    'center_only': False
}
    
# DPM or DPM_RAW or 0, raw includes the unfiltered (no nms) versions of the provided detections,
# 0 tells the tracker to use private detections (Faster R-CNN)
public_detections = False
# Do camera motion compensation
do_align = False
# Which warp mode to use (MOTION_EUCLIDEAN, MOTION_AFFINE, ...)
warp_mode = 'MOTION_EUCLIDEAN'
# maximal number of iterations (original 50)
number_of_iterations = 100
# Threshold increment between two iterations (original 0.001)
termination_eps = 0.00001
# Use siamese network to do reid
do_reid = True
# How much timesteps dead tracks are kept and cosidered for reid
inactive_patience = 50
# How many last appearance features are to keep
max_features_num = 10
# How similar do image and old track need to be to be considered the same person
reid_sim_threshold = 200.0
# How much IoU do track and image need to be considered for matching
reid_iou_threshold = 0.4

oracle = None