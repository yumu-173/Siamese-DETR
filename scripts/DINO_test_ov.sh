coco_path='../dataset/COCO/'
checkpoint='ckpts/checkpoint_ov11.pth'
python main.py \
    --coco_path $coco_path \
    --test_ov \
    --batch_size=16 \
    --resume $checkpoint \
    --output_dir logs/DINO/R50-MS4-OV