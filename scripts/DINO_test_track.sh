coco_path='../dataset/COCO/'
checkpoint='ckpts/checkpoint_1template11.pth'
python main.py \
    --coco_path $coco_path \
    --test_track \
    --batch_size=16 \
    --output_dir logs/DINO/R50-MS4-2 \
    --resume $checkpoint