coco_path='../dataset/COCO/'
checkpoint='logs/DINO/R50-MS4-coco2/checkpoint.pth'
python main.py \
    --coco_path $coco_path \
    --eval \
    --batch_size=16 \
    --resume $checkpoint