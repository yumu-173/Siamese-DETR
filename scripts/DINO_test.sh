GPUS=`nvidia-smi -L | wc -l`
# GPUS=1

[[ -z "$RANK" ]] && RANK=0
[[ -z "$AZUREML_NODE_COUNT" ]] && NODE_COUNT=1 || NODE_COUNT=$AZUREML_NODE_COUNT
[[ -z "$AZ_BATCH_MASTER_NODE" ]] && MASTER_ADDR=127.0.0.1 || MASTER_ADDR=$(echo "$AZ_BATCH_MASTER_NODE" | cut -d : -f 1)
[[ -z "$AZ_BATCH_MASTER_NODE" ]] && MASTER_PORT=44307 || MASTER_PORT=$(echo "$AZ_BATCH_MASTER_NODE" | cut -d : -f 2)

echo "node rank: ${RANK}"
echo "node count: ${NODE_COUNT}"
echo "master addr: ${MASTER_ADDR}"
echo "master port: ${MASTER_PORT}"

coco_path='../dataset/COCO/'
checkpoint='ckpts/checkpoint_swinb2.pth'
python main.py \
    --coco_path $coco_path \
    --test \
    --batch_size=16 \
    --resume $checkpoint \
    --config_file config/DINO/DINO_4scale_swin.py \
    # --dn_type no \
    --output_dir logs/DINO/R50-MS4-1