# GPUS=`nvidia-smi -L | wc -l`
# # GPUS=1

# [[ -z "$RANK" ]] && RANK=0
# [[ -z "$AZUREML_NODE_COUNT" ]] && NODE_COUNT=1 || NODE_COUNT=$AZUREML_NODE_COUNT
# [[ -z "$AZ_BATCH_MASTER_NODE" ]] && MASTER_ADDR=127.0.0.1 || MASTER_ADDR=$(echo "$AZ_BATCH_MASTER_NODE" | cut -d : -f 1)
# [[ -z "$AZ_BATCH_MASTER_NODE" ]] && MASTER_PORT=44307 || MASTER_PORT=$(echo "$AZ_BATCH_MASTER_NODE" | cut -d : -f 2)

# echo "node rank: ${RANK}"
# echo "node count: ${NODE_COUNT}"
# echo "master addr: ${MASTER_ADDR}"
# echo "master port: ${MASTER_PORT}"

coco_path='Dataset/COCO'
checkpoint='ckpts/checkpoint_2temp11.pth'
python main.py \
    --coco_path $coco_path \
    --test_track \
    --batch_size=16 \
    --output_dir logs/DINO/R50-MS4-3 \
    --config_file config/DINO/DINO_4scale.py \
    --resume $checkpoint \
    # --dn_type origin