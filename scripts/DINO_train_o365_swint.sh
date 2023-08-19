GPUS=`nvidia-smi -L | wc -l`
# GPUS=1

[[ -z "$RANK" ]] && RANK=0
[[ -z "$AZUREML_NODE_COUNT" ]] && NODE_COUNT=1 || NODE_COUNT=$AZUREML_NODE_COUNT
[[ -z "$AZ_BATCH_MASTER_NODE" ]] && MASTER_ADDR=127.0.0.1 || MASTER_ADDR=$(echo "$AZ_BATCH_MASTER_NODE" | cut -d : -f 1)
[[ -z "$AZ_BATCH_MASTER_NODE" ]] && MASTER_PORT=44306 || MASTER_PORT=$(echo "$AZ_BATCH_MASTER_NODE" | cut -d : -f 2)

echo "node rank: ${RANK}"
echo "node count: ${NODE_COUNT}"
echo "master addr: ${MASTER_ADDR}"
echo "master port: ${MASTER_PORT}"

coco_path=$1
out_put_dir=$2

# python tools/tools/merge_anno_in_coco_lasot_got.py

python -m torch.distributed.run --nproc_per_node=${GPUS} \
    --nnodes ${NODE_COUNT} \
    --node_rank ${RANK} \
    --master_addr ${MASTER_ADDR} \
    --master_port ${MASTER_PORT} \
    main.py \
    --coco_path $coco_path \
    --rank ${RANK} \
    --config_file config/DINO/DINO_4scale_swint.py \
    --n_nodes ${NODE_COUNT} \
    --batch_size=1 \
    --output_dir $out_put_dir \
    --train_with_o365