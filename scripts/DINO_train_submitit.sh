coco_path=$1
python run_with_submitit.py --coco_path $coco_path \
        --timeout 3000 --job_name DINO \
	      --ngpus 8 --nodes 1
