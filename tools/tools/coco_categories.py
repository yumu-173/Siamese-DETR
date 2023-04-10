import json
from pathlib import Path

COCO_SEEN_LABELS = [1, 2, 3, 4, 7, 8, 9, 15, 16, 19, 20, 23, 24, 25, 27, 31, 33, 34, 35, 38, 42, 44, 48, 50, 51, 
                    52, 53, 54, 55, 56, 57, 59, 60, 62, 65, 70, 72, 73, 74, 75, 78, 79, 80, 82, 84, 85, 86, 90]
COCO_UNSEEN_LABELS = [5, 6, 17, 18, 21, 22, 28, 32, 36, 41, 47, 49, 61, 63, 76, 81, 87]


def fitler_coco_category(coco_dir, image_set):
    # coco_dir = Path(coco_dir)
    origin_anno_path = coco_dir + '/annotations/instances_' + image_set + '2017.json'
    with open(origin_anno_path) as info:
        coco = json.load(info)
    annotations = coco['annotations']
    new_annotations = []
    for item in annotations:
        if item['category_id'] in COCO_SEEN_LABELS:
            new_annotations.append(item)
    coco['annotations'] = new_annotations
    
    import os
    if os.path.exists('ov_data/annotations/'):
        pass
    else:
        os.makedirs('ov_data/annotations/')

    ov_anno_path = 'ov_data/annotations/instances_ov_' + image_set + '2017.json'
    total_json = json.dumps(coco, indent=2)
    with open(ov_anno_path, "w", encoding='utf-8') as f:
        f.write(total_json)
        print("Update {}_coco to ov-coco !".format(image_set))

