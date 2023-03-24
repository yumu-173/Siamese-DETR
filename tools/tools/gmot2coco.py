import json
import glob

path_list = glob.glob('G:/tracking/GMOT40/val_anno/*.txt')

category = ['airplane', 'ball', 'balloon', 'bird', 'boat', 'car', 'fish', 'insect', 'person', 'stock']


images = []
annotations = []
image_index = 0
index = 0
for path in path_list:
    filename = path.split('\\')[-1].split('.')[-2]
    path.replace('\\', '/')
    with open(path) as info:
        annos = info.readlines()
    flag = 0
    for line in annos:
        object_image = {}
        object_annos = {}
        box = line.split(',')

        if flag != int(box[0]):
            image_index += 1
            flag = int(box[0])

        # image
        object_image['id'] = image_index
        name = 'val_gmot/' + filename + '/img1/' + box[0].rjust(6, '0') + '.jpg'
        object_image['file_name'] = name
        object_image['height'] = 1080
        object_image['width'] = 1920
        if object_image not in images:
            images.append(object_image)

        # anno
        object_annos['id'] = index
        index += 1
        object_annos['image_id'] = image_index

        object_annos['category_id'] = 0
        object_annos['segmentation'] = ''
        object_annos['area'] = int(box[4], base=10) * int(box[5], base=10)
        object_annos['bbox'] = [int(box[2], base=10), int(box[3], base=10), int(box[4], base=10), int(box[5], base=10)]
        object_annos['iscrowd'] = 0
        annotations.append(object_annos)
gt = {}
gt['info'] = 'gmot'
gt['images'] = images
gt['annotations'] = annotations
gt['categories'] = [
        {
            "id": 0,
            "name": "general"
        }
    ]
gt['licenses'] = 'lyc'
total_json = json.dumps(gt, indent=2)
json_name = 'G:/tracking/GMOT40/COCO/annotations/val.json'
with open(json_name, "w", encoding='utf-8') as f:
    f.write(total_json)
    print("保存文件到本地完成")