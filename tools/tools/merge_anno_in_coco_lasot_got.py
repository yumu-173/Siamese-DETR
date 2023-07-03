import glob
import json
import cv2
import time



def build_annotations(root):
    start = time.time()
    coco_path = root + '/COCO/annotations/instances_train2017.json'
    with open(coco_path) as f:
        info = json.load(f)
    for item in info['images']:
        item['file_name'] = 'COCO/train2017/' + item['file_name']
    
    images = info['images']
    annotations = info['annotations']

    # LaSOT 
    # lasot_category_id = 10xx
    lasot_image_id = 10000000
    LaSOT_gt_list = glob.glob(root + '/LaSOT/*/*/groundtruth.txt')
    # print(LaSOT_gt_list)
    for i, lasot_gt in enumerate(LaSOT_gt_list):
        category_id = 1000 + i
        with open(lasot_gt) as lf:
            lasot_info = lf.readlines()
        for j, line in enumerate(lasot_info):
            line = line.split(',')
            image = {}
            anno = {}
            
            # images
            image['id'] = lasot_image_id
            path = lasot_gt.replace('groundtruth.txt', 'img/'+str(j+1).rjust(8, '0')+'.jpg')
            # print(image['file_name'])
            img = cv2.imread(path)
            image['height'] = img.shape[0]
            image['width'] = img.shape[1]
            image['file_name'] = path[11:]
            images.append(image)
            # annotations
            anno['id'] = lasot_image_id
            anno['image_id'] = lasot_image_id
            anno['category_id'] = category_id
            anno['bbox'] = [int(line[0]), int(line[1]), int(line[2]), int(line[3])]
            anno['area'] = int(line[2])*int(line[3])
            annotations.append(anno)

            lasot_image_id += 1

    # GOT 
    # got_category_id = 50xx
    got_image_id = 20000000
    got_gt_list = glob.glob(root + '/GOT-10K/train_data/*/groundtruth.txt')
    # print(got_gt_list)
    for i, got_gt in enumerate(got_gt_list):
        category_id = 5000 + i
        with open(got_gt) as lf:
            got_info = lf.readlines()
        for j, line in enumerate(got_info):
            line = line.split(',')
            image = {}
            anno = {}
            
            # images
            image['id'] = got_image_id
            path = got_gt.replace('groundtruth.txt', str(j+1).rjust(8, '0')+'.jpg')
            # print(image['file_name'])
            img = cv2.imread(path)
            image['height'] = img.shape[0]
            image['width'] = img.shape[1]
            image['file_name'] = path[11:]
            images.append(image)
            # annotations
            anno['id'] = got_image_id
            anno['image_id'] = got_image_id
            anno['category_id'] = category_id
            anno['bbox'] = [float(line[0]), float(line[1]), float(line[2]), float(line[3])]
            anno['area'] = float(line[2])*float(line[3])
            annotations.append(anno)

            got_image_id += 1

    new_anno = {}
    new_anno['images'] = images
    new_anno['annotations'] = annotations
    new_anno['info'] = info['info']
    new_anno['licenses'] = info['licenses']
    new_anno['categories'] = info['categories']
    print(info.keys())

    total_json = json.dumps(new_anno, indent=2)
    with open(root + '/instances_coco_lasot_got_train.json', 'w') as f:
        f.write(total_json)
        end = time.time()
        print(end-start, 's')
        print('change lasot, got to coco over!')
    return


if __name__ == '__main__':
    root = 'Dataset'
    print('*****Merge annotations from 3 dataset*****')
    build_annotations(root)
