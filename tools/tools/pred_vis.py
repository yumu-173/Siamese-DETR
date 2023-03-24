import json
import time
import cv2

with open('D:/giga/gmot/logs/DINO/results/test_result.json') as infile:
    info = json.load(infile)
with open('G:/tracking/GMOT40/COCO/annotations/test.json') as infile:
    imgname = json.load(infile)

id_list = [99, 199, 299, 399, 499, 599, 699, 799, 899, 1099, 1199, 1299, 1399, 1499, 1599, 1699, 1799, 1899, 1999, 2099, 2199, 2299, 2399, 2499, 2599, 2699, 2799, 2899]

for search_id in id_list:
    name = imgname['images'][search_id]['file_name']
    print('id:', imgname['images'][search_id]['id'])
    image_id = imgname['images'][search_id]['id']
    print(name)
    name = 'G:/tracking/GMOT40/COCO/' + name
    image = cv2.imread(name)

    pred_list = []
    for item in info:
        if item['image_id'] == image_id and item['score'] > 0.05:
            pred_list.append(item)
    print(len(pred_list))


    for item in pred_list:
        box = item['bbox']
        # print('bbox:', box)
        tlx = int(box[0])
        tly = int(box[1])
        brx = int(box[0] + box[2])
        bry = int(box[1] + box[3])
        # print('box:', tlx, tly, brx, bry)
        # visualization
        image = cv2.rectangle(image, (tlx, tly), (brx, bry), (255, 0, 0), thickness=2)

    img_name = 'G:/tracking/GMOT40/COCO/pred/pred_' + str(image_id) + '.jpg'
    print(img_name)
    # visualization
    cv2.imwrite(img_name, image)

