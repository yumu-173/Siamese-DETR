import json
import time
import cv2

with open('G:/tracking/GMOT40/COCO/annotations/test.json') as infile:
    info = json.load(infile)

pred_list = []
info = info['annotations']
for item in info:
    if item['image_id'] == 0:
        pred_list.append(item)
print(len(pred_list))
image = cv2.imread('G:/tracking/GMOT40/COCO/test_gmot/airplane-3/img1/000000.jpg')
for item in pred_list:
    box = item['bbox']
    print('bbox:', box)
    tlx = int(box[0])
    tly = int(box[1])
    brx = int(box[0] + box[2])
    bry = int(box[1] + box[3])
    print('box:', tlx, tly, brx, bry)
    # visualization
    image = cv2.rectangle(image, (tlx, tly), (brx, bry), (255, 0, 0), thickness=2)

img_name = 'G:/tracking/GMOT40/COCO/pred/000000.jpg'
print(img_name)
# visualization
cv2.imwrite(img_name, image)