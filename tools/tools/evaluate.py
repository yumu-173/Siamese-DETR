from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
import skimage.io as io
import pylab,json

if __name__ == "__main__":
    cocoGt = COCO('G:/tracking/GMOT40/COCO/annotations/train.json')        #标注文件的路径及文件名，json文件形式
    cocoDt = cocoGt.loadRes('D:/GMOT/gmot/logs/DINO/test_result.json')  #自己的生成的结果的路径及文件名，json文件形式
    cocoEval = COCOeval(cocoGt, cocoDt, "bbox")
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()