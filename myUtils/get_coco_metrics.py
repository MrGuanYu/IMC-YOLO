import argparse

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--anno_json', type=str, default=r'D:\program\python\ultralytics_withV9\myRubbish\temp.json', help='training model path')
    parser.add_argument('--pred_json', type=str, default=r'D:\program\python\yolov7\runs\test\exp3\best_predictions.json', help='data yaml path')

    return parser.parse_known_args()[0]


if __name__ == '__main__':
    opt = parse_opt()
    anno_json = opt.anno_json
    pred_json = opt.pred_json

    anno = COCO(anno_json)  # init annotations api
    pred = anno.loadRes(pred_json)  # init predictions api
    eval = COCOeval(anno, pred, 'bbox')
    eval.evaluate()
    eval.accumulate()
    # print(eval.eval['precision'])
    eval.summarize()

    result = eval.stats
    print(result)

    # # 获取IoU阈值列表
    # iou_thresholds = eval.params.iouThrs
    #
    # # 找到IoU为0.5的索引
    # iou_index = np.where(iou_thresholds == 0.5)[0][0]
    #
    # # 获取指定IoU阈值的召回率
    # recall = eval.eval['recall'][iou_index, :, :, 0]  # 获取召回率
    #
    # print("IoU为0.5时的召回率:", recall)