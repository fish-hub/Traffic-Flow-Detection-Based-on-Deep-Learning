#!/usr/bin/env python3
import argparse
import os
import time
from loguru import logger
import cv2
import torch
import sys
import tfd_library
sys.path.append(r'D:\code_library\python_code\pytorch_WZ\YOLOX-main')
from yolox.data.data_augment import ValTransform
from yolox.data.datasets import COCO_CLASSES
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess, vis
import numpy as np
from deep_sort import DeepSort
def xyxy2xywh(x):
    '''
    (center x, center y,width, height)
    '''
    y = torch.zeros_like(x) if isinstance(x,
                                          torch.Tensor) else np.zeros_like(x)
    y[:, 0] = (x[:, 0] + x[:, 2])/2.
    y[:, 1] = (x[:, 1] + x[:, 3])/2.
    y[:, 2] = x[:, 2] - x[:, 0]
    y[:, 3] = x[:, 3] - x[:, 1]
    return y

def draw_(img, boxes, scores, cls_ids, conf=0.5, class_names=None,iou=None):

    for i in range(len(boxes)):
        box = boxes[i]
        cls_id = int(cls_ids[i])
        score = scores[i]
        if score < conf:
            continue
        x0 = int(box[0])
        y0 = int(box[1])
        x1 = int(box[2])
        y1 = int(box[3])

        color = (255,0,255)
        text = '{}:{:.1f}%'.format(class_names[cls_id], score * 100)
        txt_color = (255,255,255)
        font = cv2.FONT_HERSHEY_SIMPLEX

        txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]

        if (iou[2] > x0 > iou[0] and iou[2] > x0 > iou[0] and iou[3] > y0 > iou[1] and iou[1] < y1 < iou[3]):
            cv2.rectangle(img, (x0, y0), (x1, y1), [255,24,123], 2)
            txt_bk_color = (100,234,120)
            cv2.rectangle(
                img,
                (x0, y0 + 1),
                (x0 + txt_size[0] + 1, y0 + int(1.5 * txt_size[1])),
                txt_bk_color,
                -1
            )
            cv2.putText(img, text, (x0, y0 + txt_size[1]), font, 0.4, [255,24,123], thickness=1)
        else:
            cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)

            cv2.rectangle(
                img,
                (x0, y0 + 1),
                (x0 + txt_size[0] + 1, y0 + int(1.5 * txt_size[1])),
                color,
                -1
            )
            cv2.putText(img, text, (x0, y0 + txt_size[1]), font, 0.4, txt_color, thickness=1)
    return img
def plot_one_box(x, ori_img, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img
    img = ori_img
    tl = line_thickness or round(
        0.002 * max(img.shape[0:2])) + 1  # line thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1)  # filled
        cv2.putText(img,
                    label, (c1[0], c1[1] - 2),
                    0,
                    tl / 3, [225, 255, 255],
                    thickness=tf,
                    lineType=cv2.LINE_AA)
    return img
def draw_bboxes(ori_img, bbox, identities=None, offset=(0,0)):
    img = ori_img
    for i,box in enumerate(bbox):
        x1,y1,x2,y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        # box text and bar
        id = int(identities[i]) if identities is not None else 0
        color = [255,255,0]
        label = '{}{:d}'.format("", id)
        # t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2 , 2)[0]
        img = plot_one_box([x1,y1,x2,y2], img, color, label)
        # cv2.rectangle(img,(x1, y1),(x2,y2),color,3)
        # cv2.rectangle(img,(x1, y1),(x1+t_size[0]+3,y1+t_size[1]+4), color,-1)
        # cv2.putText(img,label,(x1,y1+t_size[1]+4), cv2.FONT_HERSHEY_PLAIN, 2, [255,255,255], 2)
    return img
class Predictor(object):
    def __init__(self,model,exp,cls_names=COCO_CLASSES,device="gpu",fp16=False,legacy=False,iou=None,deepSortCheckPoint="./weights/ckpt.t7"):
        self.model = model
        self.clsNames = cls_names
        self.confThre = exp.test_conf
        self.nmsThre = exp.nmsthre
        self.numClasses = exp.num_classes
        self.testSize = exp.test_size
        self.device = device
        self.fp16 = fp16
        self.preproc = ValTransform(legacy=legacy)
        self.iou = iou
        self.deepSortforVehicle = DeepSort(deepSortCheckPoint,0.2)

    def inference(self,img):
        imgInfo = {"id":0}
        if isinstance(img,str):
            imgInfo["file_name"] = os.path.basename(img)
            img = cv2.imread(img)
        else:
            imgInfo["file_name"] = None
        height,width = img.shape[:2]
        ratio = min(self.testSize[0]/img.shape[0],self.testSize[1]/img.shape[1])
        imgInfo["height"] = height
        imgInfo["width"] = width
        imgInfo["raw_img"] = img
        imgInfo["ratio"] = ratio

        img,_ = self.preproc(img,None,self.testSize)
        img = torch.from_numpy(img).unsqueeze(0).float()
        if self.device == "gpu":
            img = img.cuda()
            if self.fp16:
                img = img.half()
        with torch.no_grad():
            t0 = time.time()
            output = self.model(img)
            output = postprocess(output,self.numClasses,self.confThre,self.nmsThre,class_agnostic=True)#nms滤除

        logger.info("Infer time：{:.4f}s".format(time.time()-t0))
        return output,imgInfo

    def visual(self,output,img_info,cls_conf=0.35):
        ratio = img_info["ratio"]
        img = img_info["raw_img"]
        if output is None:
            return img
        output = output.cpu()
        bboxes = output[:,:4]
        # 最终网络预测得到的bbox位置+cls类别+scores分数
        bboxes/=ratio
        cls = output[:,6]
        scores = output[:,4]*output[:,5]
        #*******************
        if(bboxes is not None):
            bbox_cxcywh = xyxy2xywh(bboxes)
            outputtt = self.deepSortforVehicle.update(bbox_cxcywh,scores,img)
            if(len(outputtt)>0):
                bbox_xyxy = outputtt[:,:4]
                ids = outputtt[:,-1]
                img = draw_bboxes(img, bbox_xyxy, ids)

        visres = tfd_library.objDraw(img,bboxes,scores,cls,cls_conf,self.clsNames,self.iou)
        return visres

def main(exp, modelPath):
    exp.test_conf = 0.7
    exp.nmsthre = 0.4
    exp.test_size = (640,640)

    model = exp.get_model()
    model.cuda()
    model.eval()
    ckpt_file = modelPath
    ckpt = torch.load(ckpt_file, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    logger.info("loaded checkpoint done.")
    predictor = Predictor(
        model, exp, COCO_CLASSES, "gpu", False, False,iou=[550,300,1200,500]
    )
    return predictor

if __name__ =="__main__":
    fileName = "./src.mp4"
    modelPath = "./YOLOX_outputs/yolox_voc_s/best_ckpt.pth"
    #modelPath = "./weights/best_ckpt.pth"
    modelName = "yolox-s"
    cap = cv2.VideoCapture(fileName)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    currentFrameIndex = 0
    exp = get_exp(None,modelName)
    predictor = main(exp,modelPath)
    while True:
        retVal,frame = cap.read()
        if retVal:
            if((frames-1)==currentFrameIndex):
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

            #模型推理
            #cv2.imshow("img_show",frame)
            outputs ,img_info = predictor.inference(frame)
            result_frame = predictor.visual(outputs[0],img_info,predictor.confThre)
            cv2.rectangle(result_frame,[550,300],[1200,500],[0,0,255],2)
            cv2.imshow("result",result_frame);

            ch = cv2.waitKey(10)
            currentFrameIndex+=1
            if ch == 27:
                break
        else:
            break







