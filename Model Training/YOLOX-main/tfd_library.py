#!/usr/bin/env python3
import cv2
import torch
import sys
sys.path.append(r'D:\code_library\python_code\pytorch_WZ\YOLOX-main')
import numpy as np
import random
def xyxy2xywh(x):
    y = torch.zeros_like(x) if isinstance(x,torch.Tensor) else np.zeros_like(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2.
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2.
    y[:, 2] = x[:, 2] - x[:, 0]
    y[:, 3] = x[:, 3] - x[:, 1]
    return y

def objDraw(img,boxes,scores,cls_ids,trackTime, infTime, down_count, confThres=0.5,class_names=None,iou=None,externalFlag=1,drawObjDetectionTarget=0):
    for i in range(len(boxes)):
        box = boxes[i]
        cls_id = int(cls_ids[i])
        score = scores[i]
        if score < confThres:
            continue
        x0 = int(box[0])
        y0 = int(box[1])
        x1 = int(box[2])
        y1 = int(box[3])

        externalColor = (255,0,0)
        txtBackColor = (255,255,255)
        txtColor = (255,0,255)
        text = '{}:{:.1f}%'.format(class_names[cls_id], score * 100)
        inferTimeText = "inference time:{:.4f}".format(infTime)
        trackTimeText = "track time:{:.4f}".format(trackTime)
        objNums = "Number of detected targets:{} ".format(len(boxes))
        downCount = "Traffic statistics results:{} ".format(down_count)
        font = cv2.FONT_HERSHEY_SIMPLEX
        txt_size = cv2.getTextSize(text, font, 0.3, 1)[0]
        if(drawObjDetectionTarget):
            cv2.rectangle(img, (x0, y0), (x1, y1), externalColor, 3)
            cv2.rectangle(img, (x0, y0-1), ((x0 + txt_size[0] - 1), y0 + int(1.5 * txt_size[1])), txtBackColor, -1)
            cv2.putText(img, text, (x0, y0 + txt_size[1]), font, 0.3, color=txtColor, thickness=1)

        cv2.putText(img,inferTimeText,(115,125),font, 2.3, color=[0,0,0], thickness=5)
        cv2.putText(img,trackTimeText,(115,185),font, 2.3, color=[0,0,0], thickness=5)
        cv2.putText(img,objNums,(115,245),font, 2.3, color=[0,0,0], thickness=5)
        cv2.putText(img, objNums, (115, 245), font, 2.3, color=[0, 0, 0], thickness=5)
        cv2.putText(img, downCount, (115, 305), font, 2.3, color=[0, 0, 0], thickness=5)
        #cv2.rectangle(img,(iou[0],iou[1]),(iou[2],iou[3]),(0,0,255),2)
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
def trackDraw(img,bbox,identities=None,offset=(0, 0)):
    for i,box in enumerate(bbox):
        x1,y1,x2,y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        id = int(identities[i]) if identities is not None else 0
        color = [255, 255, 0]
        label = '{}{:d}'.format("", id)
        img = plot_one_box([x1, y1, x2, y2], img, color, label)
    return img


def outputManageBefore(output,img,ratio):# output->[1号目标7维向量，2号目标七维向量,...]
    #outputMask = torch.BoolTensor(np.ones(output.shape[0]))

    #只要类别是车的
    outputMask = torch.BoolTensor([(output[i, 6].data == 2. or output[i, 6].data == 3. or output[i, 6].data == 5. or output[i, 6].data == 7.) \
                  for i in range(len(output))])

    #滤除长宽比过大的检测框
    #outputMask = torch.BoolTensor(
    #    [1 if 0. < (outputMeta[2] - outputMeta[0]) / (outputMeta[3] - outputMeta[1]) < 2 else 0 for outputMeta in
    #    output])

    #近距离小目标滤除线
    '''
    cv2.line(img, (0, int(2 / 3 * img.shape[0])), (img.shape[1], int(2 / 3 * img.shape[0])), (255, 0, 0), 2)
    for i,out in enumerate(output):
       if(out[6]!=2.):
           outputMask[i] = False
        positionHeight = int(out[1]/ratio)
        cv2.line(img,(int(out[0]/ratio),0),(int(out[0]/ratio),int(positionHeight)),(0,0,255),2)
        positionThres = int(2 / 3 * img.shape[0])
        if(positionHeight > positionThres):
            cv2.putText(img,"滤除",(int(out[0]/ratio),int(positionHeight)),cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
            output = del_tensor_ele(output,i)
            square = (out[2]-out[0])*(out[3]-out[1])
           if(square<2000):
               outputMask[i] = False
    '''

    output = output[outputMask]
    return output

def vehicleFlowStatis(output,img):
    pass

