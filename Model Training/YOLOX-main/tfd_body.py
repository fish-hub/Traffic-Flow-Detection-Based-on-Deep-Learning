#!/usr/bin/env python3
import time
import cv2
import torch
from loguru import logger
import os
import tfd_library
from yolox.data.data_augment import ValTransform
from yolox.data.datasets import COCO_CLASSES
from yolox.data.datasets import VOC_CLASSES
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess, vis
import numpy as np
from deep_sort import DeepSort

fileName = "./Videos/car1.mp4"
yoloxModelPath = "./weights/originbest_ckpt.pth"
yoloxModelName = "yolox-s"
confThres = 0.7
nmsThres = 0.15
imgSize = (640, 640)
device = "gpu"
deepSortCheckPoint = "./weights/ckpt.t7"

class Predictor(object):
    def __init__(self, yoloxModelPath, frame):

        exp = get_exp("./exps/example/yolox_voc/yolox_voc_s.py", yoloxModelName)
        self.testSize = exp.test_size = imgSize
        self.confThres = exp.test_conf = confThres
        self.nmsThres = exp.nmsthre = nmsThres
        self.model = exp.get_model().cuda().eval()
        self.classNames = VOC_CLASSES
        ckpt = torch.load(yoloxModelPath,map_location="cpu")
        self.model.load_state_dict(ckpt["model"], strict=False)
        logger.info("loaded checkpoint done.")
        self.numClasses = exp.num_classes
        self.device = device
        self.preproc = ValTransform(legacy=False)
        self.deepSortforVehicle = DeepSort(deepSortCheckPoint, 0.2)
        self.infTime = 0
        self.trackTime = 0

        imgHeight, imgWidth = frame.shape[0], frame.shape[1]
        mask_image_temp = np.zeros((imgHeight, imgWidth), dtype=np.uint8)
        # 初始化蓝色撞线
        shift = 20
        blue_Height = int(imgHeight * 2. / 3.)
        list_pts_blue = [[0, blue_Height], [imgWidth, blue_Height], [imgWidth, blue_Height + 4 * shift],
                         [0, blue_Height + 4 * shift]]
        ndarray_pts_blue = np.array(list_pts_blue, np.int32)
        polygon_blue_value_1 = cv2.fillPoly(mask_image_temp, [ndarray_pts_blue], color=1)
        polygon_blue_value_1 = polygon_blue_value_1[:, :, np.newaxis]

        # 初始化黄色撞线
        mask_image_temp = np.zeros((imgHeight, imgWidth), dtype=np.uint8)
        blue_Height = int(imgHeight * 2. / 3.)
        list_pts_yellow = [[0, blue_Height + 8 * shift], [imgWidth, blue_Height + 8 * shift],
                           [imgWidth, blue_Height + 12 * shift], [0, blue_Height + 12 * shift]]
        ndarray_pts_yellow = np.array(list_pts_yellow, np.int32)
        polygon_blue_value_2 = cv2.fillPoly(mask_image_temp, [ndarray_pts_yellow], color=2)
        polygon_blue_value_2 = polygon_blue_value_2[:, :, np.newaxis]

        # 撞线检测mask合并
        self.polygon_mask_blue_and_yellow = polygon_blue_value_1 + polygon_blue_value_2
        blue_color_plate = [255, 0, 0]
        blue_image = np.array(polygon_blue_value_1 * blue_color_plate, np.uint8)
        yellow_color_plate = [0, 255, 255]
        yellow_image = np.array(polygon_blue_value_2 * yellow_color_plate, np.uint8)
        self.color_polygons_image = blue_image + yellow_image

        self.list_overlapping_blue_polygon = []
        self.down_count = 0



    def inference(self,img):
        imgInfo = {"id": 0}
        if isinstance(img,str):
            imgInfo["file_name"] = os.path.basename(img)
            img = cv2.imread(img)
        else:
            imgInfo["file_name"] = None
        height, width = img.shape[:2]
        ratio = min(self.testSize[0] / img.shape[0], self.testSize[1] / img.shape[1])
        imgInfo["height"] = height
        imgInfo["width"] = width
        imgInfo["raw_img"] = img
        imgInfo["ratio"] = ratio

        img, _ = self.preproc(img, None, self.testSize)
        img = torch.from_numpy(img).unsqueeze(0).float()
        if self.device == "gpu":
            img = img.cuda()
        with torch.no_grad():
            t0 = time.time()
            output = self.model(img)
            output = postprocess(output,self.numClasses,self.confThres,self.nmsThres,class_agnostic=True)
        #logger.info("Infer time：{:.4f}s".format(time.time()-t0))
        self.infTime = time.time()-t0
        #output[0] = tfd_library.outputManageBefore(output[0], imgInfo["raw_img"], imgInfo["ratio"])
        logger.info(output[0].shape)
        return output,imgInfo

    def vehicleTrack(self, input, img_info,img):
        input = input.cpu()
        bboxes = input[:, :4]
        ratio = img_info["ratio"]
        img = img
        bboxes/=ratio
        confidence = input[:, 4] * input[:, 5]
        t0 = time.time()
        if(bboxes is not None):
            bbox_cxcywh = tfd_library.xyxy2xywh(bboxes)
            output = self.deepSortforVehicle.update(bbox_cxcywh, confidence, img)
            if(len(output)>0):
                bbox_xyxy = output[:, :4]
                ids = output[:, -1]
                img = tfd_library.trackDraw(img, bbox_xyxy, ids)

                for item_box in output:
                    x1,y1,x2,y2,track_id = item_box
                    y1_offset = int(y1 + ((y2 - y1) * 0.6))
                    # 撞线检测点
                    x = x1
                    y = y1_offset
                    if(self.polygon_mask_blue_and_yellow[y,x]==1):#撞了蓝线
                        if(track_id not in self.list_overlapping_blue_polygon):
                            self.list_overlapping_blue_polygon.append(track_id)
                    if(self.polygon_mask_blue_and_yellow[y,x]==2):
                        if(track_id in self.list_overlapping_blue_polygon):
                            self.down_count += 1
                            self.list_overlapping_blue_polygon.remove(track_id)
                            print("检测到一个车辆向下")

        img = cv2.add(img,self.color_polygons_image)
        self.trackTime = time.time() - t0
        return img
    def visualize(self,output,img_info):
        output_ = output.cpu()
        bboxes = output_[:, :4]
        ratio = img_info["ratio"]
        img = img_info["raw_img"]
        cls = output_[:, 6]
        bboxes /= ratio
        confidence = output_[:, 4] * output_[:, 5]
        tfd_library.vehicleFlowStatis(output,img)
        if(len(bboxes)>0):
            visres = tfd_library.objDraw(img,bboxes,confidence,cls,self.trackTime,self.infTime,self.down_count,self.confThres,self.classNames, externalFlag=0,drawObjDetectionTarget=1)
            return visres
        else:
            return img
if __name__ == "__main__":
    cap = cv2.VideoCapture(fileName)
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    currentFrameIndex = 0
    retVal, frame = cap.read()
    predictor = Predictor(yoloxModelPath,frame)
    while True:
        retVal, frame = cap.read()
        if(retVal):
            if((frames-1) == currentFrameIndex):
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            yoloxOutput,img_info = predictor.inference(frame)
            result_frame = predictor.visualize(yoloxOutput[0], img_info)
            #result_frame = predictor.vehicleTrack(yoloxOutput[0], img_info,result_frame)

            result_frame = cv2.resize(result_frame,(1200,700))
            cv2.imshow("resultimg",result_frame)
            ch = cv2.waitKey(10)
            currentFrameIndex+=1
            if ch == 27:
                break
        else:
            break

