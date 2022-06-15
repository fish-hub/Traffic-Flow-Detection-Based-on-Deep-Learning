#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Code are based on
# https://github.com/fmassa/vision/blob/voc_dataset/torchvision/datasets/voc.py
# Copyright (c) Francisco Massa.
# Copyright (c) Ellis Brown, Max deGroot.
# Copyright (c) Megvii, Inc. and its affiliates.

import os
import os.path
import pickle
import xml.etree.ElementTree as ET
from loguru import logger

import cv2
import numpy as np

from yolox.evaluators.voc_eval import voc_eval

from .datasets_wrapper import Dataset
from .voc_classes import VOC_CLASSES
import torch

class AnnotationTransform(object):

    """Transforms a VOC annotation into a Tensor of bbox coords and label index
    Initilized with a dictionary lookup of classnames to indexes

    Arguments:
        class_to_ind (dict, optional): dictionary lookup of classnames -> indexes
            (default: alphabetic indexing of VOC's 20 classes)
        keep_difficult (bool, optional): keep difficult instances or not
            (default: False)
        height (int): height
        width (int): width
    """

    def __init__(self, class_to_ind=None, keep_difficult=True):
        self.class_to_ind = class_to_ind or dict(
            zip(VOC_CLASSES, range(len(VOC_CLASSES))) # 返回（'car',0）
        )
        self.keep_difficult = keep_difficult

    def __call__(self, target): #target传入打开xml文件的总父亲
        """
        Arguments:
            target (annotation) : the target annotation to be made usable
                will be an ET.Element
        Returns:
            a list containing lists of bounding boxes  [bbox coords, class name]
        """
        res = np.empty((0, 5))
        #找到xml中object标签
        for obj in target.iter("object"):
            #找到object中difficult标签
            difficult = obj.find("difficult")
            #如果xml中difficult标签不为空，将difficult标签内容取出
            if difficult is not None:
                difficult = int(difficult.text) == 1
            #如果对应object没有difficult标签则设置为false
            else:
                difficult = False
            #判断如果keepdifficult为0且difficult为0则直接进入下一次循环遍历下一个obj
            if not self.keep_difficult and difficult:
                continue
            name = obj.find("name").text.strip() #得到obj的label
            bbox = obj.find("bndbox")#得到obj的anchor根

            pts = ["xmin", "ymin", "xmax", "ymax"]
            bndbox = []
            #从anchor根得到两个点的坐标并放到bndbox列表中
            for i, pt in enumerate(pts):
                cur_pt = int(bbox.find(pt).text) - 1
                # scale height or width
                # cur_pt = cur_pt / width if i % 2 == 0 else cur_pt / height
                bndbox.append(cur_pt)
            #根据label得到对应的index
            label_idx = self.class_to_ind[name]
            #把labelindex也放到bndbox中
            bndbox.append(label_idx)
            #垂直方向存储得到很多行，每一行是一个目标的坐标和类别信息
            res = np.vstack((res, bndbox))  # [xmin, ymin, xmax, ymax, label_index]
            # img_id = target.find('filename').text[:-4]
        #读取当前图片的宽度和高度信息
        width = int(target.find("size").find("width").text)
        height = int(target.find("size").find("height").text)
        img_info = (height, width)
        #返回当前xml中object的位置和类别  imginfo为当前图片的宽度和高度
        return res, img_info


class VOCDetection(Dataset):

    """
    VOC Detection Dataset Object

    input is image, target is annotation

    Args:
        root (string): filepath to VOCdevkit folder.
        image_set (string): imageset to use (eg. 'train', 'val', 'test')
        transform (callable, optional): transformation to perform on the
            input image
        target_transform (callable, optional): transformation to perform on the
            target `annotation`
            (eg: take in caption string, return tensor of word indices)
        dataset_name (string, optional): which dataset to load
            (default: 'VOC2007')
    """

    def __init__(
        self,
        data_dir,
        image_sets=[("2007", "trainval"), ("2012", "trainval")],
        img_size=(416, 416),
        preproc=None,
        target_transform=AnnotationTransform(),
        dataset_name="VOC0712",
        cache=False,
    ):
        super().__init__(img_size)
        self.root = data_dir #datasets/voc2007
        self.image_set = image_sets # ["train"]
        self.img_size = img_size #640x640
        self.preproc = preproc #traintransform()
        self.target_transform = target_transform #annotationtransform
        self.name = dataset_name #"VOC0712"
        self._annopath = os.path.join("%s", "Annotations", "%s.xml")
        self._imgpath = os.path.join("%s", "JPEGImages", "%s.jpg")
        self._classes = VOC_CLASSES #只有一个(car,)了
        self.ids = list()
        for name in image_sets:
            rootpath = self.root #datasets/voc2007
            #打开datasets/voc2007/imagesets/main/train.txt并将每一行的数据写入ids列表如【（datasets/voc2007，picture1），】
            for line in open(
                os.path.join(rootpath, "ImageSets", "Main", name + ".txt")
            ):
                self.ids.append((rootpath, line.strip())) #
        #拿到标注信息
        #得到一个列表，包含很多元祖，每一个元祖为一个xml文件所对应的目标内容，内容为（objs位置+类别，图像大小，图像放缩后大小）
        self.annotations = self._load_coco_annotations()
        self.imgs = None
        if cache:
            self._cache_images()

    def __len__(self):
        return len(self.ids)

    def _load_coco_annotations(self):
        return [self.load_anno_from_ids(_ids) for _ids in range(len(self.ids))]

    def _cache_images(self):
        logger.warning(
            "\n********************************************************************************\n"
            "You are using cached images in RAM to accelerate training.\n"
            "This requires large system RAM.\n"
            "Make sure you have 60G+ RAM and 19G available disk space for training VOC.\n"
            "********************************************************************************\n"
        )
        max_h = self.img_size[0]
        max_w = self.img_size[1]
        cache_file = os.path.join(self.root, f"img_resized_cache_{self.name}.array")
        if not os.path.exists(cache_file):
            logger.info(
                "Caching images for the first time. This might take about 3 minutes for VOC"
            )
            self.imgs = np.memmap(
                cache_file,
                shape=(len(self.ids), max_h, max_w, 3),
                dtype=np.uint8,
                mode="w+",
            )
            from tqdm import tqdm
            from multiprocessing.pool import ThreadPool

            NUM_THREADs = min(8, os.cpu_count())
            loaded_images = ThreadPool(NUM_THREADs).imap(
                lambda x: self.load_resized_img(x),
                range(len(self.annotations)),
            )
            pbar = tqdm(enumerate(loaded_images), total=len(self.annotations))
            for k, out in pbar:
                self.imgs[k][: out.shape[0], : out.shape[1], :] = out.copy()
            self.imgs.flush()
            pbar.close()
        else:
            logger.warning(
                "You are using cached imgs! Make sure your dataset is not changed!!\n"
                "Everytime the self.input_size is changed in your exp file, you need to delete\n"
                "the cached data and re-generate them.\n"
            )

        logger.info("Loading cached imgs...")
        self.imgs = np.memmap(
            cache_file,
            shape=(len(self.ids), max_h, max_w, 3),
            dtype=np.uint8,
            mode="r+",
        )

    def load_anno_from_ids(self, index):
        img_id = self.ids[index] #得到ids列表中每个元素，元素为（datasets/voc2007,picturex）
        #getroot得到打开的xml文件的总父亲
        target = ET.parse(self._annopath % img_id).getroot() #self._annopath % img_id ->datasets/voc2007/Annotations/picturex.xml

        assert self.target_transform is not None
        res, img_info = self.target_transform(target) #返回当前xml的目标位置+类别index 还有图像的宽度和高度
        #拿到当前图像的高宽
        height, width = img_info
        #获得640和当前图像高宽比值的最小值，即一个图像放缩比例
        r = min(self.img_size[0] / height, self.img_size[1] / width)
        # 因为图像得放缩，所以将锚框位置进行放缩
        res[:, :4] *= r
        #拿到放缩后的图像大小
        resized_info = (int(height * r), int(width * r))

        # res返回当前xml的目标位置+类别index【【obj1位置，obj1类别】，【obj2位置】，【】】
        # img_info 返回当前图像的尺寸
        #resized_info 返回对图像进行放缩后的尺寸，放缩至图像最大边都小于640
        return (res, img_info, resized_info)


    def load_anno(self, index):
        return self.annotations[index][0]

    def load_resized_img(self, index):
        # cv2读取单张图片
        img = self.load_image(index)
        # 拿到图像放缩的最小ratio
        r = min(self.img_size[0] / img.shape[0], self.img_size[1] / img.shape[1])
        # 对图像进行放缩
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * r), int(img.shape[0] * r)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)
        #返回放缩到最长边小于640的图像
        return resized_img

    def load_image(self, index):
        img_id = self.ids[index]
        # /datasets/voc2007/jpegimages/picturename
        img = cv2.imread(self._imgpath % img_id, cv2.IMREAD_COLOR)
        assert img is not None

        return img

    def pull_item(self, index):
        """Returns the original image and target at an index for mixup

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            img, target
        """
        #cache为True且有cacheimg才能走这
        if self.imgs is not None:
            target, img_info, resized_info = self.annotations[index]
            pad_img = self.imgs[index]
            img = pad_img[: resized_info[0], : resized_info[1], :].copy()
        else:
            img = self.load_resized_img(index) #拿到放缩后的图片，（最长边小于640）
            target, img_info, _ = self.annotations[index] # 从init中写好的标注信息拿到当前图片的obj位置+类别，图像大小，缩放后大小

        return img, target, img_info, index# index--从ids中取图片的名字用

    @Dataset.mosaic_getitem
    def __getitem__(self, index):
        # cv读取的图片，当前图片obj位置+类别，图像的原始大小，图像在ids中对应的index
        img, target, img_info, img_id = self.pull_item(index)

        #进行图像与标注信息的transform
        #返回对图像进行缩放和填充得到cx640x640的图像，当前图像中的maxlabel（单张图片最大限制obj个数）个数的obj的label和obj缩放后的位置
        if self.preproc is not None:
            img, target = self.preproc(img, target, self.input_dim)#推断inputdim应该是640x640

        return img, target, img_info, img_id

    def evaluate_detections(self, all_boxes, output_dir=None):
        """
        all_boxes is a list of length number-of-classes.
        Each list element is a list of length number-of-images.
        Each of those list elements is either an empty list []
        or a numpy array of detection.

        all_boxes[class][image] = [] or np.array of shape #dets x 5
        """
        self._write_voc_results_file(all_boxes)
        IouTh = np.linspace(
            0.5, 0.95, int(np.round((0.95 - 0.5) / 0.05)) + 1, endpoint=True
        )
        mAPs = []
        maxRec,maxPrec=0,0
        # 计算每个IOU下的mAP-->mmAP
        for iou in IouTh:
            #mAP = self._do_python_eval(output_dir, iou)
            mAP = self._do_python_eval(output_dir, iou)
            mAPs.append(mAP)
        print("--------------------------------------------------------------")
        print("map_5095:", np.mean(mAPs))
        print("map_50:", mAPs[0])
        print("--------------------------------------------------------------")
        #return np.mean(mAPs), mAPs[0]
        return np.mean(mAPs), mAPs[0]

    def _get_voc_results_file_template(self):
        filename = "comp4_det_test" + "_{:s}.txt"
        filedir = os.path.join(self.root, "results")
        if not os.path.exists(filedir):
            os.makedirs(filedir)
        path = os.path.join(filedir, filename)
        return path

    def _write_voc_results_file(self, all_boxes):
        for cls_ind, cls in enumerate(VOC_CLASSES):
            cls_ind = cls_ind
            if cls == "__background__":
                continue
            print("Writing {} VOC results file".format(cls))
            filename = self._get_voc_results_file_template().format(cls)
            with open(filename, "wt") as f:
                for im_ind, index in enumerate(self.ids):
                    index = index[1]
                    dets = all_boxes[cls_ind][im_ind]
                    if dets == []:
                        continue
                    for k in range(dets.shape[0]):
                        f.write(
                            "{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n".format(
                                index,
                                dets[k, -1],
                                dets[k, 0] + 1,
                                dets[k, 1] + 1,
                                dets[k, 2] + 1,
                                dets[k, 3] + 1,
                            )
                        )

    def _do_python_eval(self, output_dir="output", iou=0.5):
        rootpath = self.root
        name = self.image_set[0]
        annopath = os.path.join(rootpath, "Annotations", "{:s}.xml")
        imagesetfile = os.path.join(rootpath, "ImageSets", "Main", name + ".txt")
        cachedir = os.path.join(
            self.root, "annotations_cache"
        )
        if not os.path.exists(cachedir):
            os.makedirs(cachedir)
        aps = []
        # The PASCAL VOC metric changed in 2010
        use_07_metric = True
        print("Eval IoU : {:.2f}".format(iou))
        if output_dir is not None and not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        for i, cls in enumerate(VOC_CLASSES):

            if cls == "__background__":
                continue

            filename = self._get_voc_results_file_template().format(cls)
            rec, prec, ap = voc_eval(
                filename,
                annopath,
                imagesetfile,
                cls,
                cachedir,
                ovthresh=iou,
                use_07_metric=use_07_metric,
            )
            aps += [ap]
            if iou == 0.5:
                print("AP for {} = {:.4f}".format(cls, ap))
            if output_dir is not None:
                with open(os.path.join(output_dir, cls + "_pr.pkl"), "wb") as f:
                    pickle.dump({"rec": rec, "prec": prec, "ap": ap}, f)
        if iou == 0.5:
            print("Mean AP = {:.4f}".format(np.mean(aps)))
            print("~~~~~~~~")
            print("Results:")
            for ap in aps:
                print("{:.3f}".format(ap))
            print("{:.3f}".format(np.mean(aps)))
            print("~~~~~~~~")
            print("")
            print("--------------------------------------------------------------")
            print("Results computed with the **unofficial** Python eval code.")
            print("Results should be very close to the official MATLAB eval code.")
            print("Recompute with `./tools/reval.py --matlab ...` for your paper.")
            print("-- Thanks, The Management")
            print("--------------------------------------------------------------")
        _, index = torch.max(torch.tensor(prec),0)
        #return np.mean(aps)
        return np.mean(aps)
