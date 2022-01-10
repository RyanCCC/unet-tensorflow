import os
from random import shuffle
import numpy as np
from PIL import Image
import cv2


class UnetDatasetGenerator(object):
    def __init__(self,batch_size,train_lines,image_size,num_classes,dataset_path):
        self.batch_size     = batch_size
        self.train_lines    = train_lines
        self.train_batches  = len(train_lines)
        self.image_size     = image_size
        self.num_classes    = num_classes
        self.dataset_path   = dataset_path

        
    def __call__(self):
        i = 0
        length = len(self.train_lines)
        inputs = []
        targets = []
        while True:
            if i == 0:
                shuffle(self.train_lines)
            annotation_line = self.train_lines[i]
            name = annotation_line.split()[0]

            # 从文件中读取图像
            jpg = Image.open(os.path.join(os.path.join(self.dataset_path, "JPEGImages"), name + ".jpg"))
            png = Image.open(os.path.join(os.path.join(self.dataset_path, "labels"), name + ".png"))

            jpg, png = letterbox_image(jpg, png, (int(self.image_size[1]),int(self.image_size[0])))
            
            inputs.append(np.array(jpg)/255)
            
            png = np.array(png)
            png[png >= self.num_classes] = self.num_classes
            seg_labels = np.eye(self.num_classes+1)[png.reshape([-1])]
            seg_labels = seg_labels.reshape((int(self.image_size[1]),int(self.image_size[0]),self.num_classes+1))
            
            targets.append(seg_labels)
            i = (i + 1) % length
            if len(targets) == self.batch_size:
                tmp_inp = np.array(inputs)
                tmp_targets = np.array(targets)
                inputs = []
                targets = []
                yield tmp_inp, tmp_targets


def letterbox_image(image, label , size):
    label = Image.fromarray(np.array(label))

    '''resize image with unchanged aspect ratio using padding'''
    iw, ih = image.size
    w, h = size
    scale = min(w/iw, h/ih)
    nw = int(iw*scale)
    nh = int(ih*scale)

    image = image.resize((nw,nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128,128,128))
    new_image.paste(image, ((w-nw)//2, (h-nh)//2))

    label = label.resize((nw,nh), Image.NEAREST)
    new_label = Image.new('L', size, (0))
    new_label.paste(label, ((w-nw)//2, (h-nh)//2))
    return new_image, new_label