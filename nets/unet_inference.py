import colorsys
import copy
import time

import numpy as np
from PIL import Image

from nets.unet import Unet as unet
import onnx
import onnxruntime

class Unet(object):

    #---------------------------------------------------#
    #   初始化UNET
    #---------------------------------------------------#
    def __init__(self, **kwargs):
        _defaults = {
            "model_path"        : kwargs['model'],
            "model_image_size"  : kwargs['model_image_size'],
            "num_classes"       : kwargs['num_classes']
        }
        self.onnx_model = kwargs['onnxmodel']
        self.__dict__.update(_defaults)
        self.generate()

    def generate(self):
        #-------------------------------#
        #   载入模型与权值
        #-------------------------------#
        self.model = unet(self.model_image_size, self.num_classes)

        self.model.load_weights(self.model_path)
        print('{} model loaded.'.format(self.model_path))

        if self.num_classes <= 21:
            self.colors = [
                # B-ground, Aero plane, Bicycle
                (0, 0, 0), (128, 0, 0), (0, 128, 0), 
                #Bird,          Boat,       Bottle
                (128, 128, 0), (0, 0, 128), (128, 0, 128), 
                #Bus,           Car,            Cat
                (0, 128, 128), (128, 128, 128), (64, 0, 0), 
                #Chair          Cow         Dining-Table
                (192, 0, 0), (64, 128, 0), (192, 128, 0), 
                #Dog            Horse           Motorbike
                (64, 0, 128), (192, 0, 128), (64, 128, 128), 
                #Person,         Potted-Plant, Sheep
                (192, 128, 128), (0, 64, 0), (128, 64, 0),
                #Sofa         Train         TV/Monitor
                (0, 192, 0), (128, 192, 0), (0, 64, 128),
            ]
        else:
            # 画框设置不同的颜色
            hsv_tuples = [(x / len(self.class_names), 1., 1.)
                        for x in range(len(self.class_names))]
            self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
            self.colors = list(
                map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                    self.colors))

    def letterbox_image(self ,image, size):
        image = image.convert("RGB")
        iw, ih = image.size
        w, h = size
        scale = min(w/iw, h/ih)
        nw = int(iw*scale)
        nh = int(ih*scale)

        image = image.resize((nw,nh), Image.BICUBIC)
        new_image = Image.new('RGB', size, (128,128,128))
        new_image.paste(image, ((w-nw)//2, (h-nh)//2))
        return new_image,nw,nh

    def detect_image(self, image):
        image = image.convert('RGB')
        old_img = copy.deepcopy(image)
        orininal_h = np.array(image).shape[0]
        orininal_w = np.array(image).shape[1]
        img, nw, nh = self.letterbox_image(image,(self.model_image_size[1],self.model_image_size[0]))
        img = np.asarray([np.array(img).astype(np.float32) /255])


        # 使用模型进行推理预测
        pr = self.model.predict(img)[0]
        pr = pr.argmax(axis=-1).reshape([self.model_image_size[0],self.model_image_size[1]])
        pr = pr[int((self.model_image_size[0]-nh)//2):int((self.model_image_size[0]-nh)//2+nh), int((self.model_image_size[1]-nw)//2):int((self.model_image_size[1]-nw)//2+nw)]

        seg_img = np.zeros((np.shape(pr)[0],np.shape(pr)[1],3))
        for c in range(self.num_classes):
            seg_img[:,:,0] += ((pr[:,: ] == c )*( self.colors[c][0] )).astype('uint8')
            seg_img[:,:,1] += ((pr[:,: ] == c )*( self.colors[c][1] )).astype('uint8')
            seg_img[:,:,2] += ((pr[:,: ] == c )*( self.colors[c][2] )).astype('uint8')

        image = Image.fromarray(np.uint8(seg_img)).resize((orininal_w,orininal_h), Image.NEAREST)
        blend_image = Image.blend(old_img,image,0.3)

        return image, blend_image

    def get_FPS(self, image, test_interval):
        orininal_h = np.array(image).shape[0]
        orininal_w = np.array(image).shape[1]

        img, nw, nh = self.letterbox_image(image,(self.model_image_size[1],self.model_image_size[0]))
        img = np.asarray([np.array(img)/255])

        pr = self.model.predict(img)[0]
        pr = pr.argmax(axis=-1).reshape([self.model_image_size[0],self.model_image_size[1]])
        pr = pr[int((self.model_image_size[0]-nh)//2):int((self.model_image_size[0]-nh)//2+nh), int((self.model_image_size[1]-nw)//2):int((self.model_image_size[1]-nw)//2+nw)]
        
        image = Image.fromarray(np.uint8(pr)).resize((orininal_w,orininal_h), Image.NEAREST)

        t1 = time.time()
        for _ in range(test_interval):
            pr = self.model.predict(img)[0]
            pr = pr.argmax(axis=-1).reshape([self.model_image_size[0],self.model_image_size[1]])
            pr = pr[int((self.model_image_size[0]-nh)//2):int((self.model_image_size[0]-nh)//2+nh), int((self.model_image_size[1]-nw)//2):int((self.model_image_size[1]-nw)//2+nw)]
            image = Image.fromarray(np.uint8(pr)).resize((orininal_w,orininal_h), Image.NEAREST)
            
        t2 = time.time()
        tact_time = (t2 - t1) / test_interval
        return tact_time
        
