import onnxruntime
import cv2
import numpy as np
from PIL import Image
import copy

img = './image/test.jpg'
onnx_model = './model/citybuildings.onnx'

def letterbox_image(image, size):
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

model_image_size = (512, 512)

# 加载onnx模型
sess = onnxruntime.InferenceSession(onnx_model)

# 读取待检测图像并进行预处理
image = Image.open(img)
old_img = copy.deepcopy(image)
orininal_h = np.array(image).shape[0]
orininal_w = np.array(image).shape[1]
img, nw, nh = letterbox_image(image,model_image_size)
img = np.asarray([np.array(img).astype(np.float32) /255])

# 使用模型进行推理
x = img if isinstance(img, list) else [img]
feed = dict([(input.name, x[n]) for n, input in enumerate(sess.get_inputs())])
pred_onnx = sess.run(None, feed)[0]
pr = np.squeeze(pred_onnx)
pr = pr.argmax(axis=-1).reshape([model_image_size[0],model_image_size[1]])
pr = pr[int((model_image_size[0]-nh)//2):int((model_image_size[0]-nh)//2+nh), int((model_image_size[1]-nw)//2):int((model_image_size[1]-nw)//2+nw)]

seg_img = np.zeros((np.shape(pr)[0],np.shape(pr)[1],3))

# 推理后处理
colors = [
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
num_class = 2
seg_img = np.zeros((np.shape(pr)[0],np.shape(pr)[1],3))
for c in range(num_class):
    seg_img[:,:,0] += ((pr[:,: ] == c )*( colors[c][0] )).astype('uint8')
    seg_img[:,:,1] += ((pr[:,: ] == c )*( colors[c][1] )).astype('uint8')
    seg_img[:,:,2] += ((pr[:,: ] == c )*( colors[c][2] )).astype('uint8')

image = Image.fromarray(np.uint8(seg_img)).resize((orininal_w,orininal_h), Image.NEAREST)
blend_image = Image.blend(old_img,image,0.3)
blend_image.show()