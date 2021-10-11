import time
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from nets.unet_inference import Unet
import configparser
from glob import glob
from tqdm import tqdm
import onnxruntime

area_threshold = 300

config = configparser.ConfigParser()
config.read('./unet_config.cfg')


def drawContours(src_img, mask_img):
    if mask_img.mode is not 'RGB':
        mask_img = mask_img.convert('RGB')
    mask_img = cv2.cvtColor(np.asarray(mask_img), cv2.COLOR_BGR2GRAY)
    # 先进行形态学的处理，尽可能减少区域
    kernel = np.ones((7, 7), np.int8)
    mask_img = cv2.morphologyEx(mask_img, cv2.MORPH_CLOSE, kernel=kernel)
    ret,binary_img=cv2.threshold(mask_img,1,255,cv2.THRESH_BINARY)
    # 画出轮廓
    '''
    mode: 轮廓检索模式
        RETE_EXTERNAL：只检索最外层的轮廓，返回值会设置所有hierarchy[i][2]=hierarchy[i][3]=-1
        RETR_LIST：检索所有的轮廓，但不建立轮廓间的层次关系(hierarchy relationship)
        RETR_TREE：检测所有的轮廓，但只建立两个等级关系，外围为顶层，若外围内轮廓还包含了其他的轮廓信息，则内围内的所有轮廓均归属于顶层，只有内围轮廓不再包含子轮廓，其为内层。
        PRETE_TREE：检索所有轮廓，所有轮廓建立一个等级树结构，外层轮廓包含内层轮廓，内层轮廓还可以继续包含内嵌轮廓。
    method：
        CHAIN_APPROX_NONE: 保存物体边界上所有连续的轮廓点到contours中，即点(x1,y1)和点(x2,y2)，满足max(abs(x1-x2),abs(y2-y1))==1，则认为其是连续的轮廓点
        CHAIN_APPROX_SIMPLE: 仅保存轮廓的拐点信息到contours，拐点与拐点之间直线段上的信息点不予保留
        CHAIN_APPROX_TC89_L1: 采用Teh-Chin chain近似算法 
        CHAIN_APPROX_TC89_KCOS:采用Teh-Chin chain近似算法
    '''
    contours, h= cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 与阈值比较轮廓包围的面积
    contours_result = []
    temp = []
    for item in contours:
        if  cv2.contourArea(item)>area_threshold:
            contours_result.append(item)
            temp.append(cv2.contourArea(item))
    rest_img = cv2.drawContours(np.asarray(src_img),contours,-1,(255,0,255),3)
    return rest_img

if __name__ == "__main__":
    img_size = config.getint('default', 'imgsize')
    unet = Unet(
        num_classes=config.getint('default', 'num_class'), 
        model=config.get('predict', 'model'),
        model_image_size = (img_size, img_size, 3),
        onnxmodel = config.get('train', 'onnx_model')
    )
    mode = config.get('predict', 'mode')


    if mode == "predict":
        for img in tqdm(glob(config.get('predict', 'test_path')+'*')):
            print(img)
        # img = config.get('predict', 'test_path')+config.get('predict', 'test_img')
        # groundTruth = './buildings_VOC/SegmentationClassPNG/20210531094431.png'
        # gt_img = Image.open(groundTruth)
            try:
                image = Image.open(img)
            except:
                print('Open Error! Try again!')
            else:
                r_image, blend_img = unet.detect_image(image)
                # label_img = drawContours(image, gt_img)
                predict_img = drawContours(image, r_image)
                plt.figure()
                savefilename =  img.split('\\')[-1].split('.')[0]
                # 展示原图
                plt.subplot(1, 4, 1)
                plt.imshow(image)
                plt.title('original')
                # image.save('./result/'+savefilename+'_ori.jpg')
                # 展示预测的gt图
                plt.subplot(1, 4, 2)
                plt.imshow(r_image)
                r_image.save('./result/'+savefilename+'_predict.jpg', quality=100,format='png', subsampling=0)
                plt.title('predict')
                # 展示原图与预测的合并图
                plt.subplot(1, 4, 3)
                plt.imshow(blend_img)
                blend_img.save('./result/'+savefilename+'_blend.jpg', quality=100,format='png', subsampling=0)
                plt.title('blend_img')
                plt.subplot(1, 4, 4)
                plt.imshow(predict_img)
                plt.title('predict_img')
                predict_img = Image.fromarray(predict_img)
                predict_img.save('./result/'+savefilename+'_Contour.jpg')
                plt.subplots_adjust(top=1,bottom=0,left=0,right=1,hspace=0,wspace=0)
                plt.margins(0,0)
                plt.savefig(f'./result/{savefilename}_result.jpg')
                plt.show()
            

    elif mode == "video":
        video_path      = config.get('predict', 'videopath')
        video_save_path = ""
        video_fps       = 25.0
        capture=cv2.VideoCapture(video_path)
        if video_save_path!="":
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            out = cv2.VideoWriter(video_save_path, fourcc, video_fps, size)

        fps = 0.0
        while(True):
            t1 = time.time()
            ref,frame=capture.read()
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(np.uint8(frame))
            frame = np.array(unet.detect_image(frame))
            # RGBtoBGR满足opencv显示格式
            frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
            
            fps  = ( fps + (1./(time.time()-t1)) ) / 2
            print("fps= %.2f"%(fps))
            frame = cv2.putText(frame, "fps= %.2f"%(fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow("video",frame)
            c= cv2.waitKey(1) & 0xff 
            if video_save_path!="":
                out.write(frame)

            if c==27:
                capture.release()
                break
        capture.release()
        out.release()
        cv2.destroyAllWindows()

    elif mode == "fps":
        test_interval = 100
        img = Image.open('img/street.jpg')
        tact_time = unet.get_FPS(img, test_interval)
        print(str(tact_time) + ' seconds, ' + str(1/tact_time) + 'FPS, @batch_size 1')
    else:
        raise AssertionError("Please specify the correct mode: 'predict', 'video' or 'fps'.")

