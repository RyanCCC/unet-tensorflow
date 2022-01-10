from os.path import join
import numpy as np
from PIL import Image


# 设标签宽W，长H
def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)  

def per_class_iu(hist):
    return np.diag(hist) / np.maximum((hist.sum(1) + hist.sum(0) - np.diag(hist)), 1) 

def per_class_PA(hist):
    return np.diag(hist) / np.maximum(hist.sum(1), 1) 

def compute_mIoU(gt_dir, pred_dir, png_name_list, num_classes, name_classes):  
    print('Num classes', num_classes)  

    hist = np.zeros((num_classes, num_classes))
    gt_imgs = [join(gt_dir, x + ".png") for x in png_name_list]  
    pred_imgs = [join(pred_dir, x + ".png") for x in png_name_list]  

    for ind in range(len(gt_imgs)): 
        pred = np.array(Image.open(pred_imgs[ind]))  
        label = np.array(Image.open(gt_imgs[ind]))  

        # 如果图像分割结果与标签的大小不一样，这张图片就不计算
        if len(label.flatten()) != len(pred.flatten()):  
            print(
                'Skipping: len(gt) = {:d}, len(pred) = {:d}, {:s}, {:s}'.format(
                    len(label.flatten()), len(pred.flatten()), gt_imgs[ind],
                    pred_imgs[ind]))
            continue

        hist += fast_hist(label.flatten(), pred.flatten(),num_classes)  
        # 每计算10张就输出一下目前已计算的图片中所有类别平均的mIoU值
        if ind > 0 and ind % 10 == 0:  
            print('{:d} / {:d}: mIou-{:0.2f}; mPA-{:0.2f}'.format(ind, len(gt_imgs),
                                                    100 * np.nanmean(per_class_iu(hist)),
                                                    100 * np.nanmean(per_class_PA(hist))))
    mIoUs   = per_class_iu(hist)
    mPA     = per_class_PA(hist)
    for ind_class in range(num_classes):
        print('===>' + name_classes[ind_class] + ':\tmIou-' + str(round(mIoUs[ind_class] * 100, 2)) + '; mPA-' + str(round(mPA[ind_class] * 100, 2)))
    print('===> mIoU: ' + str(round(np.nanmean(mIoUs) * 100, 2)) + '; mPA: ' + str(round(np.nanmean(mPA) * 100, 2)))  
    return mIoUs


if __name__ == "__main__":
    gt_dir = "VOCdevkit/VOC2007/SegmentationClass"
    pred_dir = "miou_pr_dir"
    png_name_list = open("VOCdevkit/VOC2007/ImageSets/Segmentation/val.txt",'r').read().splitlines() 
    num_classes = 21
    name_classes = ["background","aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
    compute_mIoU(gt_dir, pred_dir, png_name_list, num_classes, name_classes)  # 执行计算mIoU的函数
