import cv2
import numpy as np
print(cv2.__version__)
def showImage(filename,img):
    cv2.imshow(filename,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def drawContours(src_img, mask_img):
    if mask_img.mode is not 'RGB':
        mask_img = mask_img.convert('RGB')
    mask_img = cv2.cvtColor(np.asarray(mask_img), cv2.COLOR_BGR2GRAY)
    ret,binary_img=cv2.threshold(mask_img,1,255,cv2.THRESH_BINARY)
    contours, h= cv2.findContours(binary_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE) 
    rest_img = cv2.drawContours(np.asarray(src_img),contours,-1,(255,0,255),-1)
    # 多边形拟合
    epsilon = 0.02*cv2.arcLength(contours[0], True)
    approx = cv2.approxPolyDP(contours[0], epsilon, True)
    rest_img = cv2.drawContours(np.asarray(src_img),[approx],-1,(255,0,255),-1)
    return rest_img


filename1 = './20210609101453_predict.jpg'
filename2 = './20210609101701_predict.jpg'

img = cv2.imread(filename1)
kernel = np.ones((7, 7), np.int8)
# 
opening = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel=kernel)
opening = cv2.cvtColor(opening, cv2.COLOR_BGR2GRAY)
showImage('result', opening)

# 获取图像的联通区域
num, labels = cv2.connectedComponents(opening)
contours, hierarchy=cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
result = []
for i in range(len(contours)):
    # 如果面积少于多少即不画出来
    result.append(cv2.contourArea(contours[i]))
print('finish debug.')