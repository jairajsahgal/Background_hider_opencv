import cv2
import cvzone
from cvzone.SelfiSegmentationModule import SelfiSegmentation
import os
import numpy as np

arr = os.listdir("./Images/")
print(arr)
arr.remove(".DS_Store")
for i in range(len(arr)):

    arr[i]="./Images/"+arr[i]
print(arr)
dim = (1280, 720)
# os.chdir("./Images/")
for i in arr:
    image = cv2.imread(i, 1)
    # print(image)
    # Loading the image
    half = cv2.resize(image, dim,interpolation=cv2.INTER_AREA)
    cv2.imwrite(i,half)

arr = os.listdir("./Images/")
arr.pop(0)
imgList = []
for i in range(len(arr)):
    img = cv2.imread(f'./Images/{arr[i]}')
    imgList.append(img)
    arr[i]="./Images/"+arr[i]
print(len(imgList))

cap = cv2.VideoCapture(0)
segmentor = SelfiSegmentation(model=0)
fpsReader = cvzone.FPS()
indexImg = 0
while True:
    success,img = cap.read()
    bg_image = cv2.GaussianBlur(img, (55, 55), 0)
    imgList.append(bg_image)
    print(indexImg)
    imgOut = segmentor.removeBG(img,imgList[indexImg],threshold=0.5)
    imgStacked = cvzone.stackImages([img,imgOut],2,1)
    cv2.imshow("Image",imgStacked)
    key = cv2.waitKey(1)
    if key == ord('a'):
        if indexImg>0:
            indexImg-=1
    elif key == ord('d'):
        if indexImg < len(imgList)-1:
            indexImg+=1
    elif key == ord('q'):
        break
    imgList.pop()


