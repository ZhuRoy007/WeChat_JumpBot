import cv2
import numpy as np

img=cv2.imread('1.jpeg')
# cv2.namedWindow('image',cv2.WINDOW_NORMAL)
# cv2.namedWindow('image2',cv2.WINDOW_NORMAL)
cv2.namedWindow('image3',cv2.WINDOW_NORMAL)
# cv2.imshow('image',img) #注意参数顺序
# cv2.waitKey(100000)

hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
# lower_HSV = np.array([118,70,40])
# upper_HSV = np.array([140,255,255])
lower_HSV = np.array([118, 56, 54])
upper_HSV = np.array([140, 117, 99])
mask = cv2.inRange(hsv, lower_HSV, upper_HSV)

# # 对原图和掩模进行位运算
res = cv2.bitwise_and(img, img, mask=mask)
#
#
# cv2.imshow('image',hsv) #注意参数顺序
# cv2.waitKey(100000)
#
# res=cv2.inRange(hsv,lower_blue,upper_blue)
# cv2.imshow('image2',mask) #注意参数顺序
# cv2.waitKey(100000)

x,y,w,h = cv2.boundingRect(mask)
offset=11
res2=cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
res2=cv2.line(img,(x-5,y+h-offset),(x+w+5,y+h-offset),(0,255,0),2)


center=(x+w,y+h-offset)

#
cv2.imshow('image3',res2) #注意参数顺序
cv2.waitKey(100000)
