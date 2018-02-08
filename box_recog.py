import cv2
import numpy as np
import time

img=cv2.imread('screen.png')
# cv2.namedWindow('image',cv2.WINDOW_NORMAL)

def box_recog(img):
    img_height, img_width, _ = img.shape
    edges = cv2.Canny(img, 80, 160)
    center = [0, 0]
    for y in range(int(img_height * 0.3), int(img_height * 0.6)):
        if sum(edges[y, :]) == 0:
            continue
        else:
            _,counts = np.unique(edges[y, :], return_counts=True)
            i = 1
            for x in range(img_width):
                if edges[y, x] != 0:
                    center[0]=int(x+counts[1]/2.0)
                    # center[0] = x
                    # x = x + 1
                    # while edges[y, x] != 0:
                    #     i = i + 1
                    #     center[0] = center[0] + x
                    #     x = x + 1
                    # center[0] = int(center[0] / i)
                    center[1] = y
                    res2 = cv2.line(img, (center[0], center[1]-100), (center[0], center[1]+100), (0, 255, 0), 2)
                    cv2.imshow('image', res2)  # 注意参数顺序
                    cv2.waitKey(100000)
            break

box_recog(img)