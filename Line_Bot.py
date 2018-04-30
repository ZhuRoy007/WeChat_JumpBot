import cv2
import numpy as np
import random
import wda
import time
import json
import template_matching

wda.DEBUG = False  # default False
wda.HTTP_TIMEOUT = 600.0  # default 60.0 seconds

c = wda.Client('http://localhost:8100')
s = c.session()
# Show status
print(c.status())


def load_screenshot():
    c.screenshot('screen.png')
    img = cv2.imread('screen.png')
    return img


def body_recog(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # lower_HSV = np.array([118, 70, 40])
    # upper_HSV = np.array([140, 255, 255])
    lower_HSV = np.array([118, 56, 54])
    upper_HSV = np.array([140, 117, 99])
    mask = cv2.inRange(hsv, lower_HSV, upper_HSV)
    kernel = np.ones((5, 5), np.uint8)
    opening_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    # 对原图和掩模进行位运算
    # res = cv2.bitwise_and(img, img, mask=opening_mask)

    x, y, w, h = cv2.boundingRect(opening_mask)
    body_boundingRect = (x, y, w, h)
    assert 1000 < w * h <= 100000, "Body recognition Failed, Jump by yourself and rerun the code"
    offset = 11
    center = [int(x + w / 2), y + h - offset]
    # res2 = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # res2 = cv2.line(res2, (x - 5, y + h - offset), (x + w + 5, y + h - offset), (0, 255, 0), 2)
    # cv2.imshow('image', res2)  # 注意参数顺序
    # cv2.waitKey(100000)
    return center, body_boundingRect


def box_recog(img, body_boundingRect):
    x, y, w, h = body_boundingRect
    img_height, img_width, _ = img.shape
    edge = cv2.Canny(img, 80, 160)
    edgemask = np.ma.make_mask(edge, copy=True, shrink=True).astype(np.uint8)
    edgemask[y-28:y + h+10, x-20:x + w+10] = False
    masked_edge = cv2.bitwise_and(edge, edge, mask=edgemask)
    cv2.imshow('image', masked_edge)  # 注意参数顺序
    cv2.waitKey(100)
    center = [0, 0]
    for y in range(int(img_height * 0.3), int(img_height * 0.6)):
        if sum(masked_edge[y, :]) == 0:
            continue
        else:
            _, counts = np.unique(masked_edge[y, :], return_counts=True)
            for x in range(img_width):
                if masked_edge[y, x] != 0:
                    center[0] = int(x + counts[1] / 2.0)
                    center[1] = y
                    return center
            break


def jump(distance, timefactor):
    press_time = distance * timefactor / 1000.0
    print('Calculated press time:' + str(press_time) + '\n')
    press_position = [random.randint(200, 300), random.randint(200, 300)]
    s.tap_hold(press_position[0], press_position[1], press_time)


def draw_positon(img, body_center, box_center):
    draw_body = cv2.line(img, (body_center[0], body_center[1] - 100), (body_center[0], body_center[1] + 100),
                         (0, 255, 0), 2)
    draw_box = cv2.line(draw_body, (box_center[0], box_center[1] - 100), (box_center[0], box_center[1] + 100),
                        (0, 255, 0), 2)
    # cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.imshow('image', draw_box)  # 注意参数顺序
    cv2.waitKey(1)
    cv2.destroyAllWindows()

def load_config_json(filename='config.json'):
    return json.load(open(filename))


def main():
    configs = load_config_json()
    template = cv2.imread('/Users/roy/OneDrive/Courses/AI/WeChat Jump Bot/Code/figure_template.png', 0)
    while True:
        img = load_screenshot()
        # body_center, body_boundingRect = body_recog(img)
        _, body_boundingRect = body_recog(img)
        body_center=template_matching.figure_position(img,template)
        box_center = box_recog(img, body_boundingRect)
        # draw_positon(img,body_center,box_center)
        jump(max(abs(box_center[0] - body_center[0]), 80), configs['coefficient'])
        time.sleep(random.uniform(1.1, 1.3))


if __name__ == '__main__':
    main()
