import numpy as np
import cv2
from matplotlib import pyplot as plt

def matching(image, template, padding='zero', norm=True, print=False):
    ###roi####
    roi=image[:,600:1250]
    num_addrow = np.floor(template.shape[0] / 2).astype(int)
    num_addcol = np.floor(template.shape[1] / 2).astype(int)
    if padding == 'zero':
        output_img_shape = image.shape
        if num_addrow != 0:
            addrow = np.zeros((num_addrow, image.shape[1]))
            image = np.vstack((addrow, image))
            image = np.vstack((image, addrow))
        if num_addcol != 0:
            addcol = np.zeros((num_addcol, image.shape[0])).reshape((image.shape[0], num_addcol))
            image = np.hstack((addcol, image))
            image = np.hstack((image, addcol))
    output_img = np.zeros((output_img_shape[0], output_img_shape[1]))
    template_mean = np.mean(template)
    template = template - template_mean
    template_norm = np.sqrt(np.sum(np.square(template)))
    for yindex in range(output_img_shape[1]):
        for xindex in range(output_img_shape[0]):
            conv_sum = 0
            area = image[xindex:xindex + template.shape[0], yindex:yindex + template.shape[1]]
            area_mean = np.mean(area)
            area_norm = np.sqrt(np.sum(np.square(area)))

            if norm == False:
                conv_sum = np.sum(area * template)
            else:
                conv_sum = np.sum(area * template) / (area_norm * template_norm)
            output_img[xindex][yindex] = conv_sum
    if print == True:
        max = np.amax(output_img)
        min = np.amin(output_img)
        output_img = ((output_img - min) * 255.0 / (max - min)).astype(int)
    return output_img

def figure_position(image,template):
    corr_img = matching(image, template, norm=True, print=False)
    from numpy import unravel_index
    corr_img_thre = np.copy(corr_img)
    maxindex = unravel_index(corr_img_thre.argmax(), corr_img_thre.shape)
    center = [maxindex[0] + 5, maxindex[1] + 130]
    return center


# img=cv2.imread('/Users/roy/OneDrive/Courses/AI/WeChat Jump Bot/Code/screen.png',0)
# template=cv2.imread('/Users/roy/OneDrive/Courses/AI/WeChat Jump Bot/Code/figure_template.png',0)
# color=cv2.imread('/Users/roy/OneDrive/Courses/AI/WeChat Jump Bot/Code/screen.png')
# cv2.namedWindow('image',cv2.WINDOW_NORMAL)
##show the image##
# cv2.imshow('image',img) #注意参数顺序
# cv2.waitKey(100000)

# corr_img_show=matching(img,template,norm=True,print=True)
corr_img=matching(img,template,norm=True,print=False)

from numpy import unravel_index
corr_img_thre = np.copy(corr_img)
maxindex=unravel_index(corr_img_thre.argmax(), corr_img_thre.shape)
corr_img_thre[corr_img_thre < 0.26] = 0
print(maxindex)

center = [maxindex[0]+5, maxindex[1] + 130]
line = cv2.line(color, (maxindex[1],maxindex[0]-150), (maxindex[1],maxindex[0]+150), (0, 255, 20), 2)
line = cv2.line(line, (maxindex[1]-150,maxindex[0]+130), (maxindex[1]+150,maxindex[0]+130), (0, 255, 20), 2)



fig, axs = plt.subplots(2, 2, figsize=(10, 20))
axs[0, 0].imshow(img, cmap=plt.cm.gray)
axs[0, 0].title.set_text('Original Image')
axs[0, 1].imshow(corr_img,cmap=plt.cm.gray)
axs[0, 1].title.set_text('Corrlation Image')
axs[1, 0].imshow(corr_img_thre, cmap=plt.cm.gray)
axs[1, 0].title.set_text('Peak Point')
axs[1, 1].imshow(line)
axs[1, 1].title.set_text('Center Location')
plt.show()