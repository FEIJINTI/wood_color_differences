import numpy as np
import skimage.feature
import skimage.segmentation
import skimage.color
import skimage.filters
import cv2
import matplotlib.pyplot as plt
import time

t1 = time.time()
img = cv2.imread(r"C:\Users\FEIJINTI\OneDrive\PycharmProjects\wood_color_differences\data\rgb7.png")
# 创建SIFT对象
sift = cv2.SIFT_create()

# 在图像中检测特征点并计算描述子
keypoints, descriptors = sift.detectAndCompute(img, None)

# 绘制特征点
img_with_keypoints = cv2.drawKeypoints(img, keypoints, None)

# 显示带有特征点的图像
cv2.imshow('Image with Keypoints', img_with_keypoints)
cv2.waitKey(0)
cv2.destroyAllWindows()