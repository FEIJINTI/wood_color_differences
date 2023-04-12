# -*- codeing = utf-8 -*-
# Time : 2023/4/11 16:16
# @Auther : zhouchao
# @File: test1.py
# @Software:PyCharm
import cv2
import numpy as np

# 读取图片并转换为灰度图像
img = cv2.imread(r'C:\Users\FEIJINTI\OneDrive\PycharmProjects\wood_color_differences\data\rgb7.png')

# 将图像转换为灰度图像
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 使用自适应阈值二值化图像
ret, thresh = cv2.threshold(gray, 190, 255, cv2.THRESH_BINARY)


# 创建掩模
mask = cv2.merge((thresh, thresh, thresh))
# 应用掩模
masked_img = cv2.bitwise_and(img, mask)
# 显示结果
cv2.imshow('Result', masked_img)
cv2.waitKey(0)

# 保存输出图像
cv2.imwrite('result.jpg', masked_img)
