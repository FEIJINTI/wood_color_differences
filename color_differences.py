# -*- codeing = utf-8 -*-
# Time : 2023/3/31 9:52
# @Auther : zhouchao
# @File: color_differences.py
# @Software:PyCharm
import cv2
import numpy as np
import time

t1 = time.time()
# 读取图片
img_path = 'data/rgb69.png'
img = cv2.imread(img_path)

# 将图像resize为(2000, 500)
img = cv2.resize(img, (1000, 250))
# 将图像等分成1000个小块
n_blocks = 1000
block_h, block_w = img.shape[:2]
block_h //= int(np.sqrt(n_blocks))
block_w //= int(np.sqrt(n_blocks))

# 用来存储每个小块的LAB数据
lab_data = []

# 遍历每个小块
for i in range(int(np.sqrt(n_blocks))):
    for j in range(int(np.sqrt(n_blocks))):
        # 计算当前小块的位置
        x = j * block_w
        y = i * block_h

        # 切割出当前小块的图像
        block = img[y:y+block_h, x:x+block_w]

        # 将小块转换为LAB颜色空间
        lab = cv2.cvtColor(block, cv2.COLOR_BGR2LAB)

        # 删去亮度低于50的像素点
        lab = lab[lab[:, :, 0] > 50]


        # 如果小块的像素点不足50，则去除这个小块
        if np.sum(lab[:,0] > 0) < 100:
            continue

        # 计算小块的LAB平均值，并存储到lab_data中
        lab_mean = np.mean(lab, axis=0)
        lab_data.append(lab_mean)

#计算方差
lab_data = np.array(lab_data)
lab_std = np.std(lab_data, axis=0)
print(lab_std)
print('time:', time.time() - t1)