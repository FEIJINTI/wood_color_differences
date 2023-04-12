import cv2
import numpy as np
import time

# 读取图片
start_time = time.time()
img_path = r'C:\Users\FEIJINTI\OneDrive\PycharmProjects\wood_color_differences\data\rgb69.png'
img = cv2.imread(img_path)
t2 = time.time()
# 将图片转换成lab格式
# img = cv2.resize(img, (400, 100))
lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
t3 = time.time()

# 筛选亮度值低于50的像素，作为背景


# 提取地板区域的像素
floor_mask = lab[:, :, 0] > 50
floor_pixels = lab[floor_mask]

# 计算地板颜色的均值和方差
floor_var = np.var(floor_pixels, axis=0)
print(floor_var)
# 判断地板颜色是否整体一致
if np.all(floor_var < 300):
    print("地板颜色整体一致")
else:
    print("地板颜色存在差异")

end_time = time.time()

print("算法用时：{:.2f}秒".format(end_time - start_time))
print("读取图片用时：{:.2f}秒".format(t2 - start_time))
print("转换图片用时：{:.2f}秒".format(t3 - t2))
