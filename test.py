import cv2
import time
import numpy as np

start = time.time()
# 读取彩色图像
img = cv2.imread(r'C:\Users\FEIJINTI\OneDrive\PycharmProjects\wood_color_differences\data\rgb7.png')
# 将图像转换为灰度图像
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 阈值处理，将像素值低于50的像素点置为0，其余像素点不变
_, thresh = cv2.threshold(img_gray, 50, 255, cv2.THRESH_BINARY)

# 查找图像中的所有轮廓
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# 对于每个轮廓，计算包围它的矩形
areas = []
rects = []
for contour in contours:
    rect = cv2.boundingRect(contour)
    area = rect[2] * rect[3]
    areas.append(area)
    rects.append(rect)

# 找到面积最大的矩形
max_idx = np.argmax(areas)
max_rect = rects[max_idx]

# 将原图像中的该矩形区域裁剪出来
x, y, w, h = max_rect
cropped = img[y:y+h, x:x+w]
end = time.time()

# 显示裁剪后的图像
print('time: ', end - start)
cv2.imshow('Cropped Image', cropped)
cv2.waitKey(0)
cv2.destroyAllWindows()






