import cv2
import numpy as np
from sklearn.cluster import KMeans
import glob

# 设置SIFT特征提取器
sift = cv2.SIFT_create()

# 读取样本图片并提取SIFT特征
def extract_sift_features(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    return descriptors

# 读取所有样本图片并提取SIFT特征
def extract_all_features(img_dir):
    features = []
    for i in range(200):
        img_path = img_dir + '/' + str(i+1) + '.png'
        img = cv2.imread(img_path)
        feature = extract_sift_features(img)
        features.append(feature)
    return features

# 对每个样本图片的SIFT特征进行K-means聚类
def kmeans_clustering(features):
    kmeans = KMeans(n_clusters=128, random_state=0).fit(features)
    return kmeans.cluster_centers_

# 选取训练样本并进行K-means聚类
train_samples = []
image_filenames = sorted(glob.glob("sw/*.png"))
for image_filename in image_filenames:
    img = cv2.imread(image_filename)
    feature = extract_sift_features(img)
    train_samples.append(feature)
image_filenames = sorted(glob.glob("zw/*.png"))
for image_filename in image_filenames:
    img = cv2.imread(image_filename)
    feature = extract_sift_features(img)
    train_samples.append(feature)
train_samples = np.concatenate(train_samples)
cluster_centers = kmeans_clustering(train_samples)

# 保存聚类后的数据
np.savetxt('train_100.txt', cluster_centers)
