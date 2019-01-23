# coding:utf-8

# Copyright@hitzym, Dec,09,2017 at HIT # blog:http://blog.csdn.net/yinhuan1649/article/category/7330626

import cv2
import xml.etree.ElementTree as ET
import os
from matplotlib import pyplot as plt
from PIL import Image

IMAGE_SIZE = (12, 8)

imgpath = 'home/testimgs/'          #旋转后的图像路径
xmlpath = 'home/testxml/'           #旋转后的xml文件路径
for img in os.listdir(imgpath):
    a, b = os.path.splitext(img)
    img = cv2.imread(imgpath + a +'.jpg')
    tree = ET.parse(xmlpath + a + '.xml')
    root = tree.getroot()
    for box in root.iter('bndbox'):
        x1 = int(box.find('xmin').text)
        y1 = int(box.find('ymin').text)
        x2 = int(box.find('xmax').text)
        y2 = int(box.find('ymax').text)
        cv2.rectangle(img,(x1,y1),(x2, y2), [0,255,0], 3)

    # cv2.imshow("test", img)

    plt.figure(figsize=IMAGE_SIZE)
    plt.imshow(img)
    plt.show()

    # cv2.waitKey(1000)
    if 1 == cv2.waitKey(0):
        pass