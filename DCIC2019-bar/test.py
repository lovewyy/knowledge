import time
import cv2

from glob import glob

import csv

import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

# # This is needed to display the images.
# %matplotlib inline

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")
# from utils import label_map_util
# from utils import visualization_utils as vis_util
from research.object_detection.utils import label_map_util
from research.object_detection.utils import visualization_utils as vis_util

start = time.time()
print("start: ", start)

confident = 0.5
confident1 = confident # 置信度，即scores>confident的目标才被输出
confident2 = confident
confident3 = confident
red = (255, 0, 0)

# What model to download.
MODEL_NAME = 'detection'
# MODEL_FILE = MODEL_NAME + '.tar.gz'
# DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = r'D:\Temp\model\models\research\object_detection\test_images\detection\frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = r'D:\Temp\model\models\research\object_detection\test_images\data\label_map.pbtxt'

NUM_CLASSES = 1

# download model
# opener = urllib.request.URLopener()
# 下载模型，如果已经下载好了下面这句代码可以注释掉
"""
opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
tar_file = tarfile.open(MODEL_FILE)
for file in tar_file.getmembers():
    file_name = os.path.basename(file.name)
    if 'frozen_inference_graph.pb' in file_name:
        tar_file.extract(file, os.getcwd())
"""
# Load a (frozen) Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')
# Loading label map
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)


# Helper code
def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


# For the sake of simplicity we will use only 2 images:
# image1.jpg
# image2.jpg
# If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
PATH_TO_TEST_IMAGES_DIR = r'D:\Temp\model\models\research\object_detection\test_images\test_images'
TEST_IMAGE_PATHS = glob(PATH_TO_TEST_IMAGES_DIR + '/*jpg')

out_csv = open(r'D:\Temp\model\models\research\object_detection\test_images\submit_example.csv','a', newline='')
csv_write = csv.writer(out_csv, dialect='excel')

# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)

with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
        # Definite input and output Tensors for detection_graph
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
        detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')

        list_count = []
        run_time = 0
        for image_path in TEST_IMAGE_PATHS:
            image = Image.open(image_path)
            width, height = image.size
            # the array based representation of the image will be used later in order to prepare the
            # result image with boxes and labels on it.
            image_np = load_image_into_numpy_array(image)
            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            image_np_expanded = np.expand_dims(image_np, axis=0)
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            # Each box represents a part of the image where a particular object was detected.
            boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            scores = detection_graph.get_tensor_by_name('detection_scores:0')
            classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
            # Actual detection.
            (boxes, scores, classes, num_detections) = sess.run(
                [boxes, scores, classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})

            # boxes = boxes[scores > confident1]
            # classes = classes[scores > confident2]
            # scores = scores[scores > confident3]

            # # Visualization of the results of a detection.
            # vis_util.visualize_boxes_and_labels_on_image_array(
            #     image_np,
            #     np.squeeze(boxes),
            #     np.squeeze(classes).astype(np.int32),
            #     np.squeeze(scores),
            #     category_index,
            #     use_normalized_coordinates=True,
            #     line_thickness=3)

            s_boxes = boxes[scores > confident1]
            s_classes = classes[scores > confident2]
            s_scores = scores[scores > confident3]

            run_time += 1
            print("run_time:", run_time)
            
            count = 0
            for i in range(len(s_classes)):
                name = image_path.split("\\")[-1]
                # name = image_path.split("\\")[-1].split('.')[0]   # 不带后缀
                ymin = s_boxes[i][0] * height  # ymin
                xmin = s_boxes[i][1] * width  # xmin
                ymax = s_boxes[i][2] * height  # ymax
                xmax = s_boxes[i][3] * width  # xmax
                score = s_scores[i]

                s_1 = image_path[71:]
                s_2 = str(int(xmin)) + ' '+ str(int(ymin)) + ' ' + str(int(xmax)) + ' ' + str(int(ymax))

                stu1 = [s_1, s_2]
                csv_write.writerow(stu1)


            #     if s_classes[i] in category_index.keys():
            #         class_name = category_index[s_classes[i]]['name']  # 得到英文class名称
            #
            #     print("name:", name)
            #     print("ymin:", ymin)
            #     print("xmin:", xmin)
            #     print("ymax:", ymax)
            #     print("xmax:", xmax)
            #     print("score:", score)
            #     print("class:", class_name)
            #     count += 1
            #     print(count)
            #
            #     cv2.rectangle(image_np, (int(xmin), int(ymin)), (int(xmax), int(ymax)), red, 3)
            #
            #     print("################")
            #
            # list_count.append(count)
            #
            # plt.figure(figsize=IMAGE_SIZE)
            # plt.imshow(image_np)
            # plt.show()

end = time.time()
print("Execution Time: ", end - start)
# print("list_count", list_count)
#
# count_i = 0
# print("max: ", max(list_count))
# for i in list_count:
#     count_i += i
# print("bar: ", count_i)