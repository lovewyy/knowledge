import os

train_file = open('D:/Temp/Github/my_train/z/find_xml/ImageSets/Main/train.txt', 'w')
test_file = open('D:/Temp/Github/my_train/z/find_xml/ImageSets/Main/test.txt', 'w')
for _, _, train_files in os.walk('D:/Temp/Github/my_train/z/find_xml/train_images'):
    continue
for _, _, test_files in os.walk('D:/Temp/Github/my_train/z/find_xml/test_images'):
    continue
for file in train_files:
    train_file.write(file.split('.')[0] + '\n')

for file in test_files:
    test_file.write(file.split('.')[0] + '\n')
