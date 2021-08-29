"""这个脚本从Annotations中随机划分训练集和测试集，最终生成ImagesSet/train.txt和val.txt"""
import os
import random
from os.path import *


# --------------------------全局地址变量--------------------------------#
dir_path = dirname(dirname(abspath(__file__)))
xml_path = os.path.join(dir_path, "VOC", "Annotations")
assert os.path.exists(xml_path), "xml_path not exist!"

ImageSets_path = os.path.join(dir_path, "VOC", "ImageSets")
if not os.path.exists(ImageSets_path):
    os.makedirs(ImageSets_path)

traintxt_path = os.path.join(dir_path, "VOC", "ImageSets", "train.txt")
valtxt_path = os.path.join(dir_path, "VOC", "ImageSets", "val.txt")

if os.path.exists(traintxt_path):
    os.remove(traintxt_path)
if os.path.exists(valtxt_path):
    os.remove(valtxt_path)
# --------------------------全局地址变量--------------------------------#



def create_imagesets(xml_full_path, traintxt_full_path, valtxt_full_path):
    train_percent = 0.8
    val_percent = 0.2
    xml_path = xml_full_path
    total_xml = os.listdir(xml_path)

    num = len(total_xml)
    lists = list(range(num))

    num_train = int(num * train_percent)

    train_list = random.sample(lists, num_train)
    for i in train_list:
        lists.remove(i)
    val_list = lists

    ftrain = open(traintxt_full_path, 'w')
    fval = open(valtxt_full_path, 'w')

    for i in range(num):
        name = total_xml[i][:-4] + '\n'
        if i in train_list:
            ftrain.write(name)
        else:
            fval.write(name)

    ftrain.close()
    fval.close()



if __name__ == '__main__':
    create_imagesets(xml_path, traintxt_path, valtxt_path)