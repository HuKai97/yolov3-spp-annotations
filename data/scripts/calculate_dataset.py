"""二
1.统计训练集和验证集的数据并生成相应train_path.txt和val_path.txt文件
2.创建data.data文件，记录classes个数, train以及val数据集文件(.txt)路径和label.names文件路径
"""
import os
from os.path import *

dir_path = dirname(dirname(abspath(__file__)))
train_annotation_dir = os.path.join(dir_path, "dataset", "train", "labels")
val_annotation_dir = os.path.join(dir_path, "dataset", "val", "labels")
classes_label = os.path.join(dir_path, "dataset_classes.names")

assert os.path.exists(train_annotation_dir), "train_annotation_dir not exist!"
assert os.path.exists(val_annotation_dir), "val_annotation_dir not exist!"
assert os.path.exists(classes_label), "classes_label not exist!"

# 保存地址
train_path_txt = os.path.join(dir_path, "train_path.txt")
val_path_txt = os.path.join(dir_path, "val_path.txt")
dataset_data = os.path.join(dir_path, "dataset.data")


def calculate_data_txt(txt_path, dataset_dir):
    # create my_data.txt file that record image list
    with open(txt_path, "w") as w:
        for file_name in os.listdir(dataset_dir):
            if file_name == "classes.txt":
                continue

            img_path = os.path.join(dataset_dir.replace("labels", "images"),
                                    file_name.split(".")[0]) + ".jpg"
            line = img_path + "\n"
            assert os.path.exists(img_path), "file:{} not exist!".format(img_path)
            w.write(line)


def create_dataset_data(create_data_path, label_path, train_path, val_path, classes_info):
    # create my_data.data file that record classes, train, valid and names info.
    # shutil.copyfile(label_path, "./data/my_data_label.names")
    with open(create_data_path, "w") as w:
        w.write("classes={}".format(len(classes_info)) + "\n")  # 记录类别个数
        w.write("train={}".format(train_path) + "\n")           # 记录训练集对应txt文件路径
        w.write("valid={}".format(val_path) + "\n")             # 记录验证集对应txt文件路径
        w.write("names={}".format(classes_label) + "\n")        # 记录label.names文件路径

def main():
    # 统计训练集和验证集的数据并生成相应txt文件
    calculate_data_txt(train_path_txt, train_annotation_dir)
    calculate_data_txt(val_path_txt, val_annotation_dir)

    classes_info = [line.strip() for line in open(classes_label, "r").readlines() if len(line.strip()) > 0]
    # dataset.data文件，记录classes个数, train以及val数据集文件(.txt)路径和label.names文件路径
    create_dataset_data(dataset_data, classes_label, train_path_txt, val_path_txt, classes_info)


if __name__ == '__main__':
    main()
