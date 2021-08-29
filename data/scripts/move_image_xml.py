"""
这个脚本是将打完标签的图片和xml文件以voc的格式存储起来 运行结果最终生成Annotations和Images两个文件
"""
import os
import shutil
import json


# --------------------------全局地址变量--------------------------------#
xml_save_root = "F:\yolov3-my\data\ApplePest\Annotations"   # Annotations的地址
images_save_root = "F:\yolov3-my\data\ApplePest\Images"     # Images的地址

class_path = "F:\yolov3-my\data\\apple_pest_classes.json"   # classes.json文件的地址
data_path = "F:\dataset"                                    # 你打标签的源地址 它的子目录就是你的所有种类的文件
assert os.path.exists(class_path), "class_path not exist!"
assert os.path.exists(data_path), "data_path not exist!"

if not os.path.exists(xml_save_root):
    os.makedirs(xml_save_root)
if not os.path.exists(images_save_root):
    os.makedirs(images_save_root)
# --------------------------全局变量--------------------------------#



def move_image_xml(cla_path, xml_root, images_root, data_path):
    class_path = cla_path
    with open(class_path) as f:  # 读取label/json文件
        json_list = json.load(f)
    labels = list(json_list.keys())

    for i in range(len(labels)):
        open_root = os.path.join(data_path, labels[i])
        xml_save_root = xml_root
        images_save_root = images_root
        for file_full_name in os.listdir(open_root):
            file_name, file_type = os.path.splitext(file_full_name)[0], os.path.splitext(file_full_name)[1]
            if file_type == '.xml':
                open_path = os.path.join(open_root, file_full_name)
                save_path = os.path.join(xml_save_root, file_full_name)
                shutil.move(open_path, save_path)
            if file_type == '.jpg':
                open_path = os.path.join(open_root, file_full_name)
                save_path = os.path.join(images_save_root, file_full_name)
                shutil.move(open_path, save_path)

if __name__ == '__main__':
    move_image_xml(class_path, xml_save_root, images_save_root, data_path)