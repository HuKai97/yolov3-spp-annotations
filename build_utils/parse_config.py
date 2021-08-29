import os
import numpy as np


def parse_model_cfg(path: str):
    """
    读取并解析模型的配置文件（cfg）
    :param path: 配置文件地址  如：cfg/yolov3-spp.cfg
    :return: 返回一个list，list中由很多的dict字典组成，每个字典对应一层的信息（第0层是net）
    """
    # 1、检查配置文件是否存在/格式是否正确
    if not path.endswith(".cfg") or not os.path.exists(path):
        raise FileNotFoundError("the cfg file not exist...")

    # 2、读取配置文件所有信息 用换行符进行分割
    with open(path, "r") as f:
        lines = f.read().split("\n")

    # 3、去除空行和注释行
    lines = [x for x in lines if x and not x.startswith("#")]

    # 4、去除每行开头和结尾的空格符
    lines = [x.strip() for x in lines]

    # 5、开始解析每一行
    mdefs = []  # module definitions
    for line in lines:
        # 筛选后，所有的line都只剩两种情况: 1、[tpye名]  2、key=val
        if line.startswith("["):  # this marks the start of a new block
            mdefs.append({})
            mdefs[-1]["type"] = line[1:-1].strip()  # 记录module类型 type=层结构名称
            # 如果是卷积模块，设置默认不使用BN
            if mdefs[-1]["type"] == "convolutional":
                mdefs[-1]["batch_normalize"] = 0
        else:
            key, val = line.split("=")
            key = key.strip()  # 去除前后的空格
            val = val.strip()

            if key == "anchors":
                # anchors = 10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326
                val = val.replace(" ", "")  # 将空格去除
                # 将shape    (1, 18)=>(9, 2)  九个anchors
                mdefs[-1][key] = np.array([float(x) for x in val.split(",")]).reshape((-1, 2))  # np anchors
            elif (key in ["from", "layers", "mask"]) or (key == "size" and "," in val):
                # 如果是from/layers/mask, 就将值以数组的形式存储（一般是多个值）
                mdefs[-1][key] = [int(x) for x in val.split(",")]
            else:
                # 其他情况就是一般的一个key对应一个val存储起来
                # TODO: .isnumeric() actually fails to get the float case
                if val.isnumeric():  # return int or float 如果是数值的情况
                    mdefs[-1][key] = int(val) if (int(val) - float(val)) == 0 else float(val)
                else:
                    mdefs[-1][key] = val  # return string  是字符的情况

    # 6、检测所有读出的配置信息k是否符合标准  check all fields are supported
    supported = ['type', 'batch_normalize', 'filters', 'size', 'stride', 'pad', 'activation', 'layers', 'groups',
                 'from', 'mask', 'anchors', 'classes', 'num', 'jitter', 'ignore_thresh', 'truth_thresh', 'random',
                 'stride_x', 'stride_y', 'weights_type', 'weights_normalization', 'scale_x_y', 'beta_nms', 'nms_kind',
                 'iou_loss', 'iou_normalizer', 'cls_normalizer', 'iou_thresh', 'probability']
    # 遍历检查每个模型的配置
    for x in mdefs[1:]:  # 0对应net配置
        # 遍历每个配置字典中的key值
        for k in x:
            if k not in supported:
                raise ValueError("Unsupported fields:{} in cfg".format(k))

    return mdefs


def parse_data_cfg(path):
    # Parses the data configuration file
    if not os.path.exists(path) and os.path.exists('data' + os.sep + path):  # add data/ prefix if omitted
        path = 'data' + os.sep + path

    with open(path, 'r') as f:
        lines = f.readlines()

    options = dict()
    for line in lines:
        line = line.strip()
        if line == '' or line.startswith('#'):
            continue
        key, val = line.split('=')
        options[key.strip()] = val.strip()

    return options


