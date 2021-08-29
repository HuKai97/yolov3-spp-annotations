import collections
from PIL import Image
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont
import numpy as np

STANDARD_COLORS = [
    'AliceBlue', 'Chartreuse', 'Aqua', 'Aquamarine', 'Azure', 'Beige', 'Bisque',
    'BlanchedAlmond', 'BlueViolet', 'BurlyWood', 'CadetBlue', 'AntiqueWhite',
    'Chocolate', 'Coral', 'CornflowerBlue', 'Cornsilk', 'Crimson', 'Cyan',
    'DarkCyan', 'DarkGoldenRod', 'DarkGrey', 'DarkKhaki', 'DarkOrange',
    'DarkOrchid', 'DarkSalmon', 'DarkSeaGreen', 'DarkTurquoise', 'DarkViolet',
    'DeepPink', 'DeepSkyBlue', 'DodgerBlue', 'FireBrick', 'FloralWhite',
    'ForestGreen', 'Fuchsia', 'Gainsboro', 'GhostWhite', 'Gold', 'GoldenRod',
    'Salmon', 'Tan', 'HoneyDew', 'HotPink', 'IndianRed', 'Ivory', 'Khaki',
    'Lavender', 'LavenderBlush', 'LawnGreen', 'LemonChiffon', 'LightBlue',
    'LightCoral', 'LightCyan', 'LightGoldenRodYellow', 'LightGray', 'LightGrey',
    'LightGreen', 'LightPink', 'LightSalmon', 'LightSeaGreen', 'LightSkyBlue',
    'LightSlateGray', 'LightSlateGrey', 'LightSteelBlue', 'LightYellow', 'Lime',
    'LimeGreen', 'Linen', 'Magenta', 'MediumAquaMarine', 'MediumOrchid',
    'MediumPurple', 'MediumSeaGreen', 'MediumSlateBlue', 'MediumSpringGreen',
    'MediumTurquoise', 'MediumVioletRed', 'MintCream', 'MistyRose', 'Moccasin',
    'NavajoWhite', 'OldLace', 'Olive', 'OliveDrab', 'Orange', 'OrangeRed',
    'Orchid', 'PaleGoldenRod', 'PaleGreen', 'PaleTurquoise', 'PaleVioletRed',
    'PapayaWhip', 'PeachPuff', 'Peru', 'Pink', 'Plum', 'PowderBlue', 'Purple',
    'Red', 'RosyBrown', 'RoyalBlue', 'SaddleBrown', 'Green', 'SandyBrown',
    'SeaGreen', 'SeaShell', 'Sienna', 'Silver', 'SkyBlue', 'SlateBlue',
    'SlateGray', 'SlateGrey', 'Snow', 'SpringGreen', 'SteelBlue', 'GreenYellow',
    'Teal', 'Thistle', 'Tomato', 'Turquoise', 'Violet', 'Wheat', 'White',
    'WhiteSmoke', 'Yellow', 'YellowGreen'
]


def filter_low_thresh(boxes, scores, classes, category_index, thresh,
                      box_to_display_str_map, box_to_color_map):
    """
    1、过滤掉scores低于thresh的anchor;
    2、为每个anchor生成显示信息和框框颜色并分别保存在box_to_display_str_map和box_to_color_map中
    :param boxes: 最终预测结果 (anchor_nums, x1+y1+x2+y2)=(7, 4) (相对原图的预测结果) 分类别且按score从大到小排列
    :param scores: 所有预测anchors的得分 (7) 分类别且按score从大到小排列
    :param classes: 所有预测anchors的类别 (7) 分类别且按score从大到小排列
    :param category_index: 所有类别的信息（从data/pascal_voc_classes.json中读出）
    :param thresh: 设置阈值（默认0.1），过滤掉score太低的anchor
    :param box_to_display_str_map: 拿来存放每个anchor的显示信息（list） 每个anchor: tuple(box) = list[显示信息]
    :param box_to_color_map: 拿来存放每个anchor的框框颜色
    """
    for i in range(boxes.shape[0]):  # for anchors
        # 过滤掉score太低的anchor
        if scores[i] > thresh:
            box = tuple(boxes[i].tolist())  # numpy -> list -> tuple
            if classes[i] in category_index.keys():
                class_name = category_index[classes[i]]  # 得到每个anchor的class名
            else:
                class_name = 'N/A'
            display_str = str(class_name)
            display_str = '{}: {}%'.format(display_str, int(100 * scores[i]))  # 显示信息如 car: 90%
            # 将当前anchor的显示信息display_str加入到box_to_display_str_map中 每个anchor: tuple(box) = list[显示信息]
            box_to_display_str_map[box].append(display_str)
            # 为每个anchor对应的目标类别选择一个框框颜色 每个anchor: tuple(box) = list[颜色信息]
            box_to_color_map[box] = STANDARD_COLORS[classes[i] % len(STANDARD_COLORS)]
        else:
            break  # 网络输出概率已经排序过，当遇到一个不满足后面的肯定不满足


def draw_text(draw, box_to_display_str_map, box, left, right, top, bottom, color):
    """
    :param draw: 一个可以在给定图像(image)上绘图的对象
    :param box_to_display_str_map: 每个anchor的显示信息
    :param box: 当前anchor的预测信息 (xyxy)
    :param left: anchor的left
    :param right: anchor的right
    :param top: anchor的top
    :param bottom: anchor的bottom
    :param color: 当前anchor的信息颜色/anchor框框颜色
    :return:
    """
    try:
        # 从指定的文件('arial.ttf')中加载了一个字体对象，并且为指定大小(20)的字体创建了字体对象。
        font = ImageFont.truetype('arial.ttf', 20)
    except IOError:
        font = ImageFont.load_default()  # 加载一个默认的字体

    # 扫描ds(当前anchor的显示信息box_to_display_str_map[box])自动找到当前anchor显示信息的最大的字体大小(高)
    display_str_heights = [font.getsize(ds)[1] for ds in box_to_display_str_map[box]]
    # Each display_str has a top and bottom margin of 0.05x.
    total_display_str_height = (1 + 2 * 0.05) * sum(display_str_heights)

    # 如果添加到边界框顶部的显示字符串的总高度不超过图像顶部，就将字符串堆叠在边界框上方
    # text_bottom: 盛装显示字符的矩形框的top
    if top > total_display_str_height:
        text_bottom = top
    else:
        # 如果添加到边界框顶部的显示字符串的总高度超过图像顶部，就将字符串堆叠在边界框下方
        text_bottom = bottom + total_display_str_height
    # Reverse list and print from bottom to top.
    for display_str in box_to_display_str_map[box][::-1]:
        # 得到当前anchor的显示字符的最佳w和h
        text_width, text_height = font.getsize(display_str)
        # 得到当前anchor的显示字符的margin
        margin = np.ceil(0.05 * text_height)
        # 画盛装显示字符的矩形 传入左下角坐标+右上角坐标
        draw.rectangle([(left, text_bottom - text_height - 2 * margin),
                        (left + text_width, text_bottom)], fill=color)
        # 写入显示字符 传入显示字符的左上角坐标
        draw.text((left + margin, text_bottom - text_height - margin),
                  display_str,  # 显示字符
                  fill='red',   # 字体颜色
                  font=font)    # 加载字体
        text_bottom -= text_height - 2 * margin  # ？


def draw_box(image, boxes, classes, scores, category_index, thresh=0.1, line_thickness=3):
    """
    :param image: 原图 RGB (375, 500, 3) HWC  numpy格式(array)    img_o[:, :, ::-1]:BGR=>RGB
    :param boxes: 最终预测结果 (anchor_nums, x1+y1+x2+y2)=(7, 4) (相对原图的预测结果)
                  按score从大到小排列  numpy格式(array)
    :param classes: 所有预测anchors的类别 (7) 分类别且按score从大到小排列 numpy格式(array)
    :param scores: 所有预测anchors的得分 (7) 分类别且按score从大到小排列  numpy格式(array)
    :param category_index: 所有类别的信息（从data/pascal_voc_classes.json中读出）
    :param thresh: 设置阈值（默认0.1），过滤掉score太低的anchor
    :param line_thickness: 框框直线厚度
    :return:
    """
    box_to_display_str_map = collections.defaultdict(list)  # 拿来存放每个anchor的显示信息
    box_to_color_map = collections.defaultdict(str)  # 拿来存放每个anchor的框框颜色

    # 1、过滤掉scores低于thresh的anchor
    # 2、为每个anchor生成显示信息和框框颜色并分别保存在box_to_display_str_map和box_to_color_map中
    filter_low_thresh(boxes, scores, classes, category_index, thresh, box_to_display_str_map, box_to_color_map)

    # Draw all boxes onto image.
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)  # array(numpy) 转为Image格式
    draw = ImageDraw.Draw(image)  # 创建一个可以在给定图像(image)上绘图的对象
    for box, color in box_to_color_map.items():
        xmin, ymin, xmax, ymax = box
        (left, right, top, bottom) = (xmin * 1, xmax * 1, ymin * 1, ymax * 1)
        # 为每个anchor画框 顺序：左上->左下->右下->右上->左上
        draw.line([(left, top), (left, bottom), (right, bottom),
                   (right, top), (left, top)], width=line_thickness, fill=color)
        # 在每个框框上写上显示信息
        draw_text(draw, box_to_display_str_map,  box, left, right, top, bottom, color)
    return image
