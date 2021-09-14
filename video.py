import os
import json
import time
import torch
import cv2
import argparse
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt


from modules.model import Darknet
from utils import datasets
from utils.draw_box_utils import draw_box
from utils.post_processing_utils import non_max_suppression, scale_coords
from utils.utils import time_synchronized, check_file


def main(opt):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using {} device training.".format(device.type))


    # 1、载入opt参数
    cfg = opt.cfg               # yolo网络配置文件path
    weights = opt.weights       # 训练权重path
    json_path = opt.json_path   # voc classes json path
    video_path = opt.video_path     # 预测图片地址
    img_size = opt.img_size     # 预测图像大小（letterbox后）

    # path=""调用摄像头  不为空调用视频
    if video_path:
        capture = cv2.VideoCapture(video_path)
    else:
        capture = cv2.VideoCapture(0)


    # 2、载入json文件 得到所有class
    json_file = open(json_path, 'r')
    class_dict = json.load(json_file)
    category_index = {v: k for k, v in class_dict.items()}

    # 将检测好的图片生成视频  不知道这里为什么没成功？？？？生成的视频没用内容打不开？？？
    # fps = 30
    # size = (1920, 1280)
    # videowriter = cv2.VideoWriter("Video.avi", cv2.VideoWriter_fourcc(*'XVID'), fps, size)


    # 3、初始化模型 模型载入权重
    model = Darknet(cfg)
    model.load_state_dict(torch.load(weights, map_location=device)["model"], strict=False)
    model.to(device)
    index = 0  # 记录当前frame的index
    while(True):
        ref, img_o = capture.read()
        # eval测试模式
        model.eval()
        with torch.no_grad():
            # letterbox  numpy格式(array)   img:(384, 512, 3) H W C
            # 将原图最长边缩放到指定大小，再将原图较短边按原图比例缩放,最后将较短边两边pad操作缩放到最长边大小（不会失真）
            img = datasets.letterbox(img_o, new_shape=img_size, auto=True, color=(0, 0, 0))[0]

            # Convert (384, 512, 3) => (384, 512, 3) => (3, 384, 512)
            # img[:, :, ::-1]  BGR to RGB => transpose(2, 0, 1) HWC(384, 512, 3)  to  CHW(3, 384, 512)
            img = img[:, :, ::-1].transpose(2, 0, 1)
            img = np.ascontiguousarray(img)  # 使内存是连续的

            # numpy(3, 384, 512) CHW => torch.tensor [3, 384, 512] CHW
            img = torch.from_numpy(img).to(device).float()
            img /= 255.0  # 归一化scale (0, 255) to (0, 1)
            # [3, 384, 512] CHW => [1, 3, 384, 512] BCHW
            img = img.unsqueeze(0)  # add batch dimension

            # start inference
            t1 = time_synchronized()  # 获取当前时间 其实可以用time.time()
            # 推理阶段实际上会有两个返回值 x(相对原图的), p
            # x: predictor数据处理后的输出(数值是相对原图的，这里是img)
            #    [batch_size, anchor_num * grid * grid, xywh + obj + classes]
            #    这里pred[1,12096,25] (实际上是等于x)表示这张图片总共生成了12096个anchor(一个grid中三个anchor)
            # p: predictor原始输出即数据是相对feature map的
            #    [batch_size, anchor_num, grid, grid, xywh + obj + classes]
            pred = model(img)[0]  # only get inference result
            t2 = time_synchronized()
            print("model inference time:", t2 - t1)
            # nms pred=[7,6]=[obj_num, xyxy+score+cls] 这里的xyxy是相对img的
            # pred: 按score从大到小排列; output[0]=第一张图片的预测结果 不一定一次只传入一张图片的
            pred = non_max_suppression(pred, conf_thres=0.1, iou_thres=0.6, multi_cls=True)[0]
            t3 = time.time()
            print("nms time:", t3 - t2)

            if pred is None:
                print("No target detected.")
                exit(0)

            # 将nms后的预测结果pred tensor格式（是相对img上的）img.shape=[B,C,H,W]
            # 映射到原图img_o上 img_o.shape=[H, W, C]  pred=(anchor_nums, xyxy+score+class)
            pred[:, :4] = scale_coords(img.shape[2:], pred[:, :4], img_o.shape).round()
            print("pred shape:", pred.shape)

            # tensor.detach()截断tensor变量反向传播的梯度流，因为是预测所以不需要计算梯度信息
            # bboxes、scores、classes: 按score从大到小排列  tensor=>numpy
            bboxes = pred[:, :4].detach().cpu().numpy()  # xyxys
            scores = pred[:, 4].detach().cpu().numpy()   # scores
            classes = pred[:, 5].detach().cpu().numpy().astype(int) + 1  # classes

            # 到这一步，我们就得到了最终的相对原图的所有预测信息bboxes（位置信息）(7,4); scores(7); classes（类别）(7)

            # 画出每个预测结果
            img_o = draw_box(img_o[:, :, ::-1], bboxes, classes, scores, category_index)

            # 保存预测后的图片
            img_o.save("imgs/predict_result/{}.jpg".format(index))
            index += 1

            # opencv显示图片
            img_o = cv2.cvtColor(np.array(img_o), cv2.COLOR_RGB2BGR)
            cv2.imshow('video', img_o)
            # 将检测好的图片生成视频
            # videowriter.write(img_o)
            c = cv2.waitKey(30) & 0xff
            if c==27:
                capture.release()
                break

            # matploblib显示预测图片
            # plt.imshow(img_o)
            # plt.show()

           
    # videowriter.release()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='cfgs/yolov3-spp.cfg', help="cfg/*.cfg path")
    parser.add_argument('--weights', type=str, default='weights/yolov3spp-pest.pt',
                        help='pretrain weights path')
    parser.add_argument('--json-path', type=str, default='data/pest_classes.json',
                        help="voc_classes_json_path")
    parser.add_argument('--video-path', type=str, default='./imgs/003.avi',
                        help="path=""调用摄像头  不为空调用视频")
    parser.add_argument('--img-size', type=int, default=512,
                        help="predict img path [416, 512, 608] 32的倍数")

    opt = parser.parse_args()

    # 检查文件是否存在
    opt.cfg = check_file(opt.cfg)
    opt.data = check_file(opt.weights)
    opt.hyp = check_file(opt.json_path)
    print(opt)

    main(opt)
