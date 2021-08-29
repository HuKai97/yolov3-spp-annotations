import torch
import argparse
import yaml
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from torch import optim

from build_utils.datasets import LoadImagesAndLabels
from build_utils.parse_config import parse_data_cfg
from modules.model import DarkNet, YOLOLayer
from train_val_utils.coco_utils import get_coco_api_from_dataset
from train_val_utils.train_eval_body import train_one_epoch, evaluate
from train_val_utils.other_utils import check_file, init_seeds
import os
import math
import glob


def train(hyp):
    init_seeds()  # 初始化随机种子，保证结果可复现

    device = torch.device(opt.device if torch.cuda.is_available() else "cpu")
    print("Using {} device training.".format(device.type))

    # ---------------------------------参数设置----------------------------------
    # best weights
    wdir = "weights" + os.sep  # weights dir = 'weights\\'
    best = wdir + "best.pt"  # 'weights\\best.pt'

    # opt参数
    cfg = opt.cfg
    data = opt.data
    epochs = opt.epochs
    batch_size = opt.batch_size
    accumulate = max(round(64 / batch_size), 1)  # accumulate 每训练accumulate张图片更新权重
    weights = opt.weights  # initial training weights
    results_file = opt.result_name
    imgsz_train = opt.img_size  # train image sizes
    imgsz_test = opt.img_size  # test image sizes
    multi_scale = opt.multi_scale  # 多尺度训练(默认): True
    warmup = opt.warmup
    augment = opt.augment
    rect = opt.rect
    mosaic = opt.mosaic

    # 路径参数 data/dataset.data
    # configure run
    data_dict = parse_data_cfg(data)
    train_path = data_dict["train"]
    test_path = data_dict["valid"]
    nc = int(data_dict["classes"])  # number of classes
    hyp["cls"] *= nc / 80  # update coco-tuned hyp['cls'] to current dataset
    hyp["obj"] *= imgsz_test / 320

    # ---------------------------------多尺度训练----------------------------------
    # 图像要设置成32的倍数
    gs = 32  # (pixels) grid size
    assert math.fmod(imgsz_test, gs) == 0, "--img-size %g must be a %g-multiple" % (imgsz_test, gs)
    grid_min, grid_max = imgsz_test // gs, imgsz_test // gs
    # 是否多尺度训练 默认是
    if multi_scale:
        imgsz_min = opt.img_size // 1.5
        imgsz_max = opt.img_size // 0.667

        # 将给定的最大，最小输入尺寸向下调整到32的整数倍
        grid_min, grid_max = imgsz_min // gs, imgsz_max // gs
        imgsz_min, imgsz_max = int(grid_min * gs), int(grid_max * gs)
        imgsz_train = imgsz_max  # initialize with max size
        print("Using multi_scale training, image range[{}, {}]".format(imgsz_min, imgsz_max))  # [320, 736]

    # ---------------------------------其他----------------------------------


    # 移除之前的resutl.txt
    for f in glob.glob(results_file):
        os.remove(f)

        # =================================== step 1/5 数据处理==========================================
        # 训练集的图像尺寸指定为multi_scale_range中最大的尺寸（736）  数据增强
    train_dataset = LoadImagesAndLabels(train_path, imgsz_train, batch_size,
                                        augment=augment,
                                        hyp=hyp,  # augmentation hyperparameters
                                        rect=rect,  # rectangular training
                                        mosaic=mosaic)
    # 验证集的图像尺寸指定为img_size(512)
    val_dataset = LoadImagesAndLabels(test_path, imgsz_test, batch_size,
                                      hyp=hyp,
                                      rect=False)  # 将每个batch的图像调整到合适大小，可减少运算量(并不是512x512标准尺寸)

    # dataloader
    # nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    # 这里调用LoadImagesAndLabels.__len__ 将train_dataset按batch_size分成batch份
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=batch_size,
                                                   num_workers=0,  # win一般设为0
                                                   shuffle=not rect,
                                                   # Shuffle=True unless rectangular training is used
                                                   pin_memory=True,
                                                   collate_fn=train_dataset.collate_fn)

    val_datasetloader = torch.utils.data.DataLoader(val_dataset,
                                                    batch_size=batch_size,
                                                    num_workers=0,
                                                    pin_memory=True,
                                                    collate_fn=val_dataset.collate_fn)

    # =================================== step 2/5 模型载入==========================================
    # Initialize model
    model = DarkNet(cfg).to(device)

    # ---------------------------------训练参数设置（是否冻结训练）----------------------------------
    # 是否冻结权重，freeze_layers=true 只训练3个predictor的权重
    if opt.freeze_layers:
        # 索引减一对应的是predictor的索引，YOLOLayer并不是predictor
        output_layer_indices = [idx - 1 for idx, module in enumerate(model.module_list) if
                                isinstance(module, YOLOLayer)]
        # 冻结除predictor和YOLOLayer外的所有层
        freeze_layer_indeces = [x for x in range(len(model.module_list)) if
                                (x not in output_layer_indices) and
                                (x - 1 not in output_layer_indices)]
        # Freeze non-output layers
        # 总共训练3x2=6个parameters
        for idx in freeze_layer_indeces:
            for parameter in model.module_list[idx].parameters():
                parameter.requires_grad_(False)
    else:
        # 默认是False
        # 如果freeze_layer为False，默认仅训练除darknet53之后的部分
        # 若要训练全部权重，删除以下代码
        darknet_end_layer = 74  # only yolov3spp cfg
        # Freeze darknet53 layers
        # 总共训练21x3+3x2=69个parameters
        for idx in range(darknet_end_layer + 1):  # [0, 74]
            for parameter in model.module_list[idx].parameters():
                parameter.requires_grad_(False)

    # =================================== step 3.1/5 优化器定义==========================================
    # optimizer
    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(pg, lr=hyp["lr0"], momentum=hyp["momentum"],
                          weight_decay=hyp["weight_decay"], nesterov=True)

    # ---------------------------------载入pt----------------------------------
    start_epoch = 0
    best_map = 0.0
    if weights.endswith(".pt") or weights.endswith(".pth"):
        ckpt = torch.load(weights, map_location=device)

        # load model
        try:
            ckpt["model"] = {k: v for k, v in ckpt["model"].items() if model.state_dict()[k].numel() == v.numel()}
            model.load_state_dict(ckpt["model"], strict=False)
        except KeyError as e:
            s = "%s is not compatible with %s. Specify --weights '' or specify a --cfg compatible with %s. " \
                "See https://github.com/ultralytics/yolov3/issues/657" % (opt.weights, opt.cfg, opt.weights)
            raise KeyError(s) from e

        # load optimizer
        if ckpt["optimizer"] is not None:
            optimizer.load_state_dict(ckpt["optimizer"])
            if "best_map" in ckpt.keys():
                best_map = ckpt["best_map"]

        # load results
        if ckpt.get("training_results") is not None:
            with open(results_file, "w") as file:
                file.write(ckpt["training_results"])  # write results.txt

        # epochs
        start_epoch = ckpt["epoch"] + 1
        if epochs < start_epoch:
            print('%s has been trained for %g epochs. Fine-tuning for %g additional epochs.' %
                  (opt.weights, ckpt['epoch'], epochs))
            epochs += ckpt['epoch']  # finetune additional epochs

        del ckpt

    # =================================== step 3.2/5 优化器学习率设置======================================
    lf = lambda x: ((1 + math.cos(x * math.pi / epochs)) / 2) * (1 - hyp["lrf"]) + hyp["lrf"]  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    scheduler.last_epoch = start_epoch  # 指定从哪个epoch开始

    # # Plot lr schedule  LR.png 学习率变化曲线   L
    # y = []
    # for _ in range(epochs):
    #     scheduler.step()
    #     y.append(optimizer.param_groups[0]['lr'])
    # plt.plot(y, '.-', label='LambdaLR')
    # plt.xlabel('epoch')
    # plt.ylabel('LR')
    # plt.tight_layout()
    # plt.savefig('others/LR.png', dpi=300)
    # model.yolo_layers = model.module.yolo_layers

    # --------------------------------- step 4 损失函数参数 ----------------------------------
    model.nc = nc  # attach number of classes to model
    model.hyp = hyp  # attach hyperparameters to model
    model.gr = 1.0  # giou loss ratio (obj_loss = 1.0 or giou)
    # 计算每个类别的目标个数，并计算每个类别的比重
    # model.class_weights = labels_to_class_weights(train_dataset.labels, nc).to(device)  # attach class weights

    # =================================== step 5/5 训练======================================
    # start training
    # caching val_data when you have plenty of memory(RAM)
    # coco = None
    coco = get_coco_api_from_dataset(val_dataset)  # 方便后面计算MAP用

    print("starting traning for %g epochs..." % epochs)
    for epoch in range(start_epoch, epochs):
        # 训练集
        mloss, lr = train_one_epoch(model, optimizer, train_dataloader,
                                    device, epoch + 1, epochs,
                                    accumulate=accumulate,  # 迭代多少batch才训练完64张图片
                                    img_size=imgsz_train,  # 输入图像的大小
                                    multi_scale=multi_scale,
                                    grid_min=grid_min,  # grid的最小尺寸
                                    grid_max=grid_max,  # grid的最大尺寸
                                    gs=gs,  # grid step: 32
                                    print_freq=1,  # 每训练多少个step打印一次信息
                                    warmup=warmup)  # 第一个epoch要采用特殊的训练方式 慢慢训练
        # 更新学习率
        scheduler.step()

        # 验证集  只测试最后一个epoch  epochs=1,2,3...
        if opt.notest is False or epoch == epochs:
            # evaluate on the test dataset
            result_info = evaluate(model, val_datasetloader,
                                   coco=coco, device=device)



            # --------------------------------- 打印输出 保存模型----------------------------------
            coco_mAP = result_info[0]
            voc_mAP = result_info[1]
            coco_mAR = result_info[8]

            # write into tensorboard
            if tb_writer:
                tags = ['train/giou_loss', 'train/obj_loss', 'train/cls_loss', 'train/loss', "learning_rate",
                        "mAP@[IoU=0.50:0.95]", "mAP@[IoU=0.5]", "mAR@[IoU=0.50:0.95]"]

                for x, tag in zip(mloss.tolist() + [lr, coco_mAP, voc_mAP, coco_mAR], tags):
                    tb_writer.add_scalar(tag, x, epoch)

            # write into txt
            with open(results_file, "a") as f:
                result_info = [str(round(i, 4)) for i in result_info]
                txt = "epoch:{} {}".format(epoch, '  '.join(result_info))
                f.write(txt + "\n")

            # update best mAP(IoU=0.50:0.95)
            if coco_mAP > best_map:
                best_map = coco_mAP

            if opt.savebest is False:
                # save weights every epoch
                with open(results_file, 'r') as f:
                    save_files = {
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'training_results': f.read(),
                        'epoch': epoch,
                        'best_map': best_map}
                    torch.save(save_files, "./weights/yolov3spp-{}.pt".format(epoch))
            else:
                # only save best weights
                if best_map == coco_mAP:
                    with open(results_file, 'r') as f:
                        save_files = {
                            'model': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'training_results': f.read(),
                            'epoch': epoch,
                            'best_map': best_map}
                        torch.save(save_files, best.format(epoch))
                        torch.save(model, "yolov3_spp")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='cfg/yolov3-spp.cfg', help="cfg/*.cfg path")
    parser.add_argument('--data', type=str, default='data/dataset.data', help='cfg/*.data path')
    parser.add_argument('--hyp', type=str, default='cfg/hyp.yaml', help='hyperparameters path')
    parser.add_argument('--weights', type=str, default='weights/yolov3spp-voc-512.pt',
                        help='pretrain weights path')
    parser.add_argument('--epochs', type=int, default=30, help="train epochs")
    parser.add_argument('--batch-size', type=int, default=4, help="train batch_size")
    parser.add_argument('--img-size', type=int, default=512, help='test size')
    parser.add_argument('--savebest', type=bool, default=False, help='only save best checkpoint')
    parser.add_argument('--notest', action='store_true', help='only test final epoch')
    parser.add_argument('--result-name', type=str, default='outputs/result.txt', help='results.txt name')
    parser.add_argument('--freeze-layers', type=bool, default=False, help='Freeze non-output layers')
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')


    parser.add_argument('--multi-scale', type=bool, default=True,
                        help='adjust (67%% - 150%%) img_size every 10 batches')
    parser.add_argument('--warmup', type=bool, default=False, help='warmup train')
    parser.add_argument('--augment', type=bool, default=True, help='dataset augment')
    parser.add_argument('--rect', type=bool, default=False, help='rect training')
    parser.add_argument('--mosaic', type=bool, default=False, help='mosaic augment')

    opt = parser.parse_args()

    # 检查文件是否存在
    opt.cfg = check_file(opt.cfg)
    opt.data = check_file(opt.data)
    opt.hyp = check_file(opt.hyp)
    print(opt)

    # 载入超参数文件
    with open(opt.hyp, encoding='UTF-8') as f:
        hyp = yaml.load(f, Loader=yaml.FullLoader)

    # 实例化 tensorboard
    tb_writer = SummaryWriter()

    train(hyp)