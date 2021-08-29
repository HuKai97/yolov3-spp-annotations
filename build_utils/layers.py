import torch
from torch import nn


class FeatureConcat(nn.Module):
    """将多个特征矩阵在channel维度进行concatenate拼接"""
    def __init__(self, layers):
        """
        :param layers: 记录下当前需要进行concat操作的所有层的层号  比如layers=-1,-3,-5,-6
        """
        super(FeatureConcat, self).__init__()
        self.layers = layers  # layer index 记录当前需要concat的层序号
        self.multiple = len(layers) > 1  # 是否是多层concat

    def forward(self, x, outputs):
        """
        :param outputs: 记录网络中每一层的输出feature map
        :return: 返回concat之后的feature map
        """
        return torch.cat([outputs[i] for i in self.layers], 1) if self.multiple else outputs[self.layers[0]]

class WeightedFeatureFusion(nn.Module):
    # weighted sum of 2 or more layers https://arxiv.org/abs/1911.09070
    """将多个特征矩阵的值进行融合(add操作) 只在残差结构用到  只在2个特征矩阵进行融合"""
    def __init__(self, layers, weight=False):
        """
        :param layers: 从当前layer往前数第3个层建立到当前层的shortcut  实际上这里全都是[-3]
        :param weight: False  这里是扩展功能 我们先不做考虑 没用到
        """
        super(WeightedFeatureFusion, self).__init__()
        self.layers = layers  # layer index 需要进行融合（add）
        self.weight = weight  # apply weights boolean 没用到
        self.n = len(layers) + 1  # number of layers 融合的特征矩阵个数  这里都是2
        if weight:
            self.w = nn.Parameter(torch.zeros(self.n), requires_grad=True)  # layer weights

    def forward(self, x, outputs):
        """
        :param x: 输入的特征矩阵   shortcut前一个卷积的输出特征
        :param outputs: 收集的整个网络的每一个模块的输出
        :return: 返回shortcut后的feature map
        """
        # Weights 跳过 不执行
        if self.weight:
            w = torch.sigmoid(self.w) * (2 / self.n)  # sigmoid weights (0-1)
            x = x * w[0]

        # Fusion
        nx = x.shape[1]  # input channels

        # 这里只会遍历一次 shortcut只是两个特征矩阵进行融合 self.n=2
        for i in range(self.n - 1):
            # 其实执行的是a = outputs[self.layers[i]] a=要融合的层结构的feature map
            a = outputs[self.layers[i]] * w[i + 1] if self.weight else outputs[self.layers[i]]
            na = a.shape[1]  # 取出要融合的层结构的feature map的channels

            # Adjust channels
            # 根据相加的两个特征矩阵的channel选择相加方式
            if nx == na:  # 残差结构只是用到这个  same shape 如果channel相同，直接相加
                x = x + a
            # 下面其实是不执行的
            elif nx > na:  # slice input 如果channel不同，将channel多的特征矩阵砍掉部分channel保证相加的channel一致
                x[:, :na] = x[:, :na] + a  # or a = nn.ZeroPad2d((0, 0, 0, 0, 0, dc))(a); x = x + a
            else:  # slice feature
                x = x + a[:, :nx]
        return x