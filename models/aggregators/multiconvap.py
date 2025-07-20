import torch
import torch.nn.functional as F
import torch.nn as nn

from models.aggregators.LPN import get_part_pool

class L2Norm(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim
    def forward(self, x):
        return F.normalize(x, p=2, dim=self.dim)

class GeMPool(nn.Module):
    """Implementation of GeM as in https://github.com/filipradenovic/cnnimageretrieval-pytorch
    we add flatten and norm so that we can use it as one aggregation layer.
    """
    def __init__(self, p=3, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1)*p)
        self.eps = eps

    def forward(self, x):
        x = F.avg_pool2d(x.clamp(min=self.eps).pow(self.p), (x.size(-2), x.size(-1))).pow(1./self.p)
        x = x.flatten(1)
        return F.normalize(x, p=2, dim=1)

class MulConvAP(nn.Module):
    """Implementation of ConvAP as of https://arxiv.org/pdf/2210.10239.pdf

    Args:
        in_channels (int): number of channels in the input of ConvAP
        out_channels (int, optional): number of channels that ConvAP outputs. Defaults to 512.
        s1 (int, optional): spatial height of the adaptive average pooling. Defaults to 2.
        s2 (int, optional): spatial width of the adaptive average pooling. Defaults to 2.
    """
    def __init__(self, in_channels, out_channels=512, s1=2, s2=2, LPN=False):
        super(MulConvAP, self).__init__()
        self.out_channels = out_channels
        self.channel_pool_1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1, bias=True)
        self.channel_pool_3 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding=1,bias=True)
        self.channel_pool_5 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=5, padding=2,bias=True)
      
        # self.AAP = nn.AdaptiveAvgPool2d((s1, s2))
        self.AAP = nn.Sequential(L2Norm(), GeMPool())

        # using LPN
        if LPN == True:
            self.LPN = True
        else:
            self.LPN = False
    def forward(self, x):

        if self.LPN == False:
            # x, t = x #dinov2专属
            x1 = self.channel_pool_1(x)
            x3 = self.channel_pool_3(x)
            x5 = self.channel_pool_5(x)

            x1 = self.AAP(x1)
            x3 = self.AAP(x3)
            x5 = self.AAP(x5)

            x = [i for i in [x1, x3, x5]]
            x = torch.cat(x,dim=1)

            # x = self.AAP(x)
            x = F.normalize(x.flatten(1), p=2, dim=1)
            return x
        else:
            partition_feature = get_part_pool(x)
            partition_feature_list = []
            for one_feature in partition_feature:
                x1 = self.channel_pool_1(one_feature)
                x3 = self.channel_pool_3(one_feature)
                x5 = self.channel_pool_5(one_feature)

                x1 = self.AAP(x1)
                x3 = self.AAP(x3)
                x5 = self.AAP(x5)

                x = [i for i in [x1, x3, x5]]
                x = torch.cat(x,dim=1)

                x = F.normalize(x.flatten(1), p=2, dim=1)
                partition_feature_list.append(x)
            # partition_feature_tensor = torch.stack(partition_feature_list, dim=2).reshape(x.shape[0], -1)
            
            return partition_feature_list


if __name__ == '__main__':
    x = torch.randn(4, 2048, 10, 10)
    # m = ConvAP(2048, 512)
    # r = m(x)
    # print(r.shape)