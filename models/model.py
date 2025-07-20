import numpy as np
import torch.nn as nn
import torch

from models import helper

class GrounpDinoGlobal(nn.Module):

    def __init__(self, 
                groupnet_arch,
                agg_arch, 
                agg_config ):
    
        super(GrounpDinoGlobal, self).__init__()

        self.groupnet = helper.get_groupdinonet(groupnet_arch)
        self.aggregator = helper.get_aggregator(agg_arch, agg_config)
        self.logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    
    def forward(self, x, pts_list):

        local_feature, gfeats_lists = self.groupnet(x, pts_list)
        local_feature = local_feature.permute(0,2,1).unsqueeze(-1)
        global_feature = self.aggregator(local_feature)
        


        # img_num = len(x)
        # bs = x[0][0].shape[0]

        # global_feature = torch.zeros(bs*len(x), 256, device='cuda')
        # for i in range(img_num):
        #     imgs, pts = x[i], pts_list[i]
        #     local_feature = self.groupnet(imgs, pts)
        #     local_feature = local_feature.permute(0,2,1).unsqueeze(-1)
        #     des = self.aggregator(local_feature)
        #     for j in range(len(des)):
        #         global_feature[j*img_num+i,:] = des[j,:]
        
        return global_feature, gfeats_lists


class GrounpGlobal(nn.Module):

    def __init__(self, 
                groupnet_arch,
                agg_arch, 
                agg_config ):
    
        super(GrounpGlobal, self).__init__()

        self.groupnet = helper.get_groupnet(groupnet_arch)
        self.aggregator = helper.get_aggregator(agg_arch, agg_config)
        self.logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    
    def forward(self, x, pts_list):

        local_feature, gfeats_lists = self.groupnet(x, pts_list)
        local_feature = local_feature.permute(0,2,1).unsqueeze(-1)
        global_feature = self.aggregator(local_feature)


        # img_num = len(x)
        # bs = x[0][0].shape[0]

        # global_feature = torch.zeros(bs*len(x), 256, device='cuda')
        # for i in range(img_num):
        #     imgs, pts = x[i], pts_list[i]
        #     local_feature = self.groupnet(imgs, pts)
        #     local_feature = local_feature.permute(0,2,1).unsqueeze(-1)
        #     des = self.aggregator(local_feature)
        #     for j in range(len(des)):
        #         global_feature[j*img_num+i,:] = des[j,:]
        
        return global_feature, gfeats_lists
    
class BackboneGlobal(nn.Module):

    def __init__(self, 
                backbone_arch,
                pretrain_flag,
                agg_arch, 
                agg_config ):
    
        super(BackboneGlobal, self).__init__()

        self.backbone = helper.get_backbone(backbone_arch, pretrain_flag)
        self.aggregator = helper.get_aggregator(agg_arch, agg_config)
        self.logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        if 'dinov2' in backbone_arch.lower():
            self.FLAG = True 
        else:
            self.FLAG = False

    
    def forward(self, x):

        local_feature = self.backbone(x)
        
        # dinov2
        
        if self.FLAG:
            global_feature = self.aggregator(local_feature[0])
        else:
            global_feature = self.aggregator(local_feature)


        # img_num = len(x)
        # bs = x[0][0].shape[0]

        # global_feature = torch.zeros(bs*len(x), 256, device='cuda')
        # for i in range(img_num):
        #     imgs, pts = x[i], pts_list[i]
        #     local_feature = self.groupnet(imgs, pts)
        #     local_feature = local_feature.permute(0,2,1).unsqueeze(-1)
        #     des = self.aggregator(local_feature)
        #     for j in range(len(des)):
        #         global_feature[j*img_num+i,:] = des[j,:]
        
        return global_feature