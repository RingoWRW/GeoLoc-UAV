import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.utils import dim_extend,interpolate_feats,l2_normalize
import json

json_path = "/media/guan/新加卷/Code/Code/configs/transform_config.json"
with open(json_path, 'r', encoding='utf-8') as file:
    data = json.load(file)
group_config = data["transform_config"]

# class GroupNetConfig:
#     def __init__(self):
#         self.sample_scale_begin = 0
#         self.sample_scale_inter = 0.5 
#         self.sample_scale_num = 3

#         self.sample_rotate_begin = -45
#         self.sample_rotate_inter = 45
#         self.sample_rotate_num = 8

# class GroupNetConfig:
#     def __init__(self):
#         self.sample_scale_begin = 0
#         self.sample_scale_inter = 1 
#         self.sample_scale_num = 1

#         self.sample_rotate_begin = 0
#         self.sample_rotate_inter = 0
#         self.sample_rotate_num = 1
# group_config = GroupNetConfig()

class VanillaLightCNN(nn.Module):
    def __init__(self):
        super(VanillaLightCNN, self).__init__()
        self.conv0 = nn.Sequential(
            nn.Conv2d(384,384//2,1,1,bias=False),
            nn.InstanceNorm2d(384//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(384//2,384//4,1,1,bias=False),
            nn.InstanceNorm2d(384//4),
            nn.ReLU(inplace=True),
            nn.Conv2d(384//4,64,1,1,bias=False),
            nn.InstanceNorm2d(64),
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(3,16,5,1,2,bias=False),
            nn.InstanceNorm2d(16),
            nn.ReLU(inplace=True),

            nn.Conv2d(16,32,5,1,2,bias=False),
            nn.InstanceNorm2d(32),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2, 2))
        self.proj = nn.Conv2d(96, 64, 1, 1, bias=False)

       

    def forward(self, x, img):
        x_dino=self.conv0(x)
        x_resized = F.interpolate(img, size=(32, 32), mode='bilinear', align_corners=False)
        x_cnn = self.conv1(x_resized)
        x_cat = torch.concat((x_dino, x_cnn), dim=1)
        x_proj = self.proj(x_cat)
        x=l2_normalize(x_proj,axis=1)  # [1,c,w//2, h//2]
        return x

class ExtractorWrapper(nn.Module):
    def __init__(self,scale_num, rotation_num):
        super(ExtractorWrapper, self).__init__()
        self.extractor=VanillaLightCNN()
        self.sn, self.rn = scale_num, rotation_num 
       
        dinov2_weights = torch.hub.load('facebookresearch/dinov2', "dinov2_vits14")
        # torch.load("/media/Shen/Data/RingoData/WorldLoc/Code/dinov2_vits14_pretrain.pth")
        from models.transformer import vit_small
        vit_kwargs = dict(
            patch_size= 14,
            img_size=518,
            init_values = 1.0,
            ffn_layer = "mlp",
            block_chunks = 0,
        )

        self.dinov2_vits14 = vit_small(**vit_kwargs).eval()
        # self.dinov2_vits14.load_state_dict(dinov2_weights)

    def forward(self,img_list,pts_list):
        '''

        :param img_list:  list of [b,3,h,w]
        :param pts_list:  list of [b,n,2]
        :return:gefeats [b,n,f,sn,rn]
        '''
        assert(len(img_list)==self.rn*self.sn)
        gfeats_list = []
        # feature extraction
        
        for img_index,img in enumerate(img_list):
            # extract feature
            
            with torch.no_grad():
                dinov2_features_16 = self.dinov2_vits14.forward_features(img)
                B, _, H, W = img.shape
                features_16 = dinov2_features_16['x_norm_patchtokens'].permute(0,2,1).reshape(B,-1,H//14, W//14)
                

            feats=self.extractor(features_16, img)
            gfeats_list.append(interpolate_feats(img, pts_list[img_index], feats)[:,:,:,None])
            
        gfeats_list=torch.cat(gfeats_list,3)  # b,n,f,sn*rn
        b,n,f,_=gfeats_list.shape
        gfeats_list=gfeats_list.reshape(b,n,f,self.sn,self.rn)
        
        return gfeats_list

class BilinearGCNN(nn.Module):
    def __init__(self, scale_num, rotation_num):
        super(BilinearGCNN, self).__init__()

        self.r, self.s = rotation_num, scale_num

        self.network1_embed1 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3, 1, 1),
        )
        self.network1_embed1_short = nn.Conv2d(64, 64, 1, 1)
        self.network1_embed1_relu = nn.ReLU(True)

        self.network1_embed2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3, 1, 1),
        )
        self.network1_embed2_short = nn.Conv2d(64, 64, 1, 1)
        self.network1_embed2_relu = nn.ReLU(True)

        self.network1_embed3 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 16, 3, 1, 1),
        )

        ###########################
        self.network2_embed1 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3, 1, 1),
        )
        self.network2_embed1_short = nn.Conv2d(64, 64, 1, 1)
        self.network2_embed1_relu = nn.ReLU(True)

        self.network2_embed2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3, 1, 1),
        )
        self.network2_embed2_short = nn.Conv2d(64, 64, 1, 1)
        self.network2_embed2_relu = nn.ReLU(True)

        self.network2_embed3 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 16, 3, 1, 1),
        )

    def forward(self, x):
        '''

        :param x:  b,n,f,ssn,srn
        :return:
        '''
        
        b, n, f, ssn, srn = x.shape
        # equal = x.reshape(b, n, f, ssn*srn)
        # equ_features=torch.max(equal,dim=-1,keepdim=False)[0]
        # x = l2_normalize(equ_features, axis=1)
        assert (ssn == self.s and srn == self.r)
        x = x.reshape(b * n, f, ssn, srn)

        x1 = self.network1_embed1_relu(self.network1_embed1(x) + self.network1_embed1_short(x))
        x1 = self.network1_embed2_relu(self.network1_embed2(x1) + self.network1_embed2_short(x1))
        x1 = self.network1_embed3(x1)

        x2 = self.network2_embed1_relu(self.network2_embed1(x) + self.network2_embed1_short(x))
        x2 = self.network2_embed2_relu(self.network2_embed2(x2) + self.network2_embed2_short(x2))
        x2 = self.network2_embed3(x2)

        x1 = x1.reshape(b * n, 16, self.s * self.r)
        x2 = x2.reshape(b * n, 16, self.s * self.r).permute(0, 2, 1)  # b*n,25,16
        x = torch.bmm(x1, x2).reshape(b * n, 256)  # b*n,8,25
        assert (x.shape[1] == 256)
        x=x.reshape(b,n,256)
        x=l2_normalize(x,axis=2)
        return x

class EmbedderWrapper(nn.Module):
    def __init__(self, scale_num, rotation_num):
        super(EmbedderWrapper, self).__init__()
        self.embedder=BilinearGCNN(scale_num, rotation_num)

    def forward(self, gfeats):
        # group cnns
        gefeats=self.embedder(gfeats) # b,n,f
        return gefeats

class GroupDinoNet(nn.Module):
    def __init__(self, config=group_config):
        super(GroupDinoNet, self).__init__()
        self.scale_num = config["sample_scale_num"]
        self.rotation_num = config["sample_rotate_num"]
       

        self.extractor=ExtractorWrapper(self.scale_num, self.rotation_num).cuda()
        self.embedder=EmbedderWrapper(self.scale_num, self.rotation_num).cuda()

    def forward(self, img_list, pts_list):
        gfeats=self.extractor(dim_extend(img_list),dim_extend(pts_list))
        efeats=self.embedder(gfeats)
        return efeats, gfeats