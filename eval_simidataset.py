from torch.utils.data import DataLoader
from dataclasses import dataclass,field
from eval import eval
import os
import torch
from torchvision import transforms as T
from dataset.World import  WorldDatasetEvalVanilia, WorldDatasetEvalGroup
from models import model
import glob
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def default_group_config():

    return {
          "group_arch" : "groupdinonet", #group
        "group_config": {
            "none"
        }
    }

def default_backbone_config():

    return {
          "backbone_arch" : "resnet18",  #dinov2_vitb14,resnet18
          "pretrain_flag":True
    }

def default_agg_config():

    return {
          "agg_arch": "multiconvap", #convap
    "agg_config": {
      "in_channels": 512, #256 #512，768
      "out_channels": 512, #256
      "s1": 1,
      "s2": 1,
      'LPN':False
        }
    }

@dataclass
class Configuration:

    model: str = "resnet18"

    # Savepath for model checkpoints
    model_path: str = "./world"

    # model config
    group:dict = field(default_factory=default_group_config)
    backbone:dict = field(default_factory=default_backbone_config)

    agg:dict = field(default_factory=default_agg_config)

    # dataset
    dataset_root_dir: str = "/media/Shen/Data/RingoData/WorldLoc"
    train_query_txt: str = "/media/Shen/Data/RingoData/WorldLoc/Index/train_query.txt"

    # val_index
    val_index_txt = "/media/Shen/Data/RingoData/WorldLoc/Index/val.txt"

    # test_index
    test_index_txt = "/media/Shen/Data/RingoData/WorldLoc/Index/test_country.txt"
    save_pred_txt = "/media/Shen/Data/RingoData/WorldLoc/txt/new_rot/dinos-finetune.txt"


      # Checkpoint to start from
    checkpoint_start = None

    # set num_workers to 0 if on Windows
    num_workers: int = 0 if os.name == 'nt' else 4 
    
    # train on GPU if available
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu' 

        # for better performance
    cudnn_benchmark: bool = True
    
    # make cudnn deterministic
    cudnn_deterministic: bool = False

    # trainning 
    mixed_precision: bool = True
    custom_sampling: bool = True         # use custom sampling instead of random
    seed = 1
    epochs: int = 30
    batch_size: int = 10                # keep in mind real_batch_size = 2 * batch_size 128
    verbose: bool = True
    gpu_ids: tuple = (1,)           # GPU ids for training

      # Optimizer 
    clip_grad = 100.                     # None | float
    decay_exclue_bias: bool = False
    grad_checkpointing: bool = False     # Gradient Checkpointing
    
    # Loss
    label_smoothing: float = 0.1
    
    # Learning Rate
    lr: float = 0.001                    # 1 * 10^-4 for ViT | 1 * 10^-1 for CNN
    scheduler: str = "cosine"           # "polynomial" | "cosine" | "constant" | None
    warmup_epochs: int = 0.1
    lr_end: float = 0.0001               #  only for "polynomial"

    

#-------------------------------------------------------------------------------------------#
# Train Config
#-------------------------------------------------------------------------------------------#
config = Configuration() 


IMAGENET_MEAN_STD = {'mean': [0.485, 0.456, 0.406], 
                    'std': [0.229, 0.224, 0.225]}
eval_transform = T.Compose([
        T.Resize((224, 224), interpolation=T.InterpolationMode.BILINEAR),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN_STD["mean"], std=IMAGENET_MEAN_STD["std"]),
    ])

model = model.BackboneGlobal(config.backbone['backbone_arch'],
                             config.backbone['pretrain_flag'],
                               config.agg['agg_arch'],
                               config.agg['agg_config'])

# model = model.GrounpGlobal(config.group['group_arch'],
#                                config.agg['agg_arch'],
#                                config.agg['agg_config'])

# model = model.GrounpDinoGlobal(config.group['group_arch'],
#                                config.agg['agg_arch'],
#                                config.agg['agg_config'])

model_state_dict = torch.load("PTH/city/resnet_frozen/weights_e5_0.6413.pth", map_location='cuda:1')
model.load_state_dict(model_state_dict, strict=False)

model = model.to(config.device)

 #------------------------------------------------------------Eval---------------------------------------------------------------------#
result_list_recall = []
result_list_precision = []
with open(config.test_index_txt,"r") as val_test:
    for line in val_test:
        eva_dataset_query = WorldDatasetEvalVanilia(data_dir=config.dataset_root_dir,
                                name=line.strip('\n'),
                                mode='query',
                                transforms=eval_transform)

        eval_dataloader_query = DataLoader(eva_dataset_query,
                            batch_size=config.batch_size,
                            num_workers=config.num_workers,
                            shuffle=not config.custom_sampling,
                            pin_memory=True)
        
        eva_dataset_db = WorldDatasetEvalVanilia(data_dir=config.dataset_root_dir,
                                name=line.strip('\n'),
                                mode='DB',
                                transforms=eval_transform)

        eval_dataloader_db = DataLoader(eva_dataset_db,
                            batch_size=config.batch_size,
                            num_workers=config.num_workers,
                            shuffle=not config.custom_sampling,
                            pin_memory=True)
        
        # eva_dataset_query = WorldDatasetEvalGroup(data_dir=config.dataset_root_dir,
        #                         name=line.strip('\n'),
        #                         mode='query',
        #                         transforms=eval_transform)

        # eval_dataloader_query = DataLoader(eva_dataset_query,
        #                     batch_size=config.batch_size,
        #                     num_workers=config.num_workers,
        #                     shuffle=not config.custom_sampling,
        #                     pin_memory=True)
        
        # eva_dataset_db = WorldDatasetEvalGroup(data_dir=config.dataset_root_dir,
        #                         name=line.strip('\n'),
        #                         mode='DB',
        #                         transforms=eval_transform)

        # eval_dataloader_db = DataLoader(eva_dataset_db,
        #                     batch_size=config.batch_size,
        #                     num_workers=config.num_workers,
        #                     shuffle=not config.custom_sampling,
        #                     pin_memory=True)
        
        pos_gt = eval_dataloader_db.dataset.get_gt()
        result, predictions, really_pos_gt = eval.evaluate(config, model, eval_dataloader_query, eval_dataloader_db, pos_gt, mode='vanilia',LPN=config.agg['agg_config']['LPN'])
        print('top 1: ', round(result[0]*100,2),  'top 5: ', round(result[1]*100,2), 'top 10: ', round(result[2]*100,2)) #vanilia
      

        # ap@5
        ap_list = []
        for i in range(predictions.shape[0]):
            ex = np.isin(predictions[i, 5:], really_pos_gt[i][1])
            num_all = np.sum(ex) / 5 * 100
            ap_list.append(num_all)
        average_ap = np.mean(np.array(ap_list))
        
        result_list_recall.append(result)
        result_list_precision.append(average_ap)

    result_array = np.array(result_list_recall)
    average_result = np.mean(result_array, axis=0)
    print('Average', 'top 1: ', round(average_result[0]*100,2),  'top 5: ', round(average_result[1]*100,2), 'top 10: ', round(average_result[2]*100,2))
    
   
    result_precision = np.array(result_list_precision)
    av_p = np.mean(result_precision)
    print('AP@5 is', round(av_p,2))

# save top 1 flase or wrong
# with open(config.save_pred_txt, 'w') as f:
#     for i in range(predictions.shape[0]):
#         query_path = eval_dataloader_query.dataset.getitem(i)
#         if np.any(np.in1d(predictions[i,0], really_pos_gt[i][1])):
#             num = 1
#         else:
#             num = 0
#         pred_path = eval_dataloader_db.dataset.samples[predictions[i,0]]
#         info = query_path +  ' ' + pred_path + ' ' + str(num) + '\n'
#         f.write(info)



                       
                    
