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
import argparse

def get_parser():
    parser = argparse.ArgumentParser(description="Configuration for training the model")

    # Model Configurations
    parser.add_argument('--mode', type=str, default='group', help='Model architecture')
    parser.add_argument('--model_path', type=str, default='./world', help='Path to save model checkpoints')
    
    # Group Config
    parser.add_argument('--group_arch', type=str, default='groupdinonet', help='Group architecture')
    parser.add_argument('--group_config', type=str, default='none', help='Group configuration')

    # Backbone Config
    parser.add_argument('--backbone_arch', type=str, default='dinov2_vits14', help='Backbone architecture')
    parser.add_argument('--pretrain_flag', type=bool, default=True, help='Flag to use pre-trained weights')

    # Agg Config
    parser.add_argument('--agg_arch', type=str, default='multiconvap', help='Aggregation architecture')
    parser.add_argument('--agg_in_channels', type=int, default=384, help='Input channels for aggregation')
    parser.add_argument('--agg_out_channels', type=int, default=384, help='Output channels for aggregation')
    parser.add_argument('--agg_s1', type=int, default=1, help='Aggregation s1 parameter')
    parser.add_argument('--agg_s2', type=int, default=1, help='Aggregation s2 parameter')
    parser.add_argument('--agg_LPN', type=bool, default=False, help='Use LPN for aggregation')

    # Dataset Paths
    parser.add_argument('--dataset_root', type=str, default='/media/guan/新加卷/EdgeBing/WorldLoc/ya1/', help='Root directory of the dataset')
    parser.add_argument('--test_txt', type=str, default='/media/Shen/Data/RingoData/DenseUAV/test/db.txt', help='Root directory of the dataset')
    parser.add_argument('--save_txt', type=str, default='/media/Shen/Data/RingoData/DenseUAV/test/db.txt', help='Root directory of the dataset')
   
   #'/media/Shen/Data/RingoData/WorldLoc/TestData/vpair test_40_midref_rot0'
    # Checkpoint Config
    parser.add_argument('--checkpoint_path', type=str, default=None, help='Path to start from a checkpoint')

    # Training Parameters
    parser.add_argument('--num_workers', type=int, default=0 if os.name == 'nt' else 4, help='Number of workers for data loading')
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu', help='Device for training')
    parser.add_argument('--cudnn_benchmark', type=bool, default=True, help='Use cudnn benchmark for performance')
    parser.add_argument('--cudnn_deterministic', type=bool, default=False, help='Make cudnn deterministic')

    # Training Settings
    parser.add_argument('--mixed_precision', type=bool, default=True, help='Use mixed precision training')
    parser.add_argument('--custom_sampling', type=bool, default=True, help='Use custom sampling')
    parser.add_argument('--seed', type=int, default=1, help='Random seed')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--verbose', type=bool, default=True, help='Verbose output during training')
    parser.add_argument('--gpu_ids', type=tuple, default=(1,), help='GPU IDs for training')

    # Optimizer Config
    parser.add_argument('--clip_grad', type=float, default=100.0, help='Clip gradients (None or float)')
    parser.add_argument('--decay_exclude_bias', type=bool, default=False, help='Exclude bias from decay')
    parser.add_argument('--grad_checkpointing', type=bool, default=False, help='Use gradient checkpointing')

    # Loss Config
    parser.add_argument('--label_smoothing', type=float, default=0.1, help='Label smoothing factor')

    # Learning Rate
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--scheduler', type=str, default='cosine', help='Learning rate scheduler')
    parser.add_argument('--warmup_epochs', type=float, default=0.1, help='Warmup epochs for learning rate')
    parser.add_argument('--lr_end', type=float, default=0.0001, help='End learning rate for polynomial scheduler')

    return parser

def parse_config():
    parser = get_parser()
    args = parser.parse_args()

    # Build the config dictionaries dynamically based on parsed args
    group_config = {
        "group_arch": args.group_arch,
        "group_config": {args.group_config}
    }

    backbone_config = {
        "backbone_arch": args.backbone_arch,
        "pretrain_flag": args.pretrain_flag
    }

    agg_config = {
        "agg_arch": args.agg_arch,
        "agg_config": {
            "in_channels": args.agg_in_channels,
            "out_channels": args.agg_out_channels,
            "s1": args.agg_s1,
            "s2": args.agg_s2,
            "LPN": args.agg_LPN
        }
    }

    config = {
        "mode": args.mode,
        "model_path": args.model_path,
        "group": group_config,
        "backbone": backbone_config,
        "agg": agg_config,
        "dataset_root_dir": args.dataset_root,
        "test_index_txt": args.test_txt,
        "save_txt":args.save_txt,
        "checkpoint_path": args.checkpoint_path,
        "num_workers": args.num_workers,
        "device": args.device,
        "cudnn_benchmark": args.cudnn_benchmark,
        "cudnn_deterministic": args.cudnn_deterministic,
        "mixed_precision": args.mixed_precision,
        "custom_sampling": args.custom_sampling,
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "verbose": args.verbose,
        "gpu_ids": args.gpu_ids,
        "clip_grad": args.clip_grad,
        "decay_exclude_bias": args.decay_exclude_bias,
        "grad_checkpointing": args.grad_checkpointing,
        "label_smoothing": args.label_smoothing,
        "lr": args.lr,
        "scheduler": args.scheduler,
        "warmup_epochs": args.warmup_epochs,
        "lr_end": args.lr_end
    }

    return args, config
    

#-------------------------------------------------------------------------------------------#
# Train Config
#-------------------------------------------------------------------------------------------#
args, config = parse_config()
IMAGENET_MEAN_STD = {'mean': [0.485, 0.456, 0.406], 
                    'std': [0.229, 0.224, 0.225]}
eval_transform = T.Compose([
        T.Resize((224, 224), interpolation=T.InterpolationMode.BILINEAR),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN_STD["mean"], std=IMAGENET_MEAN_STD["std"]),
    ])

if config["mode"] == 'vanilia':

    model = model.BackboneGlobal(config["backbone"]['backbone_arch'],
                             config["backbone"]['pretrain_flag'],
                               config["agg"]['agg_arch'],
                               config["agg"]['agg_config'])
    model_state_dict = torch.load(config['checkpoint_path'], map_location=config['device'])
    model.load_state_dict(model_state_dict, strict=False)

    model = model.to(config['device'])

# model = model.GrounpGlobal(config.group['group_arch'],
#                                config.agg['agg_arch'],
#                                config.agg['agg_config'])
else:
    model = model.GrounpDinoGlobal(config["group"]['group_arch'],
                                config["agg"]['agg_arch'],
                                config["agg"]['agg_config'])

    model_state_dict = torch.load(config['checkpoint_path'], map_location=config['device'])
    model.load_state_dict(model_state_dict, strict=False)

    model = model.to(config['device'])

 #------------------------------------------------------------Eval---------------------------------------------------------------------#
result_list_recall = []
result_list_precision = []
with open(config['save_txt'], 'w') as f_w:
    with open(config["test_index_txt"],"r") as val_test:
        for line in val_test:
            if config["mode"] == 'vanilia':
                eva_dataset_query = WorldDatasetEvalVanilia(data_dir=config["dataset_root_dir"],
                                        name=line.strip('\n'),
                                        mode='query',
                                        transforms=eval_transform)

                eval_dataloader_query = DataLoader(eva_dataset_query,
                                    batch_size=config["batch_size"],
                                    num_workers=config["num_workers"],
                                    shuffle=not config["custom_sampling"],
                                    pin_memory=True)
                
                eva_dataset_db = WorldDatasetEvalVanilia(data_dir=config["dataset_root_dir"],
                                        name=line.strip('\n'),
                                        mode='DB',
                                        transforms=eval_transform)

                eval_dataloader_db = DataLoader(eva_dataset_db,
                                    batch_size=config["batch_size"],
                                    num_workers=config["num_workers"],
                                    shuffle=not config["custom_sampling"],
                                    pin_memory=True)
            else:
                eva_dataset_query = WorldDatasetEvalGroup(data_dir=config["dataset_root_dir"],
                                        name=line.strip('\n'),
                                        mode='query',
                                        transforms=eval_transform)

                eval_dataloader_query = DataLoader(eva_dataset_query,
                                    batch_size=config["batch_size"],
                                    num_workers=config["num_workers"],
                                    shuffle=not config["custom_sampling"],
                                    pin_memory=True)
                
                eva_dataset_db = WorldDatasetEvalGroup(data_dir=config["dataset_root_dir"],
                                        name=line.strip('\n'),
                                        mode='DB',
                                        transforms=eval_transform)

                eval_dataloader_db = DataLoader(eva_dataset_db,
                                    batch_size=config["batch_size"],
                                    num_workers=config["num_workers"],
                                    shuffle=not config["custom_sampling"],
                                    pin_memory=True)
            
            pos_gt = eval_dataloader_db.dataset.get_gt()
            result, predictions, really_pos_gt = eval.evaluate(config, model, eval_dataloader_query, eval_dataloader_db, pos_gt, mode=config["mode"],LPN=config["agg"]['agg_config']['LPN'])
            print('top 1: ', round(result[0]*100,2),  'top 5: ', round(result[1]*100,2), 'top 10: ', round(result[2]*100,2)) #vanilia
            f_w.write(line + ' ' + str(round(result[0]*100,2)) + ' ' + str(round(result[1]*100,2)) + '\n')


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

        info1 = 'Average' + 'top 1: ' + str(round(average_result[0]*100,2)) + 'top 5: ' + str(round(average_result[1]*100,2)) + 'top 10:' + str(round(average_result[2]*100,2)) + '\n'
        f_w.write(info1)
        f_w.write('AP@5 is'+str(round(av_p,2)))

        

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



                       
                    
