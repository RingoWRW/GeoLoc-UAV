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

from models.game4loc import DesModel

def get_parser():
    parser = argparse.ArgumentParser(description="Configuration for training the model")

    # Model Configurations
    parser.add_argument('--mode', type=str, default='vit_base_patch16_rope_reg1_gap_256.sbb_in1k', help='Model architecture')
    parser.add_argument('--model', type=str, default='vanilia', help='Path to save model checkpoints')
    

    # Dataset Paths
    parser.add_argument('--dataset_root', type=str, default='/media/guan/新加卷/EdgeBing/WorldLoc/', help='Root directory of the dataset')
    parser.add_argument('--test_txt', type=str, default='/media/Shen/Data/RingoData/DenseUAV/test/db.txt', help='Root directory of the dataset')
    parser.add_argument('--save_txt', type=str, default='/media/Shen/Data/RingoData/DenseUAV/test/db.txt', help='Root directory of the dataset')
   
   #'/media/Shen/Data/RingoData/WorldLoc/TestData/vpair test_40_midref_rot0'
    # Checkpoint Config
    parser.add_argument('--checkpoint_path', type=str, default="/media/guan/新加卷/Code/Code/vit_base_eva_gta_same_area.pth", help='Path to start from a checkpoint')

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


    config = {
        "mode": args.mode,
        "model": args.model,
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
        T.Resize((384, 384), interpolation=T.InterpolationMode.BILINEAR),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN_STD["mean"], std=IMAGENET_MEAN_STD["std"]),
    ])

model = DesModel(model_name='vit_base_patch16_rope_reg1_gap_256.sbb_in1k',
                    pretrained=True,
                    img_size=384,
                    share_weights=True)

if config["checkpoint_path"] is not None:  
    print("Start from:", config["checkpoint_path"])
    model_state_dict = torch.load(config["checkpoint_path"])  
    model.load_state_dict(model_state_dict, strict=True)     

    # Data parallel
    print("GPUs available:", torch.cuda.device_count())  
    # if torch.cuda.device_count() > 1 and len(config.gpu_ids) > 1:
    #     model = torch.nn.DataParallel(model, device_ids=config.gpu_ids)
            
    # Model to device   
    model = model.to(config["device"])

 #------------------------------------------------------------Eval---------------------------------------------------------------------#
result_list_recall = []
result_list_precision = []
with open(config['save_txt'], 'w') as f_w:
    with open(config["test_index_txt"],"r") as val_test:
        for line in val_test:
            if config["model"] == 'vanilia':
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
            
            pos_gt = eval_dataloader_db.dataset.get_gt()
            result, predictions, really_pos_gt = eval.evaluate(config, model, eval_dataloader_query, eval_dataloader_db, pos_gt, mode=config["model"],LPN=False)
            
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

        info1 = 'Average' + 'top 1: ' + str(round(average_result[0]*100,2)) + 'top 5: ' + str(round(average_result[1]*100,2)) +  'top 10' + str(round(average_result[2]*100,2)) + '\n'
        f_w.write(info1)
        f_w.write('AP@5 is'+str(round(av_p,2)))

        

# save top 1 flase or wrong
# with open(config.save_pred_txt, 'w') as f:
    for i in range(predictions.shape[0]):
        query_path = eval_dataloader_query.dataset.getitem(i)
        if np.any(np.in1d(predictions[i,0], really_pos_gt[i][1])):
            num = 1
        else:
            num = 0
        pred_path = eval_dataloader_db.dataset.samples[predictions[i,0]]
        info = query_path +  ' ' + pred_path + ' ' + str(num) + '\n'
        f_w.write(info)



                       
                    
