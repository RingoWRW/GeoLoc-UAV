import os
import sys
import torch
import argparse
import torch
from eval import eval
from torchvision import transforms as T
import numpy as np
import glob
from torch.utils.data import DataLoader
from dataset.World import  DenseUAVDatasetEvalVanilia
from dataset.World import AerialDatasetEvalVanilia
from models.anyloc import AnyModel

def get_parser():
    parser = argparse.ArgumentParser(description="Configuration for training the model")

    # Model Configurations
    parser.add_argument('--mode', type=str, default='dinov2_vitg14', help='Model architecture')
    parser.add_argument('--model', type=str, default='vanilia', help='Path to save model checkpoints')
   
    # Dataset Paths
    parser.add_argument('--dataset_query', type=str, default='/media/guan/新加卷/DenseUAV/DenseUAV/test/query.txt', help='Root directory of the dataset')
    parser.add_argument('--dataset_db', type=str, default='/media/guan/新加卷/DenseUAV/DenseUAV/test/db.txt', help='Root directory of the dataset')
    parser.add_argument('--dataset_gt', type=str, default='/media/guan/新加卷/DenseUAV/DenseUAV/test/gt.txt', help='Root directory of the dataset')
    parser.add_argument('--dataset_root_dir', type=str, default='/media/guan/新加卷/EdgeBing/TestData/test_40_midref_rot90/', help='Root directory of the dataset')
   #'/media/Shen/Data/RingoData/WorldLoc/TestData/vpair test_40_midref_rot0'
    # Checkpoint Config
    parser.add_argument('--checkpoint_path', type=str, default="/media/guan/新加卷/Code(1)/Code/vit_base_eva_gta_same_area.pth", help='Path to start from a checkpoint')
    parser.add_argument('--save_dir_path', type=str, default='/media/Shen/Data/RingoData/DenseUAV/test/db.txt', help='Root directory of the dataset')
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
    parser.add_argument('--gpu_ids', type=tuple, default=(0,), help='GPU IDs for training')

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
        # "dataset_query": args.dataset_query,
        # "dataset_db": args.dataset_db,
        # "dataset_gt": args.dataset_gt,
        "dataset_root_dir":args.dataset_root_dir,
        "checkpoint_path": args.checkpoint_path,
        "save_dir_path":args.save_dir_path,
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
        "lr_end": args.lr_end,
        "LPN":False
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

model = AnyModel(model_name=config['mode'],
                    pretrained=True)
 
model = model.to(config["device"])
 
# eva_dataset_query = DenseUAVDatasetEvalVanilia(txt=config['dataset_query'],
#                     mode='query',
#                     gt_txt=config["dataset_gt"],
#                     transforms=eval_transform)

# eval_dataloader_query = DataLoader(eva_dataset_query,
#                     batch_size=config['batch_size'],
#                     num_workers=config['num_workers'],
#                     shuffle=not config['custom_sampling'],
#                     pin_memory=True)

# eva_dataset_db = DenseUAVDatasetEvalVanilia(txt=config['dataset_db'],
#                         mode='DB',
#                         gt_txt=config["dataset_gt"],
#                         transforms=eval_transform)

# eval_dataloader_db = DataLoader(eva_dataset_db,
#                     batch_size=config['batch_size'],
#                     num_workers=config['num_workers'],
#                     shuffle=not config['custom_sampling'],
#                     pin_memory=True)



# pos_gt = eval_dataloader_db.dataset.get_gt()

if not os.path.exists(config['save_dir_path']):
    os.mkdir(config['save_dir_path'])

# test angle
angle_list = list(range(0, 1))
for angle in angle_list:
    eva_dataset_query = AerialDatasetEvalVanilia(data_dir=config['dataset_root_dir'],
                            mode='query',
                            angle=angle,
                            transforms=eval_transform)

    eval_dataloader_query = DataLoader(eva_dataset_query,
                            batch_size=config['batch_size'],
                            num_workers=config['num_workers'],
                            shuffle=not config['custom_sampling'],
                            pin_memory=True)

    eva_dataset_db = AerialDatasetEvalVanilia(data_dir=config['dataset_root_dir'],
                                mode='DB',
                                transforms=eval_transform)

    eval_dataloader_db = DataLoader(eva_dataset_db,
                            batch_size=config['batch_size'],
                            num_workers=config['num_workers'],
                            shuffle=not config['custom_sampling'],
                            pin_memory=True)
    pos_gt = eval_dataloader_db.dataset.get_gt()    
    result, predictions, really_pos_gt = eval.evaluate(config, model, eval_dataloader_query, eval_dataloader_db, pos_gt, mode=config["model"], LPN=config['LPN'])
    print(config['checkpoint_path'])
    print('top 1: ', round(result[0]*100,2),  'top 5: ', round(result[1]*100,2), 'top 10: ', round(result[2]*100,2)) #vanilia
    save_result_txt = config['save_dir_path'] + '/' + str(angle) + '.txt'
    with open(save_result_txt, 'w') as f_w:
        info = 'top 1: '+ str(round(result[0]*100,2)) + ' top 5: ' +str(round(result[1]*100,2)) + ' top 10: ' + str(round(result[2]*100,2))
        f_w.write(info + '\n')
    f_w.close()


# with open("/media/guan/新加卷/Code/result/anyloc/denseuav_g.txt", "w") as f_w:
#     for i in range(predictions.shape[0]):
#         query_path = eval_dataloader_query.dataset.getitem(i)
#         if np.any(np.in1d(predictions[i,0], really_pos_gt[i][1])):
#             num = 1
#         else:
#             num = 0
#         pred_path = eval_dataloader_db.dataset.samples[predictions[i,0]]
#         info = query_path +  ' ' + pred_path + ' ' + str(num) + '\n'
#         f_w.write(info)

