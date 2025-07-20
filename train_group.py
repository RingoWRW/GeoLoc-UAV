import os
import time
import numpy as np
import math
import shutil
import sys
import torch
from dataclasses import dataclass,field
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from transformers import get_constant_schedule_with_warmup, get_polynomial_decay_schedule_with_warmup, get_cosine_schedule_with_warmup
from torchvision import transforms as T
from torch.utils.tensorboard import SummaryWriter

from dataset.World import WorldDatasetTrainGroup, WorldDatasetEvalGroup
from models import model,trainer
from utils import setting
from utils import loss
from eval import eval

def default_group_config():

    return {
          "group_arch" : "groupnet", #group
        "group_config": {
            "none"
        }
    }

def default_backbone_config():

    return {
          "backbone_arch" : "resnet18", 
    }

def default_agg_config():

    return {
          "agg_arch": "multiconvap", #convap
    "agg_config": {
      "in_channels": 256, #256 #512
      "out_channels": 256, #256
      "s1": 1,
      "s2": 1,
      'LPN':False
        }
    }

@dataclass
class Configuration:

    model: str = "group34"

    # Savepath for model checkpoints
    model_path: str = "./world"

    # model config
    group:dict = field(default_factory=default_group_config)
    agg:dict = field(default_factory=default_agg_config)

    # dataset
    dataset_root_dir: str = "/media/Shen/Data/RingoData/WorldLoc"
    train_query_txt: str = "/media/Shen/Data/RingoData/WorldLoc/Index/train_query.txt"

    # val_index
    val_index_txt = "/media/Shen/Data/RingoData/WorldLoc/Index/val.txt"

    # test_index
    test_index_txt = "/media/Shen/Data/RingoData/WorldLoc/Index/test.txt"


      # Checkpoint to start from
    checkpoint_start = None

    # set num_workers to 0 if on Windows
    num_workers: int = 0 if os.name == 'nt' else 4 
    
    # train on GPU if available
    device: str = 'cuda:1' if torch.cuda.is_available() else 'cpu' 

        # for better performance
    cudnn_benchmark: bool = True
    
    # make cudnn deterministic
    cudnn_deterministic: bool = False

    # trainning 
    mixed_precision: bool = True
    custom_sampling: bool = True         # use custom sampling instead of random
    seed = 1
    epochs: int = 30
    batch_size: int = 128                # keep in mind real_batch_size = 2 * batch_size 128
    verbose: bool = True
    gpu_ids: tuple = (1,2,3)           # GPU ids for training

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


if __name__ == '__main__':

    model_path = "{}/{}/{}".format(config.model_path,
                                       config.model,
                                       time.strftime("%H%M%S"))

    if not os.path.exists(model_path):
        os.makedirs(model_path)
    shutil.copyfile(os.path.basename(__file__), "{}/train.py".format(model_path))

    # Redirect print to both console and log file
    sys.stdout = setting.Logger(os.path.join(model_path, 'log.txt'))

    setting.setup_system(seed=config.seed,
                 cudnn_benchmark=config.cudnn_benchmark,
                 cudnn_deterministic=config.cudnn_deterministic)
    
    #-----------------------------------------------------------------------------#
    # Model                                                                       #
    #-----------------------------------------------------------------------------#
        
    print("\nModel: {}".format(config.model))

  
    # backbone
    model = model.GrounpGlobal(config.group['group_arch'],
                                config.agg['agg_arch'],
                                config.agg['agg_config'])
    
     # Load pretrained Checkpoint    
    if config.checkpoint_start is not None:  
        print("Start from:", config.checkpoint_start)
        model_state_dict = torch.load(config.checkpoint_start)  
        model.load_state_dict(model_state_dict, strict=False)   
    
       # Data parallel
    print("GPUs available:", torch.cuda.device_count())  
    if torch.cuda.device_count() > 1 and len(config.gpu_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=config.gpu_ids)

      # Model to device   
    model = model.to(config.device)

    #------------------------setting dataset-------------------------------------------------#
    IMAGENET_MEAN_STD = {'mean': [0.485, 0.456, 0.406], 
                     'std': [0.229, 0.224, 0.225]}
    train_transform = T.Compose([
            T.Resize((224, 224), interpolation=T.InterpolationMode.BILINEAR),
            T.RandAugment(num_ops=3, interpolation=T.InterpolationMode.BILINEAR),
            T.AugMix(),
            # T.ColorJitter(brightness=0.5, contrast=0.1, saturation=0.1,
            #                         hue=0),
            # T.RandomGrayscale(p=0.2),
            # T.RandomPosterize(p=0.2, bits=4),
            # T.GaussianBlur(kernel_size=(1, 5), sigma=(0.1, 5)),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN_STD["mean"], std=IMAGENET_MEAN_STD["std"]),
        ])
    
    eval_transform = T.Compose([
            T.Resize((224, 224), interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN_STD["mean"], std=IMAGENET_MEAN_STD["std"]),
        ])

    #-----------------------------------------------------------------------------#
    # DataLoader                                                                  #
    #-----------------------------------------------------------------------------#

    train_dataset = WorldDatasetTrainGroup(data_dir=config.dataset_root_dir,
                                      query_txt=config.train_query_txt,
                                      transforms_query=train_transform,
                                      transforms_db=train_transform,
                                      shuffle_batch_size=config.batch_size)
    
    # train_dataloader = DataLoader(train_dataset,
    #                               batch_size=config.batch_size,
    #                               num_workers=config.num_workers,
    #                               shuffle=not config.custom_sampling,
    #                               pin_memory=True)
    
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=config.batch_size,
                                  num_workers=config.num_workers,
                                  shuffle=config.custom_sampling,
                                  pin_memory=True)
    
    #-----------------------------------------------------------------------------#
    # Loss                                                                        #
    #-----------------------------------------------------------------------------#

    # InfoNCE loss
    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    loss_function = loss.InfoNCE(loss_function=loss_fn,
                            device=config.device,
                            )
    # Supervised Contrastive loss
    # loss_function = loss.SupervisedContrastiveLoss(temperature = 0.07, device=config.device)

    if config.mixed_precision:
        scaler = GradScaler(init_scale=2.**10)
    else:
        scaler = None
    
    #-----------------------------------------------------------------------------#
    # optimizer                                                                   #
    #-----------------------------------------------------------------------------#
        
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)

        #-----------------------------------------------------------------------------#
    # Scheduler                                                                   #
    #-----------------------------------------------------------------------------#

    train_steps = len(train_dataloader) * config.epochs
    warmup_steps = len(train_dataloader) * config.warmup_epochs
       
    if config.scheduler == "polynomial":
        print("\nScheduler: polynomial - max LR: {} - end LR: {}".format(config.lr, config.lr_end))  
        scheduler = get_polynomial_decay_schedule_with_warmup(optimizer,
                                                              num_training_steps=train_steps,
                                                              lr_end = config.lr_end,
                                                              power=1.5,
                                                              num_warmup_steps=warmup_steps)
        
    elif config.scheduler == "cosine":
        print("\nScheduler: cosine - max LR: {}".format(config.lr))   
        scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                    num_training_steps=train_steps,
                                                    num_warmup_steps=warmup_steps)
        
    elif config.scheduler == "constant":
        print("\nScheduler: constant - max LR: {}".format(config.lr))   
        scheduler =  get_constant_schedule_with_warmup(optimizer,
                                                       num_warmup_steps=warmup_steps)
           
    else:
        scheduler = None
        
    print("Warmup Epochs: {} - Warmup Steps: {}".format(str(config.warmup_epochs).ljust(2), warmup_steps))
    print("Train Epochs:  {} - Train Steps:  {}".format(config.epochs, train_steps))


    #-----------------------------------------------------------------------------#
    # Shuffle                                                                     #
    #-----------------------------------------------------------------------------#            
    if config.custom_sampling:
        train_dataloader.dataset.shuffle()
    
    #-----------------------------------------------------------------------------#
    # Train                                                                       #
    #-----------------------------------------------------------------------------#
    start_epoch = 0   
    best_score = 0

    #-----------------------------------------------------------------------------#
    # Writer
    #-----------------------------------------------------------------------------#
    # Writer
    writer = SummaryWriter('world/' + config.model)
    LPN_flag = config.agg['agg_config']['LPN']
    

    for epoch in range(1, config.epochs+1):
        
        print("\n{}[Epoch: {}]{}".format(30*"-", epoch, 30*"-"))
        

        train_loss = trainer.train(config,
                           model,
                           dataloader=train_dataloader,
                           loss_function=loss_function,
                           optimizer=optimizer,
                           scheduler=scheduler,
                           scaler=scaler,
                           writer=writer)
        
        print("Epoch: {}, Train Loss = {:.3f}, Lr = {:.6f}".format(epoch,
                                                                   train_loss,
                                                                   optimizer.param_groups[0]['lr']))
        
        #------------------------------------------------------------Eval---------------------------------------------------------------------#
        result_list = []
        with open(config.val_index_txt,"r") as val_test:
            for line in val_test:
                eva_dataset_query = WorldDatasetEvalGroup(data_dir=config.dataset_root_dir,
                                      name=line.strip('\n'),
                                      mode='query',
                                      transforms=eval_transform)
    
                eval_dataloader_query = DataLoader(eva_dataset_query,
                                  batch_size=config.batch_size,
                                  num_workers=config.num_workers,
                                  shuffle=not config.custom_sampling,
                                  pin_memory=True)
                
                eva_dataset_db = WorldDatasetEvalGroup(data_dir=config.dataset_root_dir,
                                      name=line.strip('\n'),
                                      mode='DB',
                                      transforms=eval_transform)
    
                eval_dataloader_db = DataLoader(eva_dataset_db,
                                  batch_size=config.batch_size,
                                  num_workers=config.num_workers,
                                  shuffle=not config.custom_sampling,
                                  pin_memory=True)
                
                pos_gt = eval_dataloader_db.dataset.get_gt()
                result,_ , _ = eval.evaluate(config, model, eval_dataloader_query, eval_dataloader_db, pos_gt, mode='group', LPN=False)
                print(line.strip('\n'), 'top 1: ', round(result[0]*100,2),  'top 5: ', round(result[1]*100,2), 'top 10: ', round(result[2]*100,2))
                result_list.append(result)
                writer.add_scalar(line.strip('\n'), round(result[0]*100,2), epoch)
                
        
        result_array = np.array(result_list)
        average_result = np.mean(result_array, axis=0)
        print('Average', 'top 1: ', round(average_result[0]*100,2),  'top 5: ', round(average_result[1]*100,2), 'top 10: ', round(average_result[2]*100,2))
        writer.add_scalar('Average/top1', round(average_result[0]*100,2), epoch)
        writer.add_scalar('Average/top5', round(average_result[1]*100,2), epoch)
        
    #------------------------------------------------------------Save---------------------------------------------------------------------#
        if average_result[0] > best_score:

                best_score = average_result[0]

                if torch.cuda.device_count() > 1 and len(config.gpu_ids) > 1:
                    torch.save(model.module.state_dict(), '{}/weights_e{}_{:.4f}.pth'.format(model_path, epoch,  average_result[0]))
                else:
                    torch.save(model.state_dict(), '{}/weights_e{}_{:.4f}.pth'.format(model_path, epoch,  average_result[0]))
                

        if config.custom_sampling:
            train_dataloader.dataset.shuffle()
