import time
import torch
from tqdm import tqdm
from utils import setting
from torch.cuda.amp import autocast
import torch.nn.functional as F

def train(train_config, model, dataloader, loss_function, optimizer,scheduler=None, scaler=None, writer=None):

    # set model train mode
    model.train()
    
    losses = setting.AverageMeter()
    
    # wait before starting progress bar
    time.sleep(0.1)
    
    # Zero gradients for first step
    optimizer.zero_grad(set_to_none=True)
    
    step = 1
    
    if train_config.verbose:
        bar = tqdm(dataloader, total=len(dataloader))
    else:
        bar = dataloader
    
    # for loop over one epoch
    # 修改代码为带weight
    # for query,query_pt, reference,  reference_pt, ids, weight in bar:
    for query,query_pt, reference,  reference_pt, ids in bar:   
        if scaler:
            with autocast():
            
                # data (batches) to device   
                query = query
                reference = reference
                query_pt = query_pt
                reference_pt = reference_pt
            
                # Forward pass
                features1, _ = model(query, query_pt)
                features2, _ = model(reference, reference_pt)
               
                if torch.cuda.device_count() > 1 and len(train_config.gpu_ids) > 1: 
                    loss = loss_function(features1, features2, model.module.logit_scale.exp())
                    # loss = loss_function(features1, features2, model.module.logit_scale.exp(), weight)
                else:
                    # InfoNCE Loss
                    loss = loss_function(features1, features2, model.logit_scale.exp()) 
                    # loss = loss_function(features1, features2, model.logit_scale.exp(), weight)
                    # SupCon Loss
                    # feature = torch.cat((features1, features2), dim=0)
                    # labels = torch.cat((ids, ids), dim=0)
                    # loss = loss_function(feature, labels)

                losses.update(loss.item())
                
                  
            scaler.scale(loss).backward()
            
            # Gradient clipping 
            if train_config.clip_grad:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_value_(model.parameters(), train_config.clip_grad) 
            
            # Update model parameters (weights)
            scaler.step(optimizer)
            scaler.update()

            # Zero gradients for next step
            optimizer.zero_grad()
            
            # Scheduler
            if train_config.scheduler == "polynomial" or train_config.scheduler == "cosine" or train_config.scheduler ==  "constant":
                scheduler.step()
   
        else:
        
            # data (batches) to device   
            query = query.to(train_config.device)
            reference = reference.to(train_config.device)

            # Forward pass
            features1, features2 = model(query, reference)
            
            if torch.cuda.device_count() > 1 and len(train_config.gpu_ids) > 1: 
                # loss = loss_function(features1, features2, model.module.logit_scale.exp(), weight)
                loss = loss_function(features1, features2, model.module.logit_scale.exp())
            else:
                loss = loss_function(features1, features2, model.logit_scale.exp())
                # loss = loss_function(features1, features2, model.logit_scale.exp(), weight)
            losses.update(loss.item())
            
            # Calculate gradient using backward pass
            loss.backward()

            
            
            # Gradient clipping 
            if train_config.clip_grad:
                torch.nn.utils.clip_grad_value_(model.parameters(), train_config.clip_grad)                  
            
            # Update model parameters (weights)
            optimizer.step()
            # Zero gradients for next step
            optimizer.zero_grad()
            
            # Scheduler
            if train_config.scheduler == "polynomial" or train_config.scheduler == "cosine" or train_config.scheduler ==  "constant":
                scheduler.step()
        
        
        
        if train_config.verbose:
            
            monitor = {"loss": "{:.4f}".format(loss.item()),
                       "loss_avg": "{:.4f}".format(losses.avg),
                       "lr" : "{:.6f}".format(optimizer.param_groups[0]['lr'])}
            
            bar.set_postfix(ordered_dict=monitor)

            writer.add_scalar('Loss/train', loss.item(), step)
            writer.add_scalar('Loss/avg_loss', losses.avg, step)
            writer.add_scalar('lr', optimizer.param_groups[0]['lr'], step)

        
        step += 1

    if train_config.verbose:
        bar.close()

    return losses.avg


def train_backbone(train_config, model, dataloader, loss_function, optimizer, scheduler=None, scaler=None, writer=None, LPN=False):

   # set model train mode
    model.train()
    
    losses = setting.AverageMeter()
    
    # wait before starting progress bar
    time.sleep(0.1)
    
    # Zero gradients for first step
    optimizer.zero_grad(set_to_none=True)
    
    step = 1
    
    if train_config.verbose:
        bar = tqdm(dataloader, total=len(dataloader))
    else:
        bar = dataloader
    
    
    
    # for loop over one epoch
    for query, reference,  ids in bar:
        
        loss = 0.0
        query = query.to(train_config.device)
        reference = reference.to(train_config.device)
        
            # Forward pass
        features1 = model(query)
        features2 = model(reference)
        
        if LPN == False:
            if torch.cuda.device_count() > 1 and len(train_config.gpu_ids) > 1: 
                loss = loss_function(features1, features2, model.module.logit_scale.exp())
            else:
                loss = loss_function(features1, features2, model.logit_scale.exp()) 
        else:
            for index in range(len(features1)):
                feature1_one = features1[index]
                feature2_one = features2[index]
                if torch.cuda.device_count() > 1 and len(train_config.gpu_ids) > 1: 
                    temp_loss = loss_function(feature1_one, feature2_one, model.module.logit_scale.exp())
                else:
                    temp_loss = loss_function(feature1_one, feature2_one, model.logit_scale.exp()) 
                loss += temp_loss
        
        losses.update(loss.item())
                
                  
      

        # Zero gradients for next step
        optimizer.zero_grad()
        
        # Scheduler
        if train_config.scheduler == "polynomial" or train_config.scheduler == "cosine" or train_config.scheduler ==  "constant":
            scheduler.step()


        losses.update(loss.item())

        # Calculate gradient using backward pass
        loss.backward(retain_graph=True)
            
                 
            
        # Update model parameters (weights)
        optimizer.step()
        # Zero gradients for next step
        optimizer.zero_grad()
        
        # Scheduler
        if train_config.scheduler == "polynomial" or train_config.scheduler == "cosine" or train_config.scheduler ==  "constant":
            scheduler.step()
        
        
        
        if train_config.verbose:
            
            monitor = {"loss": "{:.4f}".format(loss.item()),
                       "loss_avg": "{:.4f}".format(losses.avg),
                       "lr" : "{:.6f}".format(optimizer.param_groups[0]['lr'])}
            
            bar.set_postfix(ordered_dict=monitor)
            writer.add_scalar('Loss/train', loss.item(), step)
            writer.add_scalar('Loss/avg_loss', losses.avg, step)
            writer.add_scalar('lr', optimizer.param_groups[0]['lr'], step)
        
        step += 1

    if train_config.verbose:
        bar.close()

    return losses.avg