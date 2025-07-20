import time
import torch
import numpy as np
from tqdm import tqdm
from torch.cuda.amp import autocast
import torch.nn.functional as F
import faiss
import faiss.contrib.torch_utils
import h5py
import os

def predict(train_config, model, dataloader):
    
    model.eval()
    
    # wait before starting progress bar
    # time.sleep(0.1)
    bar = tqdm(dataloader, total=len(dataloader))
    # if train_config.verbose:
    #     bar = tqdm(dataloader, total=len(dataloader))
    # else:
    #     bar = dataloader
        
    img_features_list = []
    # import time
    # torch.cuda.synchronize()
    # st = time.time()
    with torch.no_grad():
        
        for img, pt in bar:
        
            with autocast():
         
                img_feature, _ = model(img, pt)
                # print(f"Initial memory allocated: {torch.cuda.memory_allocated()} bytes")
            
            # save features in fp32 for sim calculation
            img_features_list.append(img_feature.to(torch.float32))
        # torch.cuda.synchronize()
        # et = time.time()
        print('---------------------------------time---------------------------------')
        # print('time cost: ', (et - st)/len(dataloader))
        # keep Features on GPU
        img_features = torch.cat(img_features_list, dim=0) 
        
    # if train_config.verbose:
    bar.close()
        
    return img_features

def predict_rerank(train_config, model, dataloader, name, mode):
    
    model.eval()
    
    # wait before starting progress bar
    time.sleep(0.1)
    
    if train_config.verbose:
        bar = tqdm(dataloader, total=len(dataloader))
    else:
        bar = dataloader
        
    img_features_list = []
    h5_name = str(name)+'_'+mode + '.h5'
    
    
    with torch.no_grad():
        
        for img, pt, img_path in bar:
        
            with autocast():
         
                img_feature, geat_list = model(img, pt)
                 # save features in fp32 for sim calculation
                img_features_list.append(img_feature.to(torch.float32))
                # average_geats = torch.mean(geat_list, dim=2)
                # average_geats = average_geats.reshape(geat_list.shape[1], geat_list.shape[3], geat_list.shape[4]).cpu()
                feature_geats = geat_list.squeeze(0).cpu()
                feature_geats = feature_geats[::60, :, :, :].reshape(-1, 24)
                
                # if os.path.exists(h5_name):
                #     pass
                with h5py.File(h5_name, 'a', libver='latest') as fd:
                    if img_path[0] in fd:
                        continue
                    grp = fd.create_group(img_path[0])
                    grp.create_dataset('global_feature', data=feature_geats.cpu())
        
        # keep Features on GPU
        img_features = torch.cat(img_features_list, dim=0) 
        print('---------------------------------save h5 file---------------------------------')
        
    if train_config.verbose:
        bar.close()
        
    return img_features

def predict_backbone(train_config, model, dataloader, LPN):
    
    model.eval()
    
    # wait before starting progress bar
    # time.sleep(0.1)
    bar = tqdm(dataloader, total=len(dataloader))
    
    # if train_config.verbose:
    #     bar = tqdm(dataloader, total=len(dataloader))
    # else:
    #     bar = dataloader
        
    img_features_list = []
    # import time
    # torch.cuda.synchronize()
    # st = time.time()
    with torch.no_grad():
        
        for img in bar:
        
            with autocast():
         
                # img_feature = model(img)
                # img_feature = model(img.to(train_config.device).half())
                img_feature = model(img.to(train_config["device"]).half())
                # img_feature = model(img.to(train_config["device"]))
            
            # save features in fp32 for sim calculation
            if LPN:
                img_feature_tensor = torch.stack(img_feature, dim=2).reshape(img_feature[0].shape[0], -1)
                img_features_list.append(img_feature_tensor.to(torch.float32))
            else:
                
                img_features_list.append(img_feature.to(torch.float32))
            
            
        # torch.cuda.synchronize()
        # et = time.time()
        # print('---------------------------------time---------------------------------')
        # print('time cost: ', (et - st)/len(dataloader))
        # keep Features on GPU
        img_features = torch.cat(img_features_list, dim=0) 
        
    # if train_config.verbose:
    bar.close()
        
    return img_features

def evaluate_reank(config,
                  model,
                  query_loader,
                  gallery_loader,
                  pos_gt,
                  ranks=[1, 5, 10],
                  name = None,
                  cleanup=True):
    # 需要保存下来group中的特征，故重新书写此代码
    
    
    print("Extract Features:")
    img_features_query  = predict_rerank(config, model, query_loader, name, 'query')
    img_features_gallery  = predict_rerank(config, model, gallery_loader, name, 'gallery')

    
    gl = img_features_gallery.cpu()
    ql = img_features_query.cpu()

    # -------------------------init------------------------------------------
    faiss_index = faiss.IndexFlatL2(gl.shape[1])
    # add references
    faiss_index.add(gl)

        # search for queries in the index
    _, predictions = faiss_index.search(ql, max(ranks))

    

    correct_at_rank = np.zeros(len(ranks))

    multi_num = ql.shape[0] / len(pos_gt)
    really_pos_gt = pos_gt * int(multi_num)

    for q_idx, pred in enumerate(predictions):
        for i, n in enumerate(ranks):
            if np.any(np.in1d(pred[:n], really_pos_gt[q_idx][1])):
                correct_at_rank[i] += 1

    correct_at_rank = correct_at_rank / len(predictions)
   
    
    return correct_at_rank, predictions,really_pos_gt


def evaluate(config,
                  model,
                  query_loader,
                  gallery_loader,
                  pos_gt,
                  mode,
                  LPN,
                  ranks=[1, 5, 10],
                  name = None,
                  cleanup=True):
    
    
    print("Extract Features:")
    
    if mode == 'group':
        
        img_features_query = predict(config, model, query_loader)
        img_features_gallery = predict(config, model, gallery_loader)
    elif mode == 'vanilia':
        
        img_features_query = predict_backbone(config, model, query_loader, LPN)
        img_features_gallery = predict_backbone(config, model, gallery_loader, LPN)
    
    gl = img_features_gallery.cpu()
    ql = img_features_query.cpu()
    # t-sne
    # import numpy as np
    # from sklearn.manifold import TSNE
    # from sklearn.preprocessing import StandardScaler
    # import matplotlib.pyplot as plt
    
    # ql_stand = StandardScaler().fit_transform(ql)
    # num = int(ql_stand.shape[0] / 76)
    # t_sne_save = config.dataset_root_dir + '/' + name + '/'
    # y =  list(range(0,10))
    # reap_y = np.array([item for item in y for _ in range(num)])

    # f_1 = ql_stand[::76, :]
    # f_2 = ql_stand[5::76, :]
    # f_3 = ql_stand[10::76, :]
    # f_4 = ql_stand[15::76, :]
    # f_5 = ql_stand[20::76, :]
    # f_6 = ql_stand[25::76, :]
    # f_7 = ql_stand[30::76, :]
    # f_8 = ql_stand[35::76, :]
    # f_9 = ql_stand[40::76, :]
    # f_10 = ql_stand[45::76, :]

    # x_stand = np.concatenate((f_1, f_2, f_3, f_4, f_5, f_6, f_7, f_8, f_9,f_10), axis=0)
    
    # tsne = TSNE(n_components=2, perplexity=num-1, n_iter=5000, n_jobs=-1)
    # X_tsne = tsne.fit_transform(x_stand)
    # plt.figure(figsize=(8, 8))
    # # 归一化颜色值
    # norm = plt.Normalize(reap_y.min(), reap_y.max())
    # # 选择不同的颜色映射
    # cmap = plt.get_cmap('plasma')

    # # 转换颜色值到[0, 1]区间内
    # colors = cmap(norm(reap_y))
    # scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1],  c=colors, alpha=0.7)
    # plt.colorbar(scatter)

    # plt.savefig(t_sne_save + 't_sne_' + 'dinov2'+ '.png')


    # -------------------------init------------------------------------------
    faiss_index = faiss.IndexFlatL2(gl.shape[1])
    # add references
    faiss_index.add(gl)

        # search for queries in the index
    _, predictions = faiss_index.search(ql, max(ranks))

    

    correct_at_rank = np.zeros(len(ranks))

    multi_num = ql.shape[0] / len(pos_gt)
    really_pos_gt = pos_gt * int(multi_num)

    for q_idx, pred in enumerate(predictions):
        for i, n in enumerate(ranks):
            # if np.any(np.in1d(pred[:n], really_pos_gt[q_idx][1][:ranks[i]])): # test_40
            if np.any(np.in1d(pred[:n], really_pos_gt[q_idx][1])):
                correct_at_rank[i] += 1
          
    # 测试是问题，设置一个train小样本，快速迭代
              

    correct_at_rank = correct_at_rank / len(predictions)
   
    
    return correct_at_rank, predictions,really_pos_gt

def evaluate_other(config,
                  model,
                  query_loader,
                  gallery_loader,
                  pos_gt,
                  ranks=[1, 5, 10],
                  name = None,
                  cleanup=True,
                  LPN=False):
    
    
    print("Extract Features:")
    # img_features_query = predict(config, model, query_loader)
    # img_features_gallery = predict(config, model, gallery_loader)
    img_features_query = predict_backbone(config, model, query_loader)
    img_features_gallery = predict_backbone(config, model, gallery_loader)
    
    gl = img_features_gallery.cpu()
    ql = img_features_query.cpu()

    # t-sne
    # import numpy as np
    # from sklearn.manifold import TSNE
    # from sklearn.preprocessing import StandardScaler
    # import matplotlib.pyplot as plt
    
    # ql_stand = StandardScaler().fit_transform(ql)
    # num = int(ql_stand.shape[0] / 76)
    # t_sne_save = config.dataset_root_dir + '/' + name + '/'
    # y =  list(range(0,10))
    # reap_y = np.array([item for item in y for _ in range(num)])

    # f_1 = ql_stand[::76, :]
    # f_2 = ql_stand[5::76, :]
    # f_3 = ql_stand[10::76, :]
    # f_4 = ql_stand[15::76, :]
    # f_5 = ql_stand[20::76, :]
    # f_6 = ql_stand[25::76, :]
    # f_7 = ql_stand[30::76, :]
    # f_8 = ql_stand[35::76, :]
    # f_9 = ql_stand[40::76, :]
    # f_10 = ql_stand[45::76, :]

    # x_stand = np.concatenate((f_1, f_2, f_3, f_4, f_5, f_6, f_7, f_8, f_9,f_10), axis=0)
    
    # tsne = TSNE(n_components=2, perplexity=num-1, n_iter=5000, n_jobs=-1)
    # X_tsne = tsne.fit_transform(x_stand)
    # plt.figure(figsize=(8, 8))
    # # 归一化颜色值
    # norm = plt.Normalize(reap_y.min(), reap_y.max())
    # # 选择不同的颜色映射
    # cmap = plt.get_cmap('plasma')

    # # 转换颜色值到[0, 1]区间内
    # colors = cmap(norm(reap_y))
    # scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1],  c=colors, alpha=0.7)
    # plt.colorbar(scatter)

    # plt.savefig(t_sne_save + 't_sne_' + 'dinov2'+ '.png')


    # -------------------------init------------------------------------------
    faiss_index = faiss.IndexFlatL2(gl.shape[1])
    # add references
    faiss_index.add(gl)

        # search for queries in the index
    _, predictions = faiss_index.search(ql, max(ranks))

    

    correct_at_rank = np.zeros(len(ranks))

  
    really_pos_gt = pos_gt 

    for q_idx, pred in enumerate(predictions):
        for i, n in enumerate(ranks):
            if np.any(np.in1d(pred[:n], really_pos_gt[q_idx][1])):
                correct_at_rank[i] += 1

              

    correct_at_rank = correct_at_rank / len(predictions)
   
    
    return correct_at_rank, predictions,really_pos_gt

   

    

    
