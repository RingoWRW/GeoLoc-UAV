import os
import cv2
import numpy as np
from torch.utils.data import Dataset
import copy
from tqdm import tqdm
import time
import random

from torch.utils.data import DataLoader
import torchvision.transforms as T

from utils.utils import TransformerCV

transform_config = {
     "sample_scale_begin": 0,
  "sample_scale_inter": 0.5, 
  "sample_scale_num": 5, 
  "sample_rotate_begin": -45,
  "sample_rotate_inter": 45, 
  "sample_rotate_num": 8,   
}

default_transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def get_data(txt):

    data = {}
    idx = 0
    with open(txt, 'r') as f:
        for line in f:
            line_list = line.split(' ')[:-1]
            data[idx] = line_list
            idx += 1

    return data

class WorldDatasetTrain(Dataset):
    def __init__(self, 
                 data_dir, 
                 query_txt,
                 transforms_query=default_transform,
                 transforms_db=default_transform,
                 shuffle_batch_size=64):
        super().__init__()

        self.pairs = []
        self.data = get_data(query_txt)

        for idx in self.data.items():
            query_img_path = os.path.join(data_dir, idx[1][0])
            label = eval(idx[1][1])
            db_image_path = os.path.join(data_dir, idx[1][2])
            self.pairs.append((label, query_img_path, db_image_path))
        
        self.transforms_query = transforms_query
        self.transforms_db = transforms_db
        self.shuffle_batch_size = shuffle_batch_size
        
        self.samples = copy.deepcopy(self.pairs)

        self.group_transformer = TransformerCV(transform_config)
        self.pts_step = 5
    
    def __getitem__(self, index):
        
        idx, query_img_path, db_img_path = self.samples[index]
        # query
        query_img = cv2.imread(query_img_path)
        query_img = cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB)
        # db
        db_img = cv2.imread(db_img_path)
        db_img = cv2.cvtColor(db_img, cv2.COLOR_BGR2RGB)
         # image transforms
        if self.transforms_query is not None:
            query_img = self.transforms_query(image=query_img)['image']
            
        if self.transforms_db is not None:
            db_img = self.transforms_db(image=db_img)['image']
        
        return query_img, db_img, idx
    
    

    def __len__(self):

        return len(self.samples)

    def shuffle(self, ):

        """
        generate unique class_id
        """
        print("\n Shuffle Dataset")

        pair_pool = copy.deepcopy(self.pairs)
        #shuffle
        random.shuffle(pair_pool)

        pairs_epoch = set()   
        label_batch = set()

        current_batch = []
        batches = []

         # progressbar
        pbar = tqdm()

        while True:
            pbar.update()
            if len(pair_pool) > 0:
                pair = pair_pool.pop(0)

                label, _, _ = pair

                if label not in label_batch and pair not in pairs_epoch:

                    label_batch.add(label)
                    current_batch.append(pair)
                    pairs_epoch.add(pair)

                    break_counter = 0
                
                else:
                    if pair not in pairs_epoch:
                        pair_pool.append(pair)
                    
                    break_counter += 1
                
                if break_counter >= 5000:
                        break
            
            else:
                break

            if len(current_batch) >= self.shuffle_batch_size:
                batches.extend(current_batch)
                label_batch = set()
                current_batch = []
        
        pbar.close()

        time.sleep(0.3)

        self.samples = batches

        print("Original Length: {} - Length after Shuffle: {}".format(len(self.pairs), len(self.samples))) 
        print("Break Counter:", break_counter)
        print("Pairs left out of last batch to avoid creating noise:", len(self.pairs) - len(self.samples))
        # print("First Element ID: {} - Last Element ID: {}".format(self.samples[0][0], self.samples[-1][0]))  


# 测试代码

# data_dir = "/media/guan/新加卷/EdgeBing/WorldLoc"
# query_txt = "/media/guan/新加卷/EdgeBing/WorldLoc/Index/train_query.txt"

# train_dataset = WorldDatasetTrain(data_dir, query_txt)

# train_dataloader = DataLoader(train_dataset,
#                                 batch_size=64,
#                                 num_workers=0,
#                                 shuffle=False,
#                                 pin_memory=True)


# train_dataloader.dataset.shuffle() 

# for  query, reference, idx in  tqdm(train_dataloader, total=len(train_dataloader)):

#     print(1)