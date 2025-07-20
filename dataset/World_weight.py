import os
import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset
import copy
from tqdm import tqdm
import time
import random
import glob
import json
import pandas as pd

from torch.utils.data import DataLoader
import torchvision.transforms as T


from utils.utils import TransformerCV

transform_config = {
     "sample_scale_begin": 0,
  "sample_scale_inter": 0.5, 
  "sample_scale_num": 3, 
  "sample_rotate_begin": 0,
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
            # line_list = line.split(' ')[:-1]
            line_list = line.split(' ')
            data[idx] = line_list
            idx += 1

    return data

class WorldDatasetTrainGroup(Dataset):
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
            query_img_path = os.path.join(data_dir, idx[1][3])
            weight = eval(idx[1][0])
            label = eval(idx[1][1])
            db_image_path = os.path.join(data_dir, idx[1][4][:-1])
            self.pairs.append((label, weight, query_img_path, db_image_path))
        
        self.transforms_query = transforms_query
        self.transforms_db = transforms_db
        self.shuffle_batch_size = shuffle_batch_size
        
        self.samples = copy.deepcopy(self.pairs)

        self.group_transformer = TransformerCV(transform_config)
        self.pts_step = 5
    
    def __getitem__(self, index):
        
        idx, weight, query_img_path, db_img_path = self.samples[index]
        # query
        query_img = self.image_loader(query_img_path)
        # db
        db_img = self.image_loader(db_img_path)
         # image transforms
        if self.transforms_query is not None:
            query_img = self.transforms_query(query_img)
            
        if self.transforms_db is not None:
            db_img = self.transforms_db(db_img)
        
        # return query_img, db_img, idx
        # group
        query_img *= 255
        query_img, query_pt = self.transformImg(query_img)

        db_img *= 255
        db_img, db_pt = self.transformImg(db_img)
        
        return query_img, query_pt, db_img, db_pt, idx, weight
    
    def transformImg(self, img):
        xs, ys = np.meshgrid(np.arange(self.pts_step,img.size()[1]-self.pts_step,self.pts_step), np.arange(self.pts_step,img.size()[2]-self.pts_step,self.pts_step))
        xs=xs.reshape(-1,1)
        ys = ys.reshape(-1,1)
        pts = np.hstack((xs,ys))
        img = img.permute(1,2,0).detach().numpy()
        transformed_imgs=self.group_transformer.transform(img,pts)
        data_img, data_pt = self.group_transformer.postprocess_transformed_imgs(transformed_imgs)
        return data_img, data_pt
    
    @staticmethod
    def image_loader(path):
        try:
            return Image.open(path)
            # return imread(path)
        except UnidentifiedImageError:
            print(f'Image {path} could not be loaded')
            return Image.new('RGB', (224, 224))

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

                label, _, _, _ = pair

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

class WorldDatasetTrainVanilia(Dataset):
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

    
    def __getitem__(self, index):
        
        idx, query_img_path, db_img_path = self.samples[index]
        # query
        query_img = self.image_loader(query_img_path)
        # db
        db_img = self.image_loader(db_img_path)
         # image transforms
        if self.transforms_query is not None:
            query_img = self.transforms_query(query_img)
            
        if self.transforms_db is not None:
            db_img = self.transforms_db(db_img)
        
        return query_img, db_img, idx
    
    @staticmethod
    def image_loader(path):
        try:
            return Image.open(path)
            # return imread(path)
        except UnidentifiedImageError:
            print(f'Image {path} could not be loaded')
            return Image.new('RGB', (224, 224))

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

class WorldDatasetEvalGroup(Dataset):
    def __init__(self, 
                data_dir, 
                 name,
                 mode,
                 transforms=default_transform
                 ):
        super().__init__()

        self.transforms = transforms
      
        self.group_transformer = TransformerCV(transform_config)
        self.pts_step = 5

        self.data_dir = data_dir
        self.name = name 

        pos_json_path = os.path.join(self.data_dir, self.name,  'positive.json')
        positive = json.load(open(pos_json_path))

        self.samples = []
        if mode == 'query':
            height_list = ["height100_rot0", "height100_rot45", "height100_rot90", "height100_rot135", "height100_rot180", "height100_rot225", "height100_rot270", "height100_rot315", 
                           "height125_rot0", "height125_rot45", "height125_rot90", "height125_rot135", "height125_rot180", "height125_rot225", "height125_rot270", "height125_rot315",
                            "height150_rot0", "height150_rot45", "height150_rot90", "height150_rot135", "height150_rot180", "height150_rot225", "height150_rot270", "height150_rot315"]
            for i in height_list:
                if os.path.exists(os.path.join(data_dir, name,'query',  i, 'footage')):
                    temp_path = os.path.join(data_dir, name,'query', i, 'footage')
                    temp = sorted(glob.glob(f'{temp_path}/{"*.jpeg"}'))
                    if len(temp) != len(positive.keys()):
                        filter_temp = [image for image in temp if image.split('/')[-1].split('.')[0].split('_')[-1] in positive.keys()]
                        self.samples.extend(filter_temp)
                    else:
                        self.samples.extend(temp)
        
        if mode == 'DB':
            temp_path = os.path.join(data_dir, name, 'DB', 'img')
            temp = sorted(glob.glob(f'{temp_path}/{"*.png"}'))
            self.samples.extend(temp)

        


    def __getitem__(self, index):
        
        img_path = self.samples[index]
        # query
        img = self.image_loader(img_path)
        
        if self.transforms is not None:
            img = self.transforms(img)

        img *= 255
        img, pt = self.transformImg(img)
        
        return img, pt
    
    def transformImg(self, img):

        xs, ys = np.meshgrid(np.arange(self.pts_step,img.size()[1]-self.pts_step,self.pts_step), np.arange(self.pts_step,img.size()[2]-self.pts_step,self.pts_step))
        xs=xs.reshape(-1,1)
        ys = ys.reshape(-1,1)
        pts = np.hstack((xs,ys))
        img = img.permute(1,2,0).detach().numpy()
        transformed_imgs=self.group_transformer.transform(img,pts)
        data_img, data_pt = self.group_transformer.postprocess_transformed_imgs(transformed_imgs)
        return data_img, data_pt
    
    def get_gt(self,):

      
        pos_json_path = os.path.join(self.data_dir, self.name,  'positive.json')
        semi_pos_json_path = os.path.join(self.data_dir, self.name,  'semi_positive.json')
        positive = json.load(open(pos_json_path))
        semi_positive = json.load(open(semi_pos_json_path))

        pos_gt = []
        for key in positive.keys():
            value = positive[key]
            
            temp_index = []
            # pos
            for one_value in value:
                temp_path_dir = os.path.join(self.data_dir, self.name, 'DB', 'img')
                temp_path = temp_path_dir + '/' + one_value
                one_index = self.samples.index(temp_path)
                temp_index.append(one_index)
            # semi-pos
            try:
                semi_value = semi_positive[key]
                for one_value in semi_value:
                    temp_path_dir = os.path.join(self.data_dir, self.name, 'DB', 'img')
                    temp_path = temp_path_dir + '/' + one_value
                    one_index = self.samples.index(temp_path)
                    temp_index.append(one_index)
            except:
                pos_gt.append([key, temp_index])
                continue

            pos_gt.append([key, temp_index])
        
        return pos_gt


    def getitem(self, index):

        return self.samples[index]


    
    @staticmethod
    def image_loader(path):
        try:
            return Image.open(path)
            # return imread(path)
        except UnidentifiedImageError:
            print(f'Image {path} could not be loaded')
            return Image.new('RGB', (224, 224))

    def __len__(self):

        return len(self.samples)

class WorldDatasetEvalVanilia(Dataset):
    def __init__(self, 
                data_dir, 
                 name,
                 mode,
                 transforms=default_transform
                 ):
        super().__init__()

        self.transforms = transforms

        self.data_dir = data_dir
        self.name = name 

        pos_json_path = os.path.join(self.data_dir, self.name,  'positive.json')
        positive = json.load(open(pos_json_path))

        self.samples = []
        if mode == 'query':
            height_list = ["height100_rot0", "height100_rot45", "height100_rot90", "height100_rot135", "height100_rot180", "height100_rot225", "height100_rot270", "height100_rot315", 
                           "height125_rot0", "height125_rot45", "height125_rot90", "height125_rot135", "height125_rot180", "height125_rot225", "height125_rot270", "height125_rot315",
                            "height150_rot0", "height150_rot45", "height150_rot90", "height150_rot135", "height150_rot180", "height150_rot225", "height150_rot270", "height150_rot315"]
            for i in height_list:
                if os.path.exists(os.path.join(data_dir, name,'query',  i, 'footage')):
                    temp_path = os.path.join(data_dir, name,'query', i, 'footage')
                    temp = sorted(glob.glob(f'{temp_path}/{"*.jpeg"}'))
                    if len(temp) != len(positive.keys()):
                        filter_temp = [image for image in temp if image.split('/')[-1].split('.')[0].split('_')[-1] in positive.keys()]
                        self.samples.extend(filter_temp)
                    else:
                        self.samples.extend(temp)
        
        if mode == 'DB':
            temp_path = os.path.join(data_dir, name, 'DB', 'img')
            temp = sorted(glob.glob(f'{temp_path}/{"*.png"}'))
            self.samples.extend(temp)

        


    def __getitem__(self, index):
        
        img_path = self.samples[index]
        # query
        img = self.image_loader(img_path)
        
        if self.transforms is not None:
            img = self.transforms(img)

        return img
        
    
    def get_gt(self,):

      
        pos_json_path = os.path.join(self.data_dir, self.name,  'positive.json')
        semi_pos_json_path = os.path.join(self.data_dir, self.name,  'semi_positive.json')
        positive = json.load(open(pos_json_path))
        semi_positive = json.load(open(semi_pos_json_path))

        pos_gt = []
        for key in positive.keys():
            value = positive[key]
            
            temp_index = []
            # pos
            for one_value in value:
                temp_path_dir = os.path.join(self.data_dir, self.name, 'DB', 'img')
                temp_path = temp_path_dir + '/' + one_value
                one_index = self.samples.index(temp_path)
                temp_index.append(one_index)
            try:
                semi_value = semi_positive[key]
            # semi-pos
                for one_value in semi_value:
                    temp_path_dir = os.path.join(self.data_dir, self.name, 'DB', 'img')
                    temp_path = temp_path_dir + '/' + one_value
                    one_index = self.samples.index(temp_path)
                    temp_index.append(one_index)
            except:
                pos_gt.append([key, temp_index])
                continue

            pos_gt.append([key, temp_index])
        
        return pos_gt


    def getitem(self, index):

        return self.samples[index]


    
    @staticmethod
    def image_loader(path):
        try:
            return Image.open(path)
            # return imread(path)
        except UnidentifiedImageError:
            print(f'Image {path} could not be loaded')
            return Image.new('RGB', (224, 224))

    def __len__(self):

        return len(self.samples)

class AerialDatasetEvalVanilia(Dataset):
    def __init__(self, 
                data_dir, 
                 mode,
                 transforms=default_transform
                 ):
        super().__init__()

        self.samples = []
        if mode == 'query':
            temp_path = os.path.join(data_dir,  'query_images')
            temp = sorted(glob.glob(f'{temp_path}/{"*.png"}'))
            self.samples.extend(temp)
        
        if mode == 'DB':
            temp_path = os.path.join(data_dir,  'reference_images')
            temp = sorted(glob.glob(f'{temp_path}/{"*.png"}'))
            self.samples.extend(temp)

        self.transforms = transforms
        self.data_dir = data_dir


    def __getitem__(self, index):
        
        img_path = self.samples[index]
        # query
        img = self.image_loader(img_path)
        
        if self.transforms is not None:
            img = self.transforms(img)

        return img
        
    
    def get_gt(self,):

        columns_to_use_by_index = [1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20,]
        pos_cvs_path = os.path.join(self.data_dir, 'gt_matches.csv')
        df  = pd.read_csv(pos_cvs_path, usecols=columns_to_use_by_index)

        pos_gt = []
        for i in range(len(df)):
            for j in range(df.shape[1]):
                if j == 0:
                    key = df.iloc[i, j]
                    temp_index = []
                else:
                    value = df.iloc[i, j]
                    temp_index.append(value)
        
            pos_gt.append([key, temp_index])
        return pos_gt
    
    def get_gt_npy(self,):

        data_path = os.path.join(self.data_dir, 'vpair_gt.npy')
        data = np.load(data_path, allow_pickle=True)
        pos_gt = []
        for i in range(data.shape[0]):
            key = data[i, 0]
            temp_index = []
            temp_value = data[i, 1]
            for j in temp_value:
                temp_index.append(j)
        
            pos_gt.append([key, temp_index])
        return pos_gt


    def getitem(self, index):

        return self.samples[index]


    
    @staticmethod
    def image_loader(path):
        try:
            img = Image.open(path)
            rotated_image = img.rotate(270)
            return rotated_image
        # Image.open(path)
            # return imread(path)
        except UnidentifiedImageError:
            print(f'Image {path} could not be loaded')
            return Image.new('RGB', (224, 224))

    def __len__(self):

        return len(self.samples)

class AerialDatasetEvalGroup(Dataset):
    def __init__(self, 
                data_dir, 
                 mode,
                 transforms=default_transform
                 ):
        super().__init__()

        self.samples = []
        if mode == 'query':
            temp_path = os.path.join(data_dir,  'query_images')
            temp = sorted(glob.glob(f'{temp_path}/{"*.png"}'))
            self.samples.extend(temp)
        
        if mode == 'DB':
            temp_path = os.path.join(data_dir,  'reference_images')
            temp = sorted(glob.glob(f'{temp_path}/{"*.png"}'))
            self.samples.extend(temp)

        self.transforms = transforms
      
        self.group_transformer = TransformerCV(transform_config)
        self.pts_step = 5

        self.data_dir = data_dir


    def __getitem__(self, index):
        
        img_path = self.samples[index]
        # query
        img = self.image_loader(img_path)
        
        if self.transforms is not None:
            img = self.transforms(img)
        
        # group
        img *= 255
        img, pt = self.transformImg(img)
        
        return img, pt
    
    def transformImg(self, img):

        xs, ys = np.meshgrid(np.arange(self.pts_step,img.size()[1]-self.pts_step,self.pts_step), np.arange(self.pts_step,img.size()[2]-self.pts_step,self.pts_step))
        xs=xs.reshape(-1,1)
        ys = ys.reshape(-1,1)
        pts = np.hstack((xs,ys))
        img = img.permute(1,2,0).detach().numpy()
        transformed_imgs=self.group_transformer.transform(img,pts)
        data_img, data_pt = self.group_transformer.postprocess_transformed_imgs(transformed_imgs)
        return data_img, data_pt
    
    def get_gt(self,):

        columns_to_use_by_index = [1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20,]
        pos_cvs_path = os.path.join(self.data_dir, 'gt_matches.csv')
        df  = pd.read_csv(pos_cvs_path, usecols=columns_to_use_by_index)

        pos_gt = []
        for i in range(len(df)):
            for j in range(df.shape[1]):
                if j == 0:
                    key = df.iloc[i, j]
                    temp_index = []
                else:
                    value = df.iloc[i, j]
                    temp_index.append(value)
        
            pos_gt.append([key, temp_index])
        return pos_gt
    
    def get_gt_npy(self,):

        data_path = os.path.join(self.data_dir, 'vpair_gt.npy')
        data = np.load(data_path, allow_pickle=True)
        pos_gt = []
        for i in range(data.shape[0]):
            key = data[i, 0]
            temp_index = []
            temp_value = data[i, 1]
            for j in temp_value:
                temp_index.append(j)
        
            pos_gt.append([key, temp_index])
        return pos_gt


    def getitem(self, index):

        return self.samples[index]


    
    @staticmethod
    def image_loader(path):
        try:
            return Image.open(path)
            # return imread(path)
        except UnidentifiedImageError:
            print(f'Image {path} could not be loaded')
            return Image.new('RGB', (224, 224))

    def __len__(self):

        return len(self.samples)
    
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

# for  query, query_pt, reference, reference_pt, idx in  tqdm(train_dataloader, total=len(train_dataloader)):

#     print(1)