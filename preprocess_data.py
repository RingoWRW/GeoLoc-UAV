import os
import json
from glob import glob

"""
根据输入的.txt，生成相应的train_query.txt, train_db.txt
"""

def generate_img_list(root ,txt,  save_path, mode):

    save_file_path_query = save_path + mode + '_query_all.txt'
    save_file_path_db = save_path + mode + '_db_all.txt'
    
    label = 0
     
     # 处理query图像
    with open(save_file_path_query, 'w') as f_query:
        with open(txt, 'r') as f:
            for line in f:
                one_root = os.path.join(root, line.strip('\n'))
                positive = json.load(open(one_root+'/positive.json'))
                semi_positive = json.load(open(one_root+'/semi_positive.json'))
                query_names = positive.keys()
                query_dirs = os.listdir(one_root+'/query')
                for query_name in query_names:
                    for query_dir in query_dirs:
                        # query路径
                        if query_dir[0] != 'h':
                            continue
                        one_query_path = line.strip('\n') + '/query/' + query_dir + '/'  + 'footage/'+ query_dir + '_' + query_name + '.jpeg'
                        f_query.write(one_query_path + ' ' + str(label) + ' ')
                        pos_dbs = positive[query_name]
                        try:
                            FLAG = True
                            semi_pos_dbs = semi_positive[query_name]
                        except:
                            FLAG = False
                        # pos GT 路径
                        for pos_db in pos_dbs:
                            temp = line.strip('\n') + '/DB/' + 'img/' + pos_db
                            f_query.write(temp + ' ')
                        # semi GT 路径
                        if FLAG:
                            for semi_pos_db in semi_pos_dbs:
                                temp = line.strip('\n') + '/DB/' + 'img/' + semi_pos_db
                                f_query.write(temp + ' ')
                        f_query.write('\n')
                    label += 1
    print('-------------------------finish-------------------------')

    # 处理DB图像
    with open(save_file_path_db, 'w') as f_db:
        with open(txt, 'r') as f:
            for line in f:
                one_root = os.path.join(root, line.strip('\n'))
                db_path = one_root + '/DB/' + 'img/'
                db_imgs = glob(db_path + '*.png')
                for db_img in db_imgs:
                    temp_list = db_img.split('/')[6:]
                    temp = ''
                    for i in temp_list:
                        temp += i
                        if not i.endswith('.png'):
                         temp += '/'
                    f_db.write(temp + '\n')
                  


root = "/media/Shen/Data/RingoData/WorldLoc/"
txt = "/media/Shen/Data/RingoData/WorldLoc/Index/train_all.txt"
save_path = "/media/Shen/Data/RingoData/WorldLoc/Index/"
mode = 'train'

generate_img_list(root, txt, save_path, mode)


# 验证代码
# import cv2
# txt = '/media/Shen/Data/RingoData/WorldLoc/Index/train_db.txt'
# with open(txt, 'r') as f:
#     for line in f:
#         line_list = line.split(' ')
#         query_img = line_list[0].strip('\n')
#         root = '/media/Shen/Data/RingoData/WorldLoc'
#         img_path = os.path.join(root, query_img)
#         img = cv2.imread(img_path)
#         print(img.shape)
#         cv2.imshow('img', img)
#         cv2.waitKey(0)