import os
import shutil

#  @author : zhaojihuai z50010333

# This file can push all file from anywhere subdir into target rootdir.



json_list = []
image_dict = dict()
def search_leaf_file(func):
    def wrapper(path):
        lsdir = os.listdir(path)
        dirs = [i for i in lsdir if os.path.isdir(os.path.join(path, i))]
        files = [i for i in lsdir if os.path.isfile(os.path.join(path, i))]
        
        if files:
            for f in files:
                json_list.append(os.path.join(path, f)) if func(os.path.join(path, f)) else None
                image_dict[f.split('.')[0]+'.jpg'] = path if func(os.path.join(path, f)) else None
        
        if dirs:
            for d in dirs:
                wrapper(os.path.join(path, d)) 

    return wrapper

# def search_leaf_file(func):
#     def wrapper(path, save_path, *args):
#         lsdir = os.listdir(path)
#         dirs = [i for i in lsdir if os.path.isdir(os.path.join(path, i))]
#         files = [i for i in lsdir if os.path.isfile(os.path.join(path, i))]
        
#         if files:
#             for f in files:
#                 func(os.path.join(path, f), save_path)
#         if dirs:
#             for d in dirs:
#                 wrapper(os.path.join(path, d), save_path) 

#     return wrapper

@search_leaf_file    
def merge_file(path, save_path, img_format = '.jpg'):

    if 'json' in path:
        json_file_name = path.split('/')[-1]
        shutil.copy(path, os.path.join(save_path, json_file_name))
        img_name = json_file_name.split('.')[0] + img_format
        img_path = path.replace(json_file_name, img_name)
        shutil.copy(img_path, os.path.join(save_path, img_name))

@search_leaf_file    
def merge_json(path):
    
    return True if 'json' in path else False

    

if __name__=='__main__':
    path = '/home/jovyan/qrcode-single-detection/0718_detection'
    save_path = '/home/jovyan/aaaaaaa'
    os.mkdir(save_path) if not os.path.exists(save_path) else None
    merge_json(path)
    