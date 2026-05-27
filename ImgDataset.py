import os
import re
import random
import numpy as np
import glob
import torch.utils.data
from PIL import Image,ImageFilter
import torch
from torchvision import transforms
from pathlib import Path


def read_names_to_list(txt_path):
    txt_path = Path(txt_path)
    if not txt_path.exists():
        raise FileNotFoundError(f"文件不存在: {txt_path}")

    with open(txt_path, "r", encoding="utf-8") as f:
        names = [line.strip() for line in f if line.strip()]  # 去除空行和换行符
    #
    # print(f"✅ 共读取 {len(names)} 个名称")
    return names

class MultiviewImgDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, scale_aug=False, rot_aug=False, test_mode=False, \
                 num_models=0, num_views=5, shuffle=True):

        
        txt_path = r"file_name.txt"
        
        self.classnames = read_names_to_list(txt_path)


       

        self.root_dir = root_dir
        self.scale_aug = scale_aug
        self.rot_aug = rot_aug
        self.test_mode = test_mode
        self.num_views = num_views


        self.filepaths = []
        self.textdescribe=[]


        for class_name in self.classnames:
            class_dir = os.path.join(root_dir, class_name)
            # entropy_dir = os.path.join(entropy_base_dir, class_name)

            model_ids = [name for name in os.listdir(class_dir) if os.path.isdir(os.path.join(class_dir, name))]
            if num_models > 0:
                model_ids = model_ids[:min(num_models, len(model_ids))]

            for model_id in model_ids:

                id_dir=os.path.join(class_dir,model_id)
                all_files = [
                    os.path.join(id_dir, f)
                    for f in os.listdir(id_dir)
                    if os.path.isfile(os.path.join(id_dir, f)) and f.endswith('.png')
                ]
                file_count = len(all_files)

                selected_files=[]

                if file_count == 5:

                    selected_files = all_files
 
                elif file_count > 5:

                    selected_files = random.sample(all_files, 5)



                if len(selected_files) == self.num_views:
                    self.filepaths.extend(selected_files)
                    self.textdescribe.extend(f.rsplit(".", 1)[0] + ".txt" for f in selected_files)

        assert len(self.filepaths) > 0

        if shuffle == True:
            # permute
            rand_idx = np.random.permutation(int(len(self.filepaths) / num_views))
            filepaths_new = []
            textdescribe_new=[]
            for i in range(len(rand_idx)):
                filepaths_new.extend(self.filepaths[rand_idx[i] * num_views:(rand_idx[i] + 1) * num_views])
                textdescribe_new.extend(self.textdescribe[rand_idx[i] * num_views:(rand_idx[i] + 1) * num_views])
            self.filepaths = filepaths_new
            self.textdescribe=textdescribe_new

        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

        
        if self.test_mode:
            self.transform = transforms.Compose([
                transforms.Resize((224,224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean,
                                     std=self.std)
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224,224)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean,
                                     std=self.std)
            ])





    def __len__(self):
        return int(len(self.filepaths) / self.num_views)

    def __getitem__(self, idx):
        path = self.filepaths[idx*self.num_views]

        class_name = path.split('/')[-3]
        class_id = self.classnames.index(class_name)

        imgs = []
        text_list = []
        for i in range(self.num_views):
            im = Image.open(self.filepaths[idx * self.num_views + i]).convert('RGB')
            if self.transform:
                im = self.transform(im)
            imgs.append(im)

            with open(self.textdescribe[idx * self.num_views + i], "r", encoding="utf-8") as f:
                text = f.read().strip()

                text_list.append(text)


        return (class_id, torch.stack(imgs),text_list)


class SingleImgDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, test_mode=False, num_models=0):
        """
        单视角读取，只从每个模型文件夹中选 1 张图片
        root_dir: train_3D 或 test_3D
        """
        txt_path = r"file_name_1000.txt"
        self.classnames = read_names_to_list(txt_path)

        self.root_dir = root_dir
        self.test_mode = test_mode
        self.filepaths = []  # 单张图片路径

        # 遍历所有类别
        for class_name in self.classnames:
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_dir):
                continue

            model_ids = [
                m for m in os.listdir(class_dir)
                if os.path.isdir(os.path.join(class_dir, m))
            ]

            # 限制每类模型数量
            if num_models > 0:
                model_ids = model_ids[:min(num_models, len(model_ids))]

            for model_id in model_ids:
                model_dir = os.path.join(class_dir, model_id)

                # 找到当前模型的所有图片
                all_imgs = [
                    os.path.join(model_dir, f)
                    for f in os.listdir(model_dir)
                    if f.endswith(".png")
                ]
                if len(all_imgs) == 0:
                    continue

                # 随机选 1 张图作为单视角输入
                selected = random.choice(all_imgs)
                self.filepaths.append(selected)

        assert len(self.filepaths) > 0, "没有找到任何单视角图片！"

        # ----------- 图像增广 -----------
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

        if self.test_mode:
            self.transform = transforms.Compose([
                transforms.Resize((518, 518)),
                # transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean, std=self.std)
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((518, 518)),
                # transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean, std=self.std)
            ])
        # --------------------------------

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        path = self.filepaths[idx]

        # 类别目录名
        class_name = path.split("/")[-3]
        class_id = self.classnames.index(class_name)

        # 读取图像
        img = Image.open(path).convert("RGB")
        img = self.transform(img)

        return class_id, img, path



