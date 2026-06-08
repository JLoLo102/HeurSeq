import torch
import torch.optim as optim
import torch.nn as nn
import os, shutil, json
import argparse
import time
import numpy as np
import random

from tools.trainer import ModelNetTrainer

from tools.ImgDataset import MultiviewImgDataset
from tools.ImgDataset_text import MultiviewImgTextDataset
from tools.utils import record_times

from model import HSL-Pat_S, HSL-Pat_M
os.environ["TOKENIZERS_PARALLELISM"] = "false"

parser = argparse.ArgumentParser()
parser.add_argument("-name", "--name", type=str, help="Name of the experiment", default="HSL-Pat")
parser.add_argument("-bs", "--batchSize", type=int, help="Batch size for the second stage", default=32)
parser.add_argument("-num_models", type=int, help="number of models per class", default=1000)
parser.add_argument("-lr", type=float, help="learning rate", default=1e-4)
parser.add_argument("-weight_decay", type=float, help="weight decay", default=0.0001)
parser.add_argument("-no_pretraining", dest='no_pretraining', action='store_true')



parser.add_argument("-cnn_name", "--cnn_name", type=str, help="cnn model name", default="swin_t")

parser.add_argument("-num_views", type=int, help="number of views", default=5)

parser.add_argument("-train_path", type=str, default="/selected_527/new_train_3D")
parser.add_argument("-val_path", type=str, default="/selected_527/new_test_3D")


parser.set_defaults(train=False)

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def create_folder(log_dir):
    # make summary folder
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    else:
        print('WARNING: summary folder already exists!! It will be overwritten!!')
        shutil.rmtree(log_dir)
        os.mkdir(log_dir)

if __name__ == '__main__':
    set_seed(42)  

    args = parser.parse_args()
    pretraining = not args.no_pretraining

    log_dir = args.name
    create_folder(args.name)
    config_f = open(os.path.join(log_dir, 'config.json'), 'w')
    json.dump(vars(args), config_f)
    config_f.close()

 

    cnet = HSL-Pat_S(args.name, nclasses=527, pretraining=pretraining, cnn_name=args.cnn_name)
    

    print('======start Stage======')
    log_dir = args.name                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               
    create_folder(log_dir)


    cnet_2 = HSL-Pat_M( args.name,cnet, nclasses=527, cnn_name=args.cnn_name, num_views=args.num_views)
    

    # del cnet
    cnet_2 = torch.nn.DataParallel(cnet_2, device_ids=[0])
    optimizer = optim.Adam(cnet_2.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999))

    train_dataset = MultiviewImgDataset(args.train_path, scale_aug=False, rot_aug=False, num_models=args.num_models, num_views=args.num_views,)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batchSize, shuffle=True, num_workers=16)
    val_dataset = MultiviewImgDataset(args.val_path, scale_aug=False, rot_aug=False, num_models=args.num_models,test_mode=True,num_views=args.num_views)
    
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batchSize, shuffle=False, num_workers=16)


    print('num_train_files: '+str(len(train_dataset.filepaths)))
    print('num_val_files: '+str(len(val_dataset.filepaths)))

   
    trainer = ModelNetTrainer(cnet_2, train_loader, val_loader, optimizer, nn.CrossEntropyLoss(), 'OVPT',log_dir, num_views=args.num_views)

    tic2 = time.time()
    trainer.train(n_epochs=10)
    toc2 = time.time()
    print('The training time of second stage:%d m' % ((toc2-tic2)))
    # record_times((toc1-tic1), (toc2-tic2), 'records.txt')
    record_times( (toc2 - tic2), 'records.txt')



