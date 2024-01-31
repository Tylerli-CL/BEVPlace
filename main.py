from __future__ import print_function
import argparse
from os.path import join, isfile
from os import environ
import os
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import faiss                                # Faiss is a library for efficient similarity search and clustering of dense vectors.

from network.bevplace import BEVPlace
from network.utils import to_cuda

from tqdm import tqdm

# tqdm是一个用于在终端中显示进度条的库，它通常用于循环迭代、文件读写等长时间运行的任务，以向用户展示任务的进展。

os.environ["CUDA_VISIBLE_DEVICES"] = "0"    # os.environ 是一个 Python 字典，它包含了所有的环境变量。

parser = argparse.ArgumentParser(description='BEVPlace')

parser.add_argument('--test_batch_size', type=int, default=8, help='Batch size for testing')
parser.add_argument('--nGPU', type=int, default=2, help='number of GPU to use.')
parser.add_argument('--nocuda', action='store_true', help='Dont use cuda')
parser.add_argument('--threads', type=int, default=40, help='Number of threads for each data loader to use')
parser.add_argument('--resume', type=str, default='checkpoints/checkpoint_paper_kitti.pth.tar', help='Path to load checkpoint from, for resuming training or testing.')


def evaluate(eval_set, model):
    test_data_loader = DataLoader(dataset=eval_set, 
                                  num_workers=opt.threads, 
                                  batch_size=opt.test_batch_size, 
                                  shuffle=False, 
                                  pin_memory=cuda)

    model.eval()        # Sets the module in evaluation mode


    global_features = []
    with torch.no_grad():
        print('====> Extracting Features')
        with tqdm(total=len(test_data_loader)) as t:
            for iteration, (input, indices) in enumerate(test_data_loader, 1):
                if cuda:
                    input = to_cuda(input)
                
                # Output of the model
                batch_feature = model(input)
                global_features.append(batch_feature.detach().cpu().numpy())
                t.update(1)


    # Processing global features
    global_features = np.vstack(global_features)


    query_feat = global_features[eval_set.num_db:].astype('float32')
    db_feat = global_features[:eval_set.num_db].astype('float32')

    # print('====> Building faiss index')
    faiss_index = faiss.IndexFlatL2(query_feat.shape[1])
    faiss_index.add(db_feat)

    # print('====> Calculating recall @ N')
    n_values = [1,5, 10, 20]

    _, predictions = faiss_index.search(query_feat, max(n_values)) 

    gt = eval_set.getPositives() 

    correct_at_n = np.zeros(len(n_values))
    whole_test_size = 0

    for qIx, pred in enumerate(predictions):
        if len(gt[qIx]) ==0 : 
            continue
        whole_test_size+=1
        for i,n in enumerate(n_values):
            if np.any(np.in1d(pred[:n], gt[qIx])):
                correct_at_n[i:] += 1
                break
    recall_at_n = correct_at_n / whole_test_size
    # print("tp+fn=%d"%(whole_test_size))
    recalls = {} 
    for i,n in enumerate(n_values):
        recalls[n] = recall_at_n[i]
    #     print("====> Recall@{}: {:.4f}".format(n, recall_at_n[i]))

    return recalls

import dataset as dataset

from network import netvlad



if __name__ == "__main__":
    
    opt = parser.parse_args()

    cuda = not opt.nocuda
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run with --nocuda")

    device = torch.device("cuda" if cuda else "cpu")
     
    print('===> Building model')
    model = BEVPlace()

    resume_ckpt = opt.resume    # 
    print("=> loading checkpoint '{}'".format(resume_ckpt))
    checkpoint = torch.load(resume_ckpt, map_location=lambda storage, loc: storage)

    model.load_state_dict(checkpoint['state_dict'], strict=False)       # load the model

    model = model.to(device)
    print("=> loaded checkpoint '{}' (epoch {})"
            .format(resume_ckpt, checkpoint['epoch']))


    if cuda:
        model = nn.DataParallel(model)
        # model = model.to(device)

    data_path = './data/KITTIRot/'
    recall_seq = {"00":0, "02":0, "05":0, "06":0}

    for seq in list(recall_seq.keys()):
        print('===> Processing KITTI Seq. %s'%(seq))
        eval_set = dataset.KITTIDataset(data_path, seq)

        recalls = evaluate(eval_set, model)
        recall_seq[seq] = recalls[1]
    
    print("===> Recalls@1", recall_seq)