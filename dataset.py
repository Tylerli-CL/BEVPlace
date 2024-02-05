import os
import numpy as np
from PIL import Image

import torch
import torchvision.transforms as transforms
import torch.utils.data as data

from sklearn.neighbors import NearestNeighbors
from network.utils import TransformerCV

from network.groupnet import group_config

def input_transform():

    return transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                     std=[0.229, 0.224, 0.225]), ])
#  torchvision.transforms
# transforms.Compose([...]): transforms.Compose 是一个将多个数据转换操作组合在一起的类。在这里，我们使用 transforms.Compose 创建了一个转换操作的列表，其中包含了两个操作：
    # transforms.ToTensor(): 这个操作将图像数据转换为PyTorch张量（tensor）。深度学习模型通常需要输入张量作为数据。
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]): 这个操作对图像进行归一化处理，减去均值并除以标准差。这是一个常见的数据预处理步骤，有助于提高模型的训练稳定性和性能。
    # this transform will normalize each channel of the input torch.*Tensor i.e., output[channel] = (input[channel] - mean[channel]) / std[channel]
    

class KITTIDataset(data.Dataset):
    def __init__(self, data_path, sequence):
        super().__init__()

        self.positives = None
        self.distances = None       # Geometry Distance in paper ???

        #protocol setting, Dictionary
        db_frames = {'00': range(0,3000), '02': range(0,3400), '05': range(0,1000), '06': range(0,600)}                     # Database frames
        query_frames = {'00': range(3200, 4541), '02': range(3600, 4661), '05': range(1200,2751), '06': range(800,1101)}    # Query frames
        
        self.pos_threshold = 5   #ground truth threshold
        
        #preprocessor
        self.input_transform = input_transform()
        self.transformer = TransformerCV(group_config)
        self.pts_step = 5

        # root pathes
        # bev_path = data_path + '/' + sequence + '/' + '/imgs/'           # BEV-Image path
        # lidar_path = data_path + '/' + sequence + '/' + '/velodyne/'     # Lidar path

        bev_path = os.path.join(data_path, sequence, 'imgs/')                # BEV-Image path
        lidar_path = os.path.join(data_path, sequence, 'velodyne/')          # Lidar path

        # geometry positions 
        # poses = np.loadtxt(data_path + '/' + sequence + '/pose.txt')
        poses = np.loadtxt(os.path.join(data_path, sequence, 'pose.txt'))

        positions = np.hstack([poses[:,3].reshape(-1,1),
                               poses[:,7].reshape(-1,1)])                   # horizontal stack

        # 这两个后面要用来进行knn。但这个位置应该是提前就计算好了的
        self.db_positions = positions[db_frames[sequence], :]
        self.query_positions = positions[query_frames[sequence], :]

        self.num_db = len(db_frames[sequence])

        #image pathes
        images = os.listdir(bev_path)
        images.sort()

        # load images
        self.images = []
        for idx in db_frames[sequence]:
            self.images.append(bev_path + images[idx])

        for idx in query_frames[sequence]:
            self.images.append(bev_path + images[idx])     



    def transformImg(self, img):
        xs, ys = np.meshgrid(np.arange(self.pts_step, img.size()[1] - self.pts_step, self.pts_step),          # self.pts_step = 5
                             np.arange(self.pts_step, img.size()[2] - self.pts_step, self.pts_step))          # self.pts_step = 5

        xs = xs.reshape(-1, 1)
        ys = ys.reshape(-1, 1)

        pts = np.hstack((xs,ys))

        img = img.permute(1,2,0).detach().numpy()                   # [channels, height, width] -> [height, width, channels]）
        
        transformed_imgs = self.transformer.transform(img, pts)

        data = self.transformer.postprocess_transformed_imgs(transformed_imgs)  # postprocess

        return data

    def __getitem__(self, index):

        img = Image.open(self.images[index]).convert('RGB')         # open a image and convert to RGB

        img = self.input_transform(img)
        img*= 255
        img = self.transformImg(img)
        
        return  img, index

    def __len__(self):

        return len(self.images)

    def getPositives(self):

        # positives for evaluation are those within trivial threshold range
        # fit NN to find them, search by radius
        if  self.positives is None:
            knn = NearestNeighbors(n_jobs=-1)       # n_jobs: Number of neighbors to use by default for kneighbors queries.

            knn.fit(self.db_positions)

            self.distances, self.positives = knn.radius_neighbors(self.query_positions,
                                                                  radius=self.pos_threshold)        # radius = 5   #ground truth threshold

        return self.positives
