import torch.nn as nn

from network.netvlad import NetVLAD
from network.groupnet import GroupNet



class BEVPlace(nn.Module):
    def __init__(self):
        super(BEVPlace, self).__init__()
        self.encoder = GroupNet()           # 
        self.netvlad = NetVLAD()

    def forward(self, input):
        local_feature = self.encoder(input) 
        local_feature = local_feature.permute(0,2,1).unsqueeze(-1)  # permute 方法改变维度的顺序，而 unsqueeze 方法添加一个新的维度。
        global_feature = self.netvlad(local_feature) 

        return global_feature
