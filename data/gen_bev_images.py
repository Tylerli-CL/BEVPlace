import numpy as np
import matplotlib.pyplot as plt
# import pcl
import open3d as o3d
import cv2
import os
import argparse             # 导入 argparse 用于解析命令行参数。
from tqdm import trange     # 导入 tqdm 的 trange 函数，用于显示进度条。

# Set up command line arguments
parser = argparse.ArgumentParser(description='BEVPlace-Generate-BEV-Images')
parser.add_argument('--vel_path', type=str, default="/mnt/share_disk/KITTI/dataset/sequences/00/velodyne/", help='path to data')    # “vel_path”是指“velodyne路径”，这里默认是在外接硬盘 /mnt 是一个标准目录，通常用于临时挂载（mount）文件系统。
parser.add_argument('--bev_save_path', type=str, default="./KITTI_new_imgs/00/imgs/", help='path to save data')                     # saving path of the BEV Images


# Function to generate BEV image from point cloud
def getBEV(all_points): # N*3 N_points with 3 dimensional
    
    all_points_pc = o3d.geometry.PointCloud()   # pcl.PointCloud()

    all_points_pc.points = o3d.utility.Vector3dVector(all_points)   # all_points_pc.from_array(all_points)
    all_points_pc = all_points_pc.voxel_down_sample(voxel_size=0.4) # f = all_points_pc.make_voxel_grid_filter()
    

    all_points = np.asarray(all_points_pc.points)# np.array(all_points_pc.to_list())


    x_min = -40
    y_min = -40
    x_max = 40
    y_max = 40

    x_min_ind = np.floor(x_min/0.4).astype(int)
    x_max_ind = np.floor(x_max/0.4).astype(int)
    y_min_ind = np.floor(y_min/0.4).astype(int)
    y_max_ind = np.floor(y_max/0.4).astype(int)

    x_num = x_max_ind-x_min_ind+1
    y_num = y_max_ind-y_min_ind+1

    # 初始化一个零矩阵，也就是图片
    mat_global_image = np.zeros((y_num, x_num),dtype=np.uint8)
          
    for i in range(all_points.shape[0]):    # 点云中所有的点

        # **注意这里的 x y 转换有不一样的
        # 映射到图像矩阵：
        # x_max_ind 和 y_max_ind 是图像矩阵中 x 和 y 轴的最大索引值
        x_ind = x_max_ind - np.floor(all_points[i,1]/0.4).astype(int)     # 通过 np.floor() 函数将这些值向下取整到最接近的整数，并转换为整型.astype(int)
        y_ind = y_max_ind - np.floor(all_points[i,0]/0.4).astype(int)

        if(x_ind>=x_num or y_ind>=y_num):
            continue
        if mat_global_image[ y_ind,x_ind]<10:
            mat_global_image[ y_ind,x_ind] += 1

    max_pixel = np.max(np.max(mat_global_image))

    # 图片中小于等于1的 都置为0
    mat_global_image[mat_global_image<=1] = 0
    # 整个图片中的大小都乘以10
    mat_global_image = mat_global_image*10
    
    # 将大于255的都置为255
    mat_global_image[np.where(mat_global_image>255)]=255

    # 
    mat_global_image = mat_global_image/np.max(mat_global_image)*255

    return mat_global_image, x_max_ind, y_max_ind


if __name__ == "__main__":

    args = parser.parse_args()

    bins_path = os.listdir(args.vel_path)   # velodyne path
    bins_path.sort()    # 因为KITTI的数据集有好几个，并且按照数字标号，排序后方便按顺序处理

    os.system('mkdir -p ' + args.bev_save_path)   # 当使用 -p 选项时，mkdir 会创建所有必需的上级目录，从而确保整个目录路径被成功创建。如果所指定的目录已存在，mkdir -p 不会显示错误信息。

    for i in trange(len(bins_path)):    # trange 进度条

        b_p = bins_path[i]
        pcs = np.fromfile(args.vel_path + '/' + b_p, dtype=np.float32).reshape(-1,4)[:,:3] # 从二进制文件中读取Velodyne点云数据，将数据类型设为np.float32，并将每行数据重新整形为三列（x、y、z），丢弃了第四列，因为通常它是强度信息。

        # ang = np.random.randint(360)/180.0*np.pi
        # rot_mat = np.array([[np.cos(ang),np.sin(ang),0],[-np.sin(ang),np.cos(ang),0],[0,0,1]])
        # pcs = pcs.dot(rot_mat)

        # 用于过滤点云数据，删除超出指定范围的点。在这个示例中，对x、y和z坐标都进行了过滤，将超出[-25, 25]的点删除。
        pcs = pcs[np.where(np.abs(pcs[:, 0])<25)[0],:]
        pcs = pcs[np.where(np.abs(pcs[:, 1])<25)[0],:]
        pcs = pcs[np.where(np.abs(pcs[:, 2])<25)[0],:]

        pcs = pcs.astype(np.float32)    # 转化为np.float32

        img, _, _ = getBEV(pcs)     # generate BEV images

        cv2.imwrite(args.bev_save_path+'/'+b_p[:-4]+".png",img)     # save the BEV Images

exit()
