# Step 1: 追踪；
#         1.1 找所有有重叠的四边形, 有多个就选最近的
#         1.2 没有追踪到的，该点不管
# Step 2: 最小二乘估计追踪到的框的变化transform 【之前的帧 -> anchor的变化】
# Step 3: 利用transform将所有在车辆内部的点进行
#

import torch
import torch.nn as nn
import numpy as np
import math
import random

Using_GPU = False
# rect: [4,2] array
# point: tunple或array

def getCross(p1, p2, p):
	return (p2[0] - p1[0]) * (p[1] - p1[1]) - (p[0] - p1[0]) * (p2[1] - p1[1])

def isPointInQuad(point, rect):
    # 判断一个点是否在四边形内部， 要求点要按照顺序
    a = getCross(rect[0], rect[1], point)
    b = getCross(rect[1], rect[2], point)
    c = getCross(rect[2], rect[3], point)
    d = getCross(rect[3], rect[0], point)
    if (a>0 and b>0 and c>0 and d>0) or (a<0 and b<0 and c<0 and d<0):
         return True
    else:
         return False

def getPointsInQuad(rect):
    # 得到四边形内部所有的点
    min_x = rect[:, 0].min()
    max_x = rect[:, 0].max()
    min_y = rect[:, 1].min()
    max_y = rect[:, 1].max()
    points = []
    for x in range(math.floor(min_x), math.ceil(max_x) + 1):
        for y in range(math.floor(min_y), math.ceil(max_y) + 1):
            if isPointInQuad((x, y), rect):
                points.append([x, y])
    return np.array(points)

def isRectsOverlap(rect1, rect2):
    # 判断两个是rect是否交叉, 只要有一个点在内部就True
    # 1） 判断rect1的所有点是否在rect2内
    # 2) 判断rect2的所有店是否在rect1内
    boolean_v = False
    for i in range(4):
        boolean_v = boolean_v or isPointInQuad(rect1[i], rect2)
    for i in range(4):
        boolean_v = boolean_v or isPointInQuad(rect2[i], rect1)
    return boolean_v

class TwoDTrandform(nn.Module):
    def __init__(self, center):
        super().__init__()
        self.rotation = nn.Parameter(torch.zeros(1))
        self.translation = nn.Parameter(torch.zeros(2))
        # self.scale = nn.Parameter(torch.ones(2))
        self.center = center

    def forward(self, points):
        # x: [N, 2] N个点
        # rotate
        x_prime = points[:,0] * torch.cos(self.rotation) +  \
                points[:,1]*(-torch.sin(self.rotation)) + \
                self.center[0]*(1 - torch.cos(self.rotation)) + self.center[1]*torch.sin(self.rotation)
        y_prime = points[:,0]* torch.sin(self.rotation) + \
                points[:,1]*torch.cos(self.rotation)+ \
                self.center[1]*(1 - torch.cos(self.rotation)) - self.center[0]*torch.sin(self.rotation)
        
        # scale
        # x_prime = self.scale[0]*x_prime + self.center[1]*(1-self.scale[0])
        # y_prime = self.scale[1]*y_prime + self.center[0]*(1-self.scale[1])

        x_prime = x_prime + self.translation[0]
        y_prime = y_prime + self.translation[1]
        return torch.cat([x_prime.unsqueeze(-1), y_prime.unsqueeze(-1)], dim=-1)
    
    def get_transformation_matrix(self):
        beta = self.rotation.detach().cpu().numpy()[0]
        px = self.center.cpu().numpy()[0]
        py = self.center.cpu().numpy()[1]
        t_x = self.translation.detach().cpu().numpy()[0]
        t_y = self.translation.detach().cpu().numpy()[1]
        # s_x = self.scale.detach().numpy()[0]
        # s_y = self.scale.detach().numpy()[1]
        T = [[np.cos(beta), -np.sin(beta), px*(1 - np.cos(beta)) + py*np.sin(beta) + t_x],
         [np.sin(beta),  np.cos(beta), py*(1 - np.cos(beta)) - px*np.sin(beta) + t_y],
         [0           ,  0           , 1                                      ]]
        # T2 = [[s_x, 0, px*(1-s_x) + t_x],
        #       [0, s_y, py*(1-s_y) + t_y],
        #       [0, 0, 1]]
        # T1 = np.array(T1)
        # T2 = np.array(T2)
        return np.array(T)

def applyTransform(points, T):
    # points: N*2
    # T: 3*3
    ones = np.ones((points.shape[0], 1))
    P = np.concatenate((points, ones), axis=1)
    return (np.dot(T, P.T).T)[:,:2] # (3,3) * (3,N) = (3, N)  --> N,3

def random_sample_points(rect1, rect2, N=8):
    # rect1: 4*2  
    # 四个坐标任意比例回合, 总和为1
    # 一个比例 P= N*4  --> P*rect1 = 1*2
    P = np.random.rand(N, 4)  # 先生成四个数，再norm一下
    P = P / P.sum(axis=1, keepdims=True)
    return np.dot(P, rect1), np.dot(P, rect2)
    
def getTransform(rect1, rect2, max_iter=200):
    # rect1 -> rect2 预测两种之间的变换
    rect1_rand, rect2_rand = random_sample_points(rect1, rect2) # 随机采样的在矩形中间的点
    rect1_rand = torch.from_numpy(rect1_rand) 
    rect2_rand = torch.from_numpy(rect2_rand)

    rect1 = torch.from_numpy(rect1) 
    rect2 = torch.from_numpy(rect2)

    rect1_center = rect1.mean(dim=0, keepdim=True)
    rect2_center = rect2.mean(dim=0, keepdim=True)

    x = torch.cat([rect1, rect1_center, rect1_rand], dim=0)
    y = torch.cat([rect2, rect2_center, rect2_rand], dim=0)
    if Using_GPU:
        x = x.cuda()
        y = y.cuda()

    transform_model = TwoDTrandform(rect1_center.squeeze())
    if Using_GPU:
        transform_model = transform_model.cuda()
    # Initialize the loss function
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(transform_model.parameters(), lr=0.02) # 用小学习率，加大步数

    transform_model.train()
    # 这里应该去loss最小时候的矩阵
    min_loss = 1e6
    transformation_matrix = None
    for _ in range(max_iter):
        # Compute prediction and loss
        pred = transform_model(x)
        loss = loss_fn(pred, y)
        if loss < min_loss:
            min_loss = loss
            transformation_matrix = transform_model.get_transformation_matrix()
        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
  
    return transformation_matrix
