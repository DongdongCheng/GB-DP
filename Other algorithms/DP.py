import time

import numpy
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Rectangle
from munkres import Munkres
from  scipy.spatial.distance import pdist, squareform
# 计算两点距离
from matplotlib.widgets import RectangleSelector
from sklearn import datasets
from sklearn import metrics
from sklearn.cluster import k_means
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from spyder_kernels.utils.lazymodules import scipy
import random

def get_distance(x):
    Y = pdist(x, 'euclidean')
    distance1 = squareform(Y)
    distance1 = distance1
    # size = x.shape[0]
    # distance1 = np.zeros(shape=(size, size), dtype=np.float16)
    # for i in range(0, size - 1):
    #         for j in range(i + 1, size):
    #           distance1[i][j] = np.sqrt(np.sum(np.power(x[i] - x[j], 2)))
    #           distance1[j][i] = distance1[i][j]
    return distance1


# 计算dc
def get_dc(distance1, percent):
    N = distance1.shape[0]
    temp = distance1.flatten()
    temp.sort()
    dc1 = temp[int(N*(N-1) * percent / 100)]
    return dc1


# 计算局部密度
# cut-off kernel
# def get_density1(distance, dc):
#     density = np.zeros(distance.shape[0])
#     for index, dist in enumerate(distance):
#           print('1')
#     return density


# Gaussian kernal
def get_density2(distance, dc):
    N = distance.shape[0]
    density = np.zeros(N)
    for i in range(N):
        density[i] = np.sum(np.exp(-(distance[i, :] / (dc+1e-5)))**2)
    return density


# 找最近的高密度点
def get_min_dist(distance, density):
    N = distance.shape[0]
    min_dist = np.zeros(N)
    nearest = np.zeros(N)
    # argsort=>按density从大到小排列的标号
    index_density = np.argsort(-density)
    for i, index in enumerate(index_density):
        if i == 0:
            continue
        # 取到i之前的标号
        index_higher_density = index_density[:i]
        # 找到i之前标号的距离的最小值
        min_dist[index] = np.min([distance[index, j] for j in index_higher_density])
        # print(min_dist[index])
        index_near = np.argmin([distance[index, j] for j in index_higher_density])
        nearest[index] = int(index_higher_density[index_near])
    min_dist[index_density[0]] = np.max(min_dist)
    if np.max(min_dist) < 1:
        min_dist = min_dist * 10
    return min_dist, nearest


# 中心点
def find_centers(density, min_dist, lst):
    density_threshold = lst[0][0]
    min_dist_threshold = lst[0][1]
    max_dist = np.max(min_dist)
    centers = []
    N = density.shape[0]
    for i in range(N):
        if density[i] >= density_threshold and min_dist[i] >= min_dist_threshold:
            centers.append(i)
    return np.array(centers)




# 聚类labs = cluster(density, centers, nearest, distance)
def cluster(density, centers, nearest , distance):
    K = centers.shape[0]
    if K == 0:
        print("no centers")
        return

    N = density.shape[0]
    labs = -1 * np.ones(N).astype(int)
    # print(labs)
    for i, cen in enumerate(centers):
        labs[cen] = i
    #先对密度比较的点进行标号
    index_density = np.argsort(-density)
    for i, index in enumerate(index_density):
        if labs[index] == -1:
            labs[index] = labs[int(nearest[index])]
    return labs
# halos
#     halos = np.zeros(N).astype(int)
#     if K > 1:#作者建议
#         for i in range(N):
#            halos[i] = labs[i]
#         #初始化数组 bord_rho 为 0,每个 cluster 定义一个 bord_rho 值
#         bord_rho = np.zeros(K)
#         #print(bord_rho)
#         for j in range(0, N-1):
#             for t in range(j+1, N):
#
#                 if labs[j] != labs[t] and distance[j][t] <= dc:#边缘
#                     density_aver = (density[j] + density[t]) / 2
#                 #print(density_aver)
#                 #将平均密度赋值给边缘密度
#                     if density_aver > bord_rho[labs[j]]:
#                         bord_rho[labs[j]] = density_aver
#                     if density_aver > bord_rho[labs[t]]:
#                         bord_rho[labs[t]] = density_aver
#     #密度小于边缘密度的，为hole
#         for n in range(N):
#             if density[n] < bord_rho[labs[n]]:#当点的密度小于该簇的边缘密度时
#                 halos[n] = -1
#     return halos

def draw_point(data):
    N = data.shape[0]
    plt.figure()
    plt.axis()
    for i in range(N):
        # plt.scatter(data[i][0],data[i][1],s=16.,c='k')
        plt.plot(data[i][0],data[i][1], marker='o', markersize=4.0, color='k')
        plt.annotate(str(i), xy=(data[i][0],data[i][1]), xytext=(data[i][0],data[i][1]))
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('origin graph')
    plt.show()



def draw_decision(density, min_dist):
    interval_start = time.time()
    fig, ax = plt.subplots()
    N = density.shape[0]
    lst = []
    for i in range(N):
        # ax.scatter(density[i], min_dist[i], s=16., c='k')
        ax.plot(density[i], min_dist[i], marker='o', markersize=4.0, color='k')
        # ax.annotate(str(i), xy=(density[i], min_dist[i]), xytext=(density[i], min_dist[i]))
        plt.xlabel('density')
        plt.ylabel('min_dist')
        ax.set_title('decision graph')

    # 矩形选区选择时的回调函数
    def select_callback(eclick, erelease):
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        lst.append([x1, y1])
    RS = RectangleSelector(ax, select_callback,
                           drawtype='box', useblit=True,
                           button=[1, 3],  # disable middle button
                           minspanx=0, minspany=0,
                           spancoords='data',
                           interactive=True)
    # a = Annotate()

    plt.show()
    interval_end = time.time()
    interval = (interval_end - interval_start)
    return lst


def draw_cluster(datas, labs, centers, dic_colors):
    val2_start = time.time()
    plt.cla()
    K = centers.shape[0]
    for k in range(K):
        sub_index = np.where(labs == k)
        sub_datas = datas[sub_index]
        plt.scatter(sub_datas[:, 0], sub_datas[:, 1], s=16., color=dic_colors[k])
        # plt.scatter(datas[centers[k], 0], datas[centers[k], 1], color="k", marker='+', s=50.)
    # sub_halos_index = np.where(labs == -1)
    # sub_halos_datas = datas[sub_halos_index]
    # plt.scatter(sub_halos_datas[:, 0], sub_halos_datas[:, 1],  s=16.,  c='k')
    plt.savefig("DP_s3.jpg", dpi=600)
    plt.show()
    val2_end = time.time()
    val2 = (val2_end - val2_start)
    return val2


def best_map(L1, L2):
    """L1 should be the labels and L2 should be the clustering number we got"""
    Label1 = np.unique(L1)  # 去除重复的元素，由小大大排列
    nClass1 = len(Label1)  # 标签的大小
    Label2 = np.unique(L2)
    nClass2 = len(Label2)
    nClass = np.maximum(nClass1, nClass2)
    G = np.zeros((nClass, nClass))
    for i in range(nClass1):
        ind_cla1 = L1 == Label1[i]
        ind_cla1 = ind_cla1.astype(float)
        for j in range(nClass2):
            ind_cla2 = L2 == Label2[j]
            ind_cla2 = ind_cla2.astype(float)
            G[i, j] = np.sum(ind_cla2 * ind_cla1)
    m = Munkres()
    index = m.compute(-G.T)
    index = np.array(index)
    c = index[:, 1]
    newL2 = np.zeros(L2.shape)
    for i in range(nClass2):
        newL2[L2 == Label2[i]] = Label1[c[i]]
    return newL2


def err_rate(gt_s, s):
    c_x = best_map(gt_s, s)
    err_x = np.sum(gt_s[:] != c_x[:])
    missrate = err_x.astype(float) / (gt_s.shape[0])
    return missrate


if __name__ == "__main__":
    dic_colors = {
        0: (0.3, 0.9, 0), 1: (0, .8, 0),
        2: (0, 0, .8), 3: (.8, .8, 0),
        4: (.8, 0, .8), 5: (0, .8, .8),
        6: (0, 0, 0), 7: (0.8, 0.8, 0.8),
        8: (0.6, 1, 0), 9: (1, 0.6, 0),
        10: (0, 1, 0.6), 11: (1, 0, 0.6),
        12: (0.8, 0.2, 0), 13: (0.6, 0.8, 0),
        14: (0, 0.8, 0.6), 15: (0.3, 0.9, 0),
    }

    np.set_printoptions(threshold=1e8)

    lines = np.loadtxt("./data/Aggregation.txt")

    datas = np.array(lines)
    print(datas)
    start = time.time()
    # 计算距离矩阵
    distance = get_distance(datas)

    # 计算dc
    dc = get_dc(distance, 2)
    # print("dc", dc)
    # 计算局部密度
    density = get_density2(distance, dc)
    # print(rho)
    # 计算密度距离
    min_dist, nearest = get_min_dist(distance, density)

    # 获取聚类中心点
    # centers = find_centers(density, min_dist)
    start1 = time.time()
    lst = draw_decision(density, min_dist)
    end1 = time.time()
    centers = find_centers(density, min_dist, lst)
    # print(centers)
    labs = cluster(density, centers, nearest, distance)
    start2 = time.time()
    times = (start2 - start) - (end1 - start1)
    print('运行时间：%s s ' % (times))
    draw_cluster(datas, labs, centers, dic_colors)
    print(labs)
    # np.savetxt("../data/人工数据集2/data/Dataset6_label.txt",labs)

    #
    err_rate1 = err_rate(line_target, labs)
    print("error:", err_rate1)
    get_matchlabel = best_map(line_target, labs)
    # print(get_matchlabel)
    result_ACC = accuracy_score(line_target, get_matchlabel)
    result_NMI = metrics.normalized_mutual_info_score(line_target, get_matchlabel)

    print('算法的ACC准确率为：', result_ACC)
    print('算法的NMI准确率为：', result_NMI)