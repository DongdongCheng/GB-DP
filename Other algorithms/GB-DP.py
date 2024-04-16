#-------------------------------------------------------------------------
#Aim:
#introduce granular-ball and propose a granular-ball-based DP algorithm, called GB-DP
# -------------------------------------------------------------------------
# Written by Dongdong Cheng, Ya Li
# Chongqing University of Posts and Telecommunications
# 2023

import time
import numpy as np
import matplotlib.pyplot as plt


from scipy.spatial.distance import pdist, squareform
# 计算两点距离
from matplotlib.widgets import RectangleSelector
from sklearn.cluster import k_means
from spyder_kernels.utils.lazymodules import scipy


def draw_point(data):
    N = data.shape[0]
    plt.figure()
    plt.axis()
    for i in range(N):
        plt.scatter(data[i][0],data[i][1],s=16.,c='k')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('origin graph')
    plt.show()

# 判断粒球的标签和纯度
def get_num(gb):
    # 矩阵的行数
    num = gb.shape[0]
    return num

# 返回粒球中心和半径
def calculate_center_and_radius(gb):
    data_no_label = gb[:,:]#取坐标
    center = data_no_label.mean(axis=0)#压缩行，对列取均值  取出平均的 x,y
    radius = np.max((((data_no_label - center) ** 2).sum(axis=1) ** 0.5))  #（x1-x1）**2 + (y1-y2)**2   所有点到中心的距离平均
    return center, radius

def gb_plot(gb_list, plt_type=0):
    plt.figure()
    plt.axis()
    for gb in gb_list:
        center, radius = calculate_center_and_radius(gb)  # 返回中心和半径
        if plt_type == 0:  # 绘制所有点
            plt.plot(gb[:, 0], gb[:, 1], '.', c='k', markersize=5)
        if plt_type == 0 or plt_type == 1:  # 绘制粒球
            theta = np.arange(0, 2 * np.pi, 0.01)
            x = center[0] + radius * np.cos(theta)
            y = center[1] + radius * np.sin(theta)
            plt.plot(x, y, c='r', linewidth=0.8)
        plt.plot(center[0], center[1], 'x' if plt_type == 0 else '.', color='r')  # 绘制粒球中心
    plt.show()


def splits(gb_list, num, splitting_method):
    gb_list_new = []
    for gb in gb_list:
        p = get_num(gb)
        if p < num:
            gb_list_new.append(gb)#该粒球包含的点数小于等于num，那
        else:
            gb_list_new.extend(splits_ball(gb, splitting_method))#反之，进行划分，本来是[[1],[2],[3]]  变成[...,[1],[2],[3]]
    return gb_list_new

def splits_ball(gb, splitting_method):
    splits_k = 2
    ball_list = []

    # 数组去重
    len_no_label = np.unique(gb, axis=0)
    if splitting_method == '2-means':
        if len_no_label.shape[0] < splits_k:
            splits_k = len_no_label.shape[0]
        # n_init:用不同聚类中心初始化运行算法的次数
        #random_state，通过固定它的值，每次可以分割得到同样的训练集和测试集
        label = k_means(X=gb, n_clusters=splits_k, n_init=1, random_state=8)[1]  # 返回标签
    elif splitting_method == 'center_split':
        # 采用正、负类中心直接划分
        p_left = gb[gb[:, 0] == 1, 1:].mean(0)#求坐标平均值
        p_right = gb[gb[:, 0] == 0, 1:].mean(0)
        distances_to_p_left = distances(gb, p_left)#求出各点到平均点的距离
        distances_to_p_right = distances(gb, p_right)

        relative_distances = distances_to_p_left - distances_to_p_right
        label = np.array(list(map(lambda x: 0 if x <= 0 else 1, relative_distances)))

    elif splitting_method == 'center_means':
        # 采用正负类中心作为 2-means 的初始中心点
        p_left = gb[gb[:, 0] == 1, 1:].mean(0)
        p_right = gb[gb[:, 0] == 0, 1:].mean(0)
        centers = np.vstack([p_left, p_right])#[[],[]]
        label = k_means(X=gb, n_clusters=2, init=centers, n_init=10)[1]#以centers为中心进行聚类
    else:
        return gb
    for single_label in range(0, splits_k):
        ball_list.append(gb[label == single_label, :])#按照新打的标签分类
    return ball_list


# 距离
def distances(data, p):
    return ((data - p) ** 2).sum(axis=1) ** 0.5


#计算所有点到粒球中心的平均距离：
def get_ball_quality(gb, center):
    N = gb.shape[0]
    ball_quality =  N
    mean_r = np.mean(((gb - center) **2)**0.5)
    return ball_quality, mean_r


#计算粒球的密度---计算密度的方法二：粒球的密度=粒球的质量/粒球的体积
#粒球的质量=所有点到中心点的平均距离  体积=粒球半径的维数次方radiusA, dimen, ball_qualitysA
def ball_density2(radiusAD, ball_qualitysA, mean_rs):
    N = radiusAD.shape[0]
    ball_dens2 = np.zeros(shape=N)
    for i in range(N):
        if radiusAD[i] == 0:
            ball_dens2[i] = 0
        else:
            ball_dens2[i] = ball_qualitysA[i] / (radiusAD[i] * radiusAD[i] * mean_rs[i])
    return ball_dens2


#计算粒球的相对距离
def ball_distance(centersAD):
    Y1 = pdist(centersAD)
    ball_distAD = squareform(Y1)
    return ball_distAD

#计算最小密度峰距离以及该点ball_min_dist3
def ball_min_dist(ball_distS, ball_densS):
    N3 = ball_distS.shape[0]
    ball_min_distAD = np.zeros(shape=N3)
    ball_nearestAD = np.zeros(shape=N3)
    #按密度从大到小排号
    index_ball_dens = np.argsort(-ball_densS)
    for i3, index in enumerate(index_ball_dens):
        if i3 == 0:
            continue
        index_ball_higher_dens = index_ball_dens[:i3]
        ball_min_distAD[index] = np.min([ball_distS[index, j]for j in index_ball_higher_dens])
        ball_index_near = np.argmin([ball_distS[index, j]for j in index_ball_higher_dens])
        ball_nearestAD[index] = int(index_ball_higher_dens[ball_index_near])
    ball_min_distAD[index_ball_dens[0]] = np.max(ball_min_distAD)
    if np.max(ball_min_distAD) < 1:
        ball_min_distAD = ball_min_distAD * 10
    return ball_min_distAD, ball_nearestAD

#画图
def ball_draw_decision(ball_densS, ball_min_distS):
    Bval1_start = time.time()
    fig, ax = plt.subplots()
    N = ball_densS.shape[0]
    lst = []
    for i4 in range(N):
        ax.plot(ball_densS[i4], ball_min_distS[i4], marker='o', markersize=4.0, c='k')
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
    Bval1_end = time.time()
    Bval1 = Bval1_end - Bval1_start
    return lst, Bval1


#找粒球中心点
def ball_find_centers(ball_densS, ball_min_distS, lst):
    ball_density_threshold = lst[0][0]
    ball_min_dist_threshold = lst[0][1]
    centers = []
    N4 = ball_densS.shape[0]
    for i4 in range(N4):
        if ball_densS[i4] >= ball_density_threshold and ball_min_distS[i4] >= ball_min_dist_threshold:
            centers.append(i4)
    return np.array(centers)


def ball_cluster(ball_densS, ball_centers, ball_nearest, ball_min_distS):
    K1 = len(ball_centers)
    if K1 == 0:
        print('no centers')
        return
    N5 = ball_densS.shape[0]
    ball_labs = -1 * np.ones(N5).astype(int)
    for i5, cen1 in enumerate(ball_centers):
        ball_labs[cen1] = int(i5+1)
    ball_index_density = np.argsort(-ball_densS)
    for i5, index2 in enumerate(ball_index_density):
        if ball_labs[index2] == -1:
            ball_labs[index2] = ball_labs[int(ball_nearest[index2])]
    return ball_labs

def  ball_draw_cluster(centersA, radiusA, ball_labs, dic_colors, gb_list, ball_centers):
    plt.figure()
    N6 = centersA.shape[0]
    for i6 in range(N6):
        for j6, point in enumerate(gb_list[i6]):
            plt.plot(point[0], point[1], marker='o', markersize=4.0, color=dic_colors[ball_labs[i6]])
    plt.show()



if __name__ == "__main__":
    dic_colors = {0: (.8, 0, 0), 1: (0, .8, 0),
                  2: (0, 0, .8), 3: (.8, .8, 0),
                  4: (.8, 0, .8), 5: (0, .8, .8),
                  6: (0, 0, 0), 7: (0.8, 0.8, 0.8),
                  8: (0.6, 0, 0), 9: (0, 0.6, 0),
                  10: (1, 0, .8), 11: (0, 1, .8),
                  12: (1, 1, .8), 13: (0.4, 0, .8),
                  14: (0, 0.4, .8), 15: (0.4, 0.4, .8),
                  16: (1, 0.4, .8), 17: (1, 0, 1),
                  18: (1, 0, .8), 19: (.8, 0.2, 0), 20: (0, 0.7, 0),
                  21: (0.9, 0, .8), 22: (.8, .8, 0.1),
                  23: (.8, 0.5, .8), 24: (0, .1, .8),
                  25: (0.9, 0, .8), 26: (.8, .8, 0.1),
                  27: (.8, 0.5, .8), 28: (0, .1, .8),
                  }


    np.set_printoptions(threshold=1e16)

    # data_mat = np.loadtxt('./data/TB.txt')

    data_mat = scipy.io.loadmat('./data/data_TB_1048576.mat')

    # print(data_mat.keys())
    # print(data_mat['gt'])
    data_mat = data_mat['fea'][0:500000]

    np.savetxt("./data/TB.txt", data_mat)

    #开始时间
    start = time.time()
    data = data_mat
    num = np.ceil(np.sqrt(data.shape[0]))
    # print(max_radius)
    gb_list = [data]
    #全部粒球的展示，不包括在时间的计算中
    # draw_point(data)
    #绘制初始粒球
    # gb_plot(gb_list)
    while True:
        ball_number_1 = len(gb_list)  # 点数
        gb_list = splits(gb_list, num=num, splitting_method='2-means')
        ball_number_2 = len(gb_list)  # 被划分成了几个
        # gb_plot(gb_list)
        if ball_number_1 == ball_number_2:  # 没有划分出新的粒球
            break


    centers = []
    radiuss = []
    ball_num = []#粒球里面的元素个数
    ball_qualitys = []#每个粒球的质量
    mean_rs = []
    i = 0
    for gb in gb_list:
        center, radius = calculate_center_and_radius(gb)
        ball_quality, mean_r = get_ball_quality(gb, center)
        ball_qualitys.append(ball_quality)
        mean_rs.append(mean_r)
        centers.append(center)
        radiuss.append(radius)
        ball_num.append(gb.shape[0])
    centersA = np.array(centers)
    radiusA = np.array(radiuss)
    ball_numA = np.array(ball_num)
    ball_qualitysA = np.array(ball_qualitys)#每一个粒球的半径和中心

    ball_densS = ball_density2(radiusA, ball_qualitysA, mean_rs)

    #计算每个粒球的相对距离
    ball_distS = ball_distance(centersA)
    #计算最小密度峰距离以及该点ball_min_dist  ball_min_distAD, ball_nearestAD
    ball_min_distS, ball_nearest = ball_min_dist(ball_distS, ball_densS)
    # Bval1选中中心所花的时间
    start1 = time.time()
    lst, Bval1 = ball_draw_decision(ball_densS, ball_min_distS)
    end1 = time.time()
    ball_centers = ball_find_centers(ball_densS, ball_min_distS, lst)

    ball_labs = ball_cluster(ball_densS, ball_centers, ball_nearest, ball_min_distS)

    start2 = time.time()
    times = (start2 - start) - (end1 - start1)
    print('运行时间：%s s ' % (times))

    # 最后的聚类结果
    # ball_draw_cluster(centersA, radiusA, ball_labs, dic_colors, gb_list, ball_centers)











