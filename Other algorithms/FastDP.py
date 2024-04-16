import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.widgets import RectangleSelector
from munkres import Munkres
from scipy.spatial.distance import pdist, squareform, cdist
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KDTree
from spyder_kernels.utils.lazymodules import scipy
from sklearn import metrics
from cover_tree import CoverTree

def Label_cluster(PN, cluster, cl, data_size):
    c = 0
    Point_cl = np.zeros(shape=data_size).astype(int)
    for i in range(0, data_size):
        flag = -1
        p = i
        while flag == -1:
            for j in range(0, cl):
                if PN[p] == cluster[j] or p == cluster[j]:
                    flag = 0
                    c = j+1
            if flag == 0:
                Point_cl[i] = c
            else:
                p = PN[p]
    return Point_cl








def draw_cluster(data, Point_cl, cluster, dic_colors):
    time1 = time.time()
    fig, ax = plt.subplots()
    N = cluster.shape[0]
    for i in range(0, N):
        sub_index = np.where(Point_cl == (i+1))
        sub_datas = data[sub_index]
        plt.scatter(sub_datas[:, 0], sub_datas[:, 1], s=16.0, color=dic_colors[i])
        plt.plot(data[int(cluster[i]), 0], data[int(cluster[i]), 1], color="k", marker='+')
    plt.show()
    time2 = time.time()
    intal_time = (time2 - time1)
    return intal_time




def draw1():
    plt.figure()
    plt.plot(data[:, 0], data[:, 1], marker='o', markersize=4.0, c='k')
    for i in range(0, data_size):
        plt.plot([data[i, 0], data[PN[i], 0]], [data[i, 1], data[PN[i], 1]], linewidth=1.5, c='k')
    plt.plot(data[LDP, 0], data[LDP, 1], marker='+', c='g')
    plt.show()

def draw2():
    plt.figure()
    plt.plot(data[:, 0], data[:, 1], marker='o', markersize=4.0, c='k')
    for i in range(0, data_size):
        plt.plot([data[i, 0], data[PN[i], 0]], [data[i, 1], data[PN[i], 1]], linewidth=1.5, c='k')
    plt.plot(data[LDP2, 0], data[LDP2, 1], marker='+', c='g')
    plt.show()


def Search_childnodes_child(childnodes, ldp, n):
    visited = np.zeros(shape=n)
    subnodes = []
    queue = np.zeros(shape=(n+1))
    front = 0
    rear = 0
    queue[rear] = ldp
    rear = rear + 1
    while front != rear:
        tmp = queue[front]
        front = front + 1
        visited[int(tmp)] = 1
        subnodes.append(tmp)
        tmp_sub2 = childnodes[int(tmp)]
        if len(tmp_sub2) != 0:
            for j1 in range(0, len(tmp_sub2)):
                if visited[tmp_sub2[j1]] == 0 and tmp_sub2[j1] not in queue:
                    queue[rear] = tmp_sub2[j1]
                    rear = rear + 1
    subnodes[0] = []
    return subnodes

def Search_subnodes(PN, ldp):
    n = PN.shape[0]
    visited = np.zeros(shape = n)
    subnodes = []
    queue = np.zeros(shape = n).astype(int)
    front = 0
    rear = 0
    queue[rear] = ldp
    rear = rear + 1
    while rear != front:
        tmp = queue[front]
        front = front + 1
        visited[tmp] = 1
        tmp_sub2 = childnodes[int(tmp)]
        if len(tmp_sub2) != 0:
            for j in range(0, len(tmp_sub2)):
                if visited[tmp_sub2[j]] == 0:
                    queue[rear] = tmp_sub2[j]
                    rear = rear + 1
    subnodes.insert(0, [])
    return subnodes

def draw_cluster(data, Point_cl, cluster, dic_colors):
    time1 = time.time()
    fig, ax = plt.subplots()
    N = cluster.shape[0]
    for i in range(0, N):
        sub_index = np.where(Point_cl == (i+1))
        sub_datas = data[sub_index]
        plt.scatter(sub_datas[:, 0], sub_datas[:, 1], s=16.0, color=dic_colors[i])
    # plt.savefig("K40fig2.svg", dpi=600)
    plt.show()
    time2 = time.time()
    intal_time = (time2 - time1)
    return intal_time


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


if __name__ == '__main__':

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

    lines = np.loadtxt("./data/Aggregation.txt")
    datas = np.array(lines).astype(np.float32)
    start_time = time.time()
    data = datas
    dim = np.shape(data)[1]
    data_size = np.shape(data)[0]
    # 初始设置
    cl = 5
    K = 16
    batch_num = 100

    local_peak_threshold = int(0.01 * data_size)  # 最多30个密度

    tree = KDTree(data)
    batch_dist, batch_ind = tree.query(data[:, :], k=K)
    res = batch_ind
    rho = np.zeros(shape=data_size)
    LDP = []
    PN = np.zeros(shape=data_size).astype(int)
    for i in range(0, data_size):
        dist = batch_dist[i][K - 1]
        rho[i] = 1 / dist
    delta = np.ones(shape=data_size)
    delta = delta * float("inf")
    visited = np.zeros(shape=data_size)
    childnodes = []
    for i in range(data_size):
        childnodes.append([])
    num_childnodes = np.zeros(shape=data_size).astype(int)
    for i in range(0, data_size):
        if visited[i] == 0:
            visited[i] = 1
            NNK = res[i, 0:K]
            maxind = np.argmax(rho[NNK])  # 选出K近邻里面密度最大的点的坐标
            if NNK[maxind] == i:
                LDP.append(i)
                visited[NNK] = 1
                PN[NNK] = i
                childnodes[i].extend(NNK[1:K])
                num_childnodes[i] = num_childnodes[i] + NNK.shape[0] - 1
                m = 0
                for k in range(1, K):
                    delta[NNK[m]] = batch_dist[i][k - 1]
                    m += 1
                delta[i] = float("inf")
            else:
                PN[i] = NNK[maxind]
                delta[i] = np.sqrt(np.sum((data[i, :] - data[PN[i], :]) ** 2))
                # childnodes[maxind][num_childnodes[maxind]:num_childnodes[maxind]+1] = i
                t = i
                childnodes[maxind].extend([t])

                num_childnodes[maxind] = num_childnodes[maxind] + 1

    # draw1()
    LDP1 = LDP
    nLDP = len(LDP)
    print('LDP的数目：', nLDP)
    K2 = K
    while nLDP > 0.01 * data_size and K2 <= 0.05 * data_size and K2 < 200:
        K2 = 2 * K2
        for i in range(0, nLDP):
            larger_rho_LDP = []
            if LDP[i] != '':
                batch_dist2, batch_ind2 = tree.query([data[LDP[i], :]], k=K2)
                NNK_LDP = batch_ind2[0:K2 - 1]  # 取到LDP的K2近邻的索引
                for k in range(1, K2):
                    if rho[LDP[i]] < rho[NNK_LDP[0][k]]:  #
                        larger_rho_LDP.append(k)
            if len(larger_rho_LDP) > 0:
                dist2 = cdist([data[LDP[i], :]], data[NNK_LDP[0][larger_rho_LDP], :])  # 求出密度峰到近邻的距离
                dind = np.argmin(dist2)
                temp_pn = NNK_LDP[0][larger_rho_LDP[dind]]
                PN[LDP[i]] = temp_pn
                # childnodes[temp_pn][num_childnodes[temp_pn]:num_childnodes[temp_pn]+1] = LDP[i]
                childnodes[temp_pn].extend([LDP[i]])
                num_childnodes[temp_pn] = num_childnodes[temp_pn] + 1
                delta[LDP[i]] = np.sqrt(np.sum(data[LDP[i], :] - data[PN[LDP[i]], :]) ** 2)
                LDP[i] = 0
    LDP2 = []
    for index, value in enumerate(LDP):
        if value != 0:
            LDP2.append(value)
    nLDP2 = len(LDP2)
    print('LDP2的数目：', nLDP2)
    # draw2()
    # 快速找到局部密度峰的父节点
    sldp = np.argsort(rho[LDP2])  # 从小到大排列
    delta[LDP2[sldp[nLDP2 - 1]]] = float('inf')
    w = nLDP2 - 1
    for w1 in range(w, -1, -1):  # 从大到小查找
        i = LDP2[sldp[w1]]
        mindist = float('inf')
        local_index = -1
        for q in range(w1 + 1, nLDP2):
            j = LDP2[sldp[q]]
            d0 = np.sqrt(np.sum((data[i, :] - data[j, :]) ** 2))
            if mindist > d0:
                mindist = d0
                local_index = j
            subnodes = Search_childnodes_child(childnodes, j, data_size)
            flag = np.zeros(shape=data_size).astype(int)
            for k3 in range(1, len(subnodes)):
                temp = int(subnodes[k3])
                if flag[int(temp)] == 0 and rho[i] <= rho[int(temp)] and i != int(temp) and PN[temp] != i:
                    flag[int(temp)] = 1
                    d = np.sqrt(np.sum((data[i, :] - data[int(temp), :]) ** 2))
                    if mindist > d:
                        mindist = d
                        local_index = temp
                        continue
                    d2 = cdist([data[temp, :]], data[res[temp, 1:K]])
                    l = K
                    while l > 0:
                        if d > delta[i] + d2[0][l - 2]:
                            flag[res[temp, 1:K]] = 1
                            break
                        l -= 1
        delta[i] = mindist
        PN[i] = local_index
        print(w1)
    for i in range(0, data_size):
        if delta[i] == float('inf'):
            delta[i] = np.max(delta)
            delta[i] = 0
    for i in range(0, data_size):
        if delta[i] == 0:
            delta[i] = np.max(delta)
    t = int(len(LDP))
    r = np.zeros(shape=t)
    for i in range(0, t):
        r[i] = rho[LDP[i]] * delta[LDP[i]]
    r_index = np.argsort(-r)
    r_cluster = r_index[0:cl]
    cluster = np.zeros(shape=cl)
    for i in range(0, cl):
        cluster[i] = LDP[int(r_cluster[i])]
    print(cluster)
    for i in range(0, cl):
        PN[int(cluster[i])] = -1
    Point_cl = Label_cluster(PN, cluster, cl, data_size)
    print('2')
    start1 = time.time()
    all_time = (start1 - start_time)

    print('运行时间：%s s ' % (all_time))
    intal_time = draw_cluster(data, Point_cl, cluster, dic_colors)
    #
    # print(Point_cl.shape)
    # print(line_target.shape)
    err_rate1 = err_rate(line_target, Point_cl)
    print("error:", err_rate1)
    get_matchlabel = best_map(line_target, Point_cl)
    # print(get_matchlabel)
    result_ACC = accuracy_score(line_target, get_matchlabel)
    result_NMI = metrics.normalized_mutual_info_score(line_target, get_matchlabel)

    print('算法的ACC准确率为：', result_ACC)
    print('算法的NMI准确率为：', result_NMI)