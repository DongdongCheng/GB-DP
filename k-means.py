import time

import matplotlib.pyplot as plt
import numpy as np
from munkres import Munkres
from sklearn.metrics import accuracy_score
from spyder_kernels.utils.lazymodules import scipy

from sklearn import datasets, metrics

from sklearn.cluster import k_means

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

np.set_printoptions(threshold=1e6)
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

def draw_cluster(datas, labs, dic_colors):
    val2_start = time.time()
    plt.cla()
    for k in range(5):
        sub_index = np.where(labs == k)
        sub_datas = datas[sub_index]
        plt.scatter(sub_datas[:, 0], sub_datas[:, 1], s=16., color=dic_colors[k])
    # sub_halos_index = np.where(labs == -1)
    # sub_halos_datas = datas[sub_halos_index]
    # plt.scatter(sub_halos_datas[:, 0], sub_halos_datas[:, 1],  s=16.,  c='k')
    # plt.savefig("6.svg", dpi=600)
    plt.show()
    val2_end = time.time()
    val2 = (val2_end - val2_start)
    return val2
dic_colors = {0: (.8, 0, 0), 1: (0, .8, 0),
                  2: (0, 0, .8), 3: (.8, .8, 0),
                  4: (.8, 0, .8), 5: (0, .8, .8),
                  6: (0, 0, 0), 7: (0.8, 0.8, 0.8),}



cl = 5
lines = np.loadtxt("./data/Aggregation.txt")
start1 = time.time()
label1 = k_means(X=lines, n_clusters=5, n_init=1, random_state=10)[1]
end1 = time.time()
times = end1 - start1
print('running time ：%s s ' % (times))
# label1 = best_map(line_target,label)

draw_cluster(lines, label1, dic_colors)
# np.savetxt("../data/人工数据集2/data/large_label.txt", label)
err_rate1 = err_rate(line_target, label1)
# print("error:", err_rate1)
get_matchlabel = best_map(line_target, label1)
# # print(get_matchlabel)
result_ACC = accuracy_score(line_target, get_matchlabel)
result_NMI = metrics.normalized_mutual_info_score(line_target, get_matchlabel)


print('算法的ACC准确率为：', result_ACC)
print('算法的NMI准确率为：', result_NMI)
