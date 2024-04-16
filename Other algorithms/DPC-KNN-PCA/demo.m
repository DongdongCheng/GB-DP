dataName = 'dataset16';
%dataName = 'DBRHD';
fea = [];

load(['',dataName,'.mat'],'ijcnn1')
% A = load('2clusters_100.txt')
% [n,m] = size(B)
% disp([n,m])
% disp(B)
% gt = load('mushrooms_tsne_2_label.txt')
% gt = A(:,3)
% % disp(gt)
% A(:,3) = []
gt = []
%  SNNDPC2(DS1,20,gt)
DPC_KNN_PCA(ijcnn1, 0.05, gt)
% csvwrite('pd_d6.csv',cl,0,0)