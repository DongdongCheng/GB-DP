function [ cl2,sharedCount,short_path] = DPSLORE( A )
%基于局部核心点共享近邻距离的DP聚类算法
%   基本思想是获得局部核心点，然后重新定义局部核心点之间基于共享近邻的距离，利用密度定义和新的距离定义，利用DP算法对局部核心点进行聚类，最后将局部核心点的聚类结果扩展到原始数据集上
%A 数据集
%stopK聚类数
tic;
%  [A] = max_min_norm(A);
[N,dim]=size(A);
%首先搜索核心点
% [index,supk,max_nb,rho,local_core,cores,cl,cluster_number ] = CoreSearch4(A);
% [index,supk,nb,rho,local_core,cores,cl,cluster_number ] = CoreSearch3(A);
  [dist,index,supk,max_nb,rho,local_core,cores,cl,cluster_number] = CoreSearch2(A);
% [cores,local_core,cl,supk,cluster_number,index] = KeyPoint(A);
%利用核心点之间的距离构造最小生成树

%第一种方法是利用每个簇与其他簇的k近邻的交集
[rho_sorted,ordrho]=sort(rho,'ascend');
% alpha=0.15;%0.05;
rho_threshold=0;%rho_sorted(floor(N*alpha));%0;
for i=1:cluster_number
    if cores(i)~=0
    if rho(cores(i))<rho_threshold %排除密度较小的核心点
        mind=inf;
        p=0;
        for j=1:cluster_number
            if i~=j
                x=A(cores(i),:);
                y=A(cores(j),:);
                distance=sqrt(sum((x-y).^2));
%                 distance=pdist2(A(cores(i),:),A(cores(j),:));
            if mind>distance&&rho(cores(j))>rho_threshold
                mind=distance;
                p=j;
            end
            end
        end
        for j=1:N
            if local_core(j)==cores(i)
                local_core(j)=cores(p);
            end
        end
    end
    end
end
cluster_number=0;
cl=zeros(N,1);
for i=1:N
    if local_core(i)==i;
       cluster_number=cluster_number+1;
       cores2(cluster_number)=i;
       cl(i)=cluster_number;
    end
end
% disp('初始子簇个数为：');disp(cluster_number);
% 以下是得出准核心直接得到的子簇
for i=1:N
    cl(i)=cl(local_core(i));
end
% %画出核心点图以及相应的初始聚类结果
% figure(1);
% plot(A(:,1),A(:,2),'.');
% hold on;
% for i=1:N
%     plot([A(i,1),A(local_core(i),1)],[A(i,2),A(local_core(i),2)]);
%     hold on;
% end
% % drawcluster2(A,cl,cluster_number+1);
% hold on;
% plot(A(local_core,1),A(local_core,2),'ro','MarkerSize',5,'MarkerFaceColor','r','MarkerEdgeColor','r');
% hold off;
cdata=cell(1,cluster_number);%保存每个簇中都有哪些点
cdataexp=cell(1,cluster_number);%保存每个簇中的点及每个点中的k近邻
nc=zeros(1,cluster_number);%保存属于某个核心点的点数
ncexp=zeros(1,cluster_number);
core_dist=zeros(cluster_number,cluster_number);
for i=1:cluster_number
    for j=i+1:cluster_number
        x=A(cores2(i),:);
        y=A(cores2(j),:);
        d=sqrt(sum((x-y).^2));
        core_dist(i,j)=d;
        core_dist(j,i)=d;
    end
end

maxd=max(max(core_dist));
sd=zeros(cluster_number,1);
for i=1:cluster_number
    %保存每个簇中的点
    nc(i)=0;
    ncexp(i)=0;
    x=A(cores2(i),:);
    for j=1:N
        if cl(j)==i
            nc(i)=nc(i)+1;
            y=A(j,:);
            sd(i)=sd(i)+sqrt(sum((x-y).^2));
            ncexp(i)=ncexp(i)+1;
            cdata{1,i}(1,nc(i))=j;
            cdataexp{1,i}(1,ncexp(i))=j;
        end
    end
    %寻找第i个簇中每个对象的Non最近邻居加入到该簇中
    for j=1:ncexp(i)
       x=cdata{1,i}(1,j);
       d2=sqrt(sum((A(x,:)-A(cores2(i),:)).^2));
%        if d2<1.5*sd(i)/nc(i)
       for k=2:supk+1
           kneighbor=index(x,k);
           if iscontain(cdataexp{1,i}',kneighbor)==0&&rho(kneighbor)>rho_threshold
               ncexp(i)=ncexp(i)+1;
               cdataexp{1,i}(1,ncexp(i))=kneighbor;
           end
       end
%        end
    end 
   
end
%计算任意两个簇之间的重新定义的距离dist/intersect

sim=zeros(cluster_number,cluster_number);
sharedCount=zeros(cluster_number,cluster_number);
for i=1:cluster_number
    for j=i+1:cluster_number
        inset1=intersect(cdataexp{1,i},cdataexp{1,j});
%         inset2=intersect(cdata{1,i},cdataexp{1,j});
        averho=sum(rho(inset1));
        [~,numinset1]=size(inset1);
        sharedCount(i,j)=numinset1;
        sharedCount(j,i)=numinset1;
%         [~,numinset2]=size(inset2);
%         fprintf('第%d个点和第%d个点的交集数为：%d,密度和为%f\n',i,j,numinset1,averho);
        if numinset1==0%&&numinset2==0
            core_dist(i,j)=maxd;%core_dist(i,j);
            core_dist(j,i)=core_dist(i,j);
        else
%             dist1=0;
%             dist2=0;
%             for k=1:numinset1
%                 dist1=dist1+sqrt(sum((A(cores(i),:)-A(inset1(k),:)).^2));
%                 dist2=dist2+sqrt(sum((A(cores(j),:)-A(inset1(k),:)).^2));
%             end
%             core_dist(i,j)=(dist1+dist2)/(numinset1^2);
            core_dist(i,j)=core_dist(i,j)/(averho*numinset1);
            core_dist(j,i)=core_dist(i,j);
        end
        
    end
end
short_path=zeros(cluster_number,cluster_number);
for i=1:cluster_number
     short_path(i,i)=maxd;
     [D,Z]=dijkstra2(core_dist,i);%D中存放的是原点到其他每个顶点的最短路径的长度，Z点存放的是每个点的先驱顶点
     for j=i+1:cluster_number
         short_path(i,j)=D(j);
         if short_path(i,j)==inf
             short_path(i,j)=0;
         end
         
         short_path(j,i)=short_path(i,j);
     end
end
% maxsp=max(max(short_path));
% short_path(find(short_path==maxsp))=2*maxsp;
%利用DP算法对局部核心点进行聚类
core_data=A(cores2,:);
core_rho=rho(cores2);
[ core_cl,ncluster ] = DP(core_data,short_path,core_rho );
% [ core_cl,ncluster ] = DP2SNN(core_data, short_path,core_rho,sharedCount,supk/2 );
cl2=zeros(N,1);
cl2(cores2)=core_cl;
for i=1:N
    cl2(i)=cl2(local_core(i));
end

%csvwrite('./data/label2.csv',cl2,0,0);
% SD=sparse(core_dist);
% UG=tril(SD);
% [ST,pred] = graphminspantree(UG,'METHOD','Prim');
% figure(2);
% plot(A(:,1),A(:,2),'.');
% hold on;
% % for i=1:N
% %     if local_core(i)~=0;
% %     plot([A(i,1),A(local_core(i),1)],[A(i,2),A(local_core(i),2)],[A(i,3),A(local_core(i),3)]);
% %     hold on;
% %     end
% % end
% % plot(A(cores,1),A(cores,2),'r*','MarkerSize',8);
% % hold on;
figure;
drawcluster2(A,cl2,ncluster);

end

