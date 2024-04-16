%重新修改的寻找核心点的算法
function [dist,index,supk,max_nb,rho,local_core,cores,cl,cluster_number] = CoreSearch2(A)
[N,dim]=size(A);
dist=zeros(N,N);
    for i=1:N
        for j=1:N
            for k=1:dim
            dist(i,j)=dist(i,j)+(A(i,k)-A(j,k))^2;
            end
            dist(i,j)=sqrt(dist(i,j));
        end
    end
[sdist,index]=sort(dist,2);%对dist按行进行排序
%初始化基本数据
r=1;
flag=0;         
nb=zeros(1,N);  %自然邻居个数 
%NNN=zeros(N,N); %各点的自然邻居集
count=0;        %自然最近邻数为零的数据量连续相同的次数
count1=0;       %前一次自然最近邻数为零的数据量
count2=0;       %此次自然最近邻数为零的数据量

%搜索自然最近邻居
while flag==0
    for i=1:N
        k=index(i,r+1);
        nb(k)=nb(k)+1;
      %  NNN(k,nb(k))=i;
    end
    r=r+1;
    count2=0;
    for i=1:N
        if nb(i)==0
            count2=count2+1;
        end
    end
    %计算nb(i)=0的点的数量连续不变化的次数
    if count1==count2
        count=count+1;
    else
        count=1;
    end
    if count2==0 || (r>2 && count>=2)   %邻居搜索终止条件
        flag=1;
    end
    count1=count2;
end

%计算自然最近邻的各种特征量
supk=r-1;               %最终K值，也是自然最近邻居的平均数
max_nb=max(nb);         %自然邻居的最大数目
min_nb=min(nb);         %自然邻居的最小数目
%NN=index(:,2:SUPk+1);   %各数据点的K近邻数据点集
%ratio_nb=nb./(N*SUPk);  %各数据点的自然最近邻居数目所占比例
%计算每个数据点的密度
%disp(SUPk);
%构造连接矩阵
%disp(supk);
%disp(max_nb);
%disp(min_nb);
rho=zeros(N,1);
Non=max_nb;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%构造自然邻域图
conn=zeros(N,N);
for i=1:N
    for j=2:supk+1
        x=index(i,j);
        conn(i,x)=1/(1+dist(i,x));%距离的倒数作为两点的相似度
        conn(x,i)=conn(i,x);
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for i=1:N
    d=0;
    for j=1:Non+1
        d=d+sdist(i,j);
    end
    rho(i)=(Non/d);
end
[rho_sorted,ordrho]=sort(rho,'descend');%ordrho就是密度从大到小的顺序
local_core=zeros(N,1);%存放n个点的局部核心点
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%按照密度从大到小的顺序去发现数据的准核心点
for i=1:N
%      if local_core(ordrho(i))==0
         p=ordrho(i);
         maxrho=rho(p);
         maxindex=p;
         for j=1:nb(p)+1%求出k近邻中密度最大的点作为i及其k近邻的核心点
             x=index(p,j);
             if maxrho<rho(x)
                 maxrho=rho(x);
                 maxindex=x;
             end
         end
         %对具有最大密度的点分配局部代表点
         if local_core(maxindex)==0%如果该最大密度点也没有分配核心点
             local_core(maxindex)=maxindex;            
         end
         %得到点p的k近邻的局部代表点
         for j=1:nb(p)+1
             if local_core(index(p,j))==0%如果第j个近邻还没有代表点
                 local_core(index(p,j))=local_core(maxindex);
             else%如果第j个近邻已经有代表点了，选择距离较近的一个代表点作为新的代表点
                 q=local_core(index(p,j));
                 if dist(index(p,j),q)>dist(index(p,j),local_core(maxindex))%rho(local_core(maxindex))>=rho(q)%
                     local_core(index(p,j))=local_core(maxindex);
                 end
             end 
             for m=1:N
                 if local_core(m)==index(p,j)
                     local_core(m)=local_core(index(p,j));
                 end
             end
         end
         
%        for j=1:Non+1
%         if local_core(index(ordrho(i),j))==0||local_core(index(ordrho(i),j))~=0&&rho(local_core(index(ordrho(i),j)))<rho(maxindex)%dist(index(ordrho(i),j),local_core(index(ordrho(i),j)))>dist(index(ordrho(i),j),maxindex)%%当这个点还没有分配核心点或者已经分配核心点，比较这两个核心点哪个比较近就分配到哪个核心点中
%             local_core(index(ordrho(i),j))=maxindex;
%             delta(index(ordrho(i),j))=dist(index(ordrho(i),j),maxindex);
%             if local_core(maxindex)==0%如果这个密度较大的点还没有分配核心点，就为其分配核心点为它自己
%                 local_core(maxindex)=maxindex;
%            else%如果这个点已经分配核心点，比较两个核心点的密度，取较大密度的核心点作为核心点
% %                if rho(local_core(index(ordrho(i),j)))<rho(maxindex)
%            local_core(index(ordrho(i),j))=local_core(maxindex);
%            delta(index(ordrho(i),j))=dist(index(ordrho(i),j),local_core(maxindex));
% %                end
%             end
%         end 
%         for m=1:N
%              if local_core(m)==index(ordrho(i),j)
%                  local_core(m)=local_core(index(ordrho(i),j));
%              end
%         end
%          
%       end
%     end    
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%得到准核心点，准核心点以其自身为核心点
 cluster_number=0;
 cl=zeros(N,1);
for i=1:N
    if local_core(i)==i;
       cluster_number=cluster_number+1;
       cores(cluster_number)=i;
       cl(i)=cluster_number;
    end
end
disp('初始子簇个数为：');disp(cluster_number);
% 以下是得出准核心直接得到的子簇
for i=1:N
    cl(i)=cl(local_core(i));
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%画出核心点图以及相应的初始聚类结果
plot(A(:,1),A(:,2),'.');
hold on;
for i=1:N
    plot([A(i,1),A(local_core(i),1)],[A(i,2),A(local_core(i),2)]);
    hold on;
end
%drawcluster2(A,cl,cluster_number+1);
hold on;
plot(A(local_core,1),A(local_core,2),'r.','MarkerSize',8);
% title('搜索maxnb近邻的结果');
end


