% -------------------------------------------------------------------------
%Aim:
%Draw culstering results
% -------------------------------------------------------------------------
%Input:
%A: the data set
%cluster:clustering result
%ncluster:number of clusters
% -------------------------------------------------------------------------
% Written by Dongdong Cheng
% Department of Computer Science, Chongqing University 
% December 2017
function  drawcluster2(A,cluster,ncluster)
[n,d]=size(A);
map = [0.8 0 0
    0  0.8  0
    0 0  0.8
    0.8  0.8 0
    0.8  0 0.8
    0  0.8 0.8
    0 0 0
    0.8 0.8 0.8
    0.6  0 0
    0 0.6 0]
cmap=colormap(map);
axis()
for i=1:n
    if cluster(i)>0
    ic=int8(cluster(i));
    
    x=A(i,1);
    y=A(i,2);
%     z=A(i,3);
    plot(x,y,'o','MarkerSize',5,'MarkerFaceColor',cmap(ic,:),'MarkerEdgeColor',cmap(ic,:));
    hold on;
    else
        x=A(i,1);
        y=A(i,2);
%         z=A(i,3);
        plot(x,y,'k.');%,'MarkerSize',10);
        hold on;
    end
end
% set(gca,'position',[0.05,0.05,0.9,0.9]);
% for i=1:n
%     if cluster(i)~=-1&&cluster(i)<6
%     ic=int8(((cluster(i)+1)*64.)/(6*1.));
%     x=A(i,1);
%     y=A(i,2);
%     plot(x,y,'.','MarkerSize',8,'MarkerFaceColor',cmap(ic,:),'MarkerEdgeColor',cmap(ic,:));
%     hold on;
%     
%     end
% end
% set(gca,'xtick',-inf:inf:inf);set(gca,'ytick',-inf:inf:inf);
% hold off;
end

