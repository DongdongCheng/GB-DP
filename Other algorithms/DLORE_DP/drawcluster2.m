function  drawcluster2(A,cluster,ncluster)
[n,d]=size(A);
cmap=colormap;
for i=1:n
    if cluster(i)>0
    ic=int8(((cluster(i))*64.)/(ncluster*1.));
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
% set(gca,'position',[0.05,0.05,0.9,0.9]);
% hold off;
end

